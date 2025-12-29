# plugins/FleetOperations/fpl_loader.py
# ------------------------------------------------------------
# Helper utilities for FleetOperations dashboard
# - Locate latest FPL folder and normalise CSV directories
# - Aggregate KPI values (operation window, pax, aircraft, vertiports)
# - Build Gantt chart structures for Gate / FATO / Aircraft views
# ------------------------------------------------------------
from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from datetime import datetime, date, time as dtime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple
from collections import defaultdict

# ---------- Directory helpers -------------------------------------------------
_DATE_DIR_RE = re.compile(r"^\d{8}(_\d+)?$")


def find_latest_fpl_folder(project_root: Optional[Path] = None) -> Optional[Path]:
    root = Path(project_root) if project_root else Path(__file__).resolve().parents[2]
    base = root / "plugins" / "Scheduler" / "FPL_Result"
    if not base.exists():
        return None

    dated = [p for p in base.iterdir() if p.is_dir() and _DATE_DIR_RE.match(p.name)]
    if dated:
        dated.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return dated[0]

    any_dirs = [p for p in base.iterdir() if p.is_dir()]
    if any_dirs:
        any_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return any_dirs[0]
    return None


def _normalize_to_csv_dir(path: Path) -> Path:
    if path.is_file():
        return path.parent
    here_csv = list(path.glob("*.csv"))
    if here_csv:
        return path
    dated = [p for p in path.iterdir() if p.is_dir() and _DATE_DIR_RE.match(p.name)]
    if dated:
        dated.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return dated[0]
    for sub in path.iterdir():
        if sub.is_dir() and list(sub.glob("*.csv")):
            return sub
    return path


def resolve_fpl_container_path(path_like: Optional[str]) -> Optional[Path]:
    if path_like:
        return _normalize_to_csv_dir(Path(path_like))
    return find_latest_fpl_folder()


def _list_csvs(container: Path) -> List[Path]:
    folder = _normalize_to_csv_dir(container)
    return sorted(folder.glob("*.csv"))


# ---------- CSV parsing helpers ----------------------------------------------
_TIME_KEYS = {
    "std", "etd", "sta", "eta", "atot", "aldt", "dep", "arr",
    "dep_time", "arr_time", "departure", "arrival",
    "time", "start", "end", "start_time", "end_time",
    "ETOT", "STA", "ELDT", "ALDT", "STD"
}
_PAX_KEYS = {"pax", "passengers", "passenger", "load", "num_pax", "pax_count"}
_DEP_KEYS = {"dep_port", "departure", "dep", "from", "origin", "oport", "dep_vp", "dep_vertiport", "From", "Origin"}
_ARR_KEYS = {"arr_port", "arrival", "arr", "to", "dest", "destination", "dport", "arr_vp", "arr_vertiport", "To", "Destination"}
_AIRCRAFT_KEYS = {
    "id", "localid", "vehicleid", "uam", "uam_id", "aircraft", "aircraft_id", "acid",
    "uav", "uav_id", "vehicle", "ac_id"
}
_DEP_FATO_KEYS = {"DepFATO_No", "dep_fato", "dep_fato_no", "dep_fato_id"}
_ARR_FATO_KEYS = {"ArrFATO_No", "arr_fato", "arr_fato_no", "arr_fato_id"}
_DEP_GATE_KEYS = {"DepGate_No", "dep_gate", "dep_gate_no", "dep_gate_id"}
_ARR_GATE_KEYS = {"ArrGate_No", "arr_gate", "arr_gate_no", "arr_gate_id"}

_TIME_PATTERNS = [
    "%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S", "%Y.%m.%d %H:%M:%S",
    "%Y-%m-%d %H:%M", "%Y/%m/%d %H:%M", "%Y.%m.%d %H:%M",
    "%H:%M:%S", "%H:%M",
]


def _parse_time_any(s: str) -> Optional[datetime]:
    s = (s or "").strip()
    if not s:
        return None
    for pat in _TIME_PATTERNS:
        try:
            t = datetime.strptime(s, pat)
            if t.year < 2000:
                today = date.today()
                return datetime.combine(today, t.time())
            return t
        except Exception:
            continue
    return None


def _first_key(row: dict, keys: Iterable[str]) -> Optional[str]:
    lower = {k.lower(): k for k in row.keys() if k}
    for k in keys:
        if k.lower() in lower:
            return lower[k.lower()]
    return None


def _open_csv_any(path: Path):
    for enc in ("utf-8-sig", "cp949"):
        try:
            f = open(path, "r", encoding=enc, newline="")
            f.seek(0)
            return f
        except Exception:
            continue
    return open(path, "r", encoding="utf-8", errors="ignore", newline="")


def _num_token(s: str) -> Optional[int]:
    m = re.search(r"(\d+)", s or "")
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _time_field(row: dict, *names: str) -> Optional[datetime]:
    for name in names:
        if name in row and str(row[name]).strip():
            t = _parse_time_any(str(row[name]).strip())
            if t:
                return t
    return None


def _mins(base0: datetime, t: Optional[datetime]) -> Optional[float]:
    if not t:
        return None
    return (t - base0).total_seconds() / 60.0


# ---------- KPI --------------------------------------------------------------
@dataclass
class FplStats:
    op_start: Optional[datetime]
    op_end: Optional[datetime]
    pax_total: int
    ports_count: int
    aircraft_estimate: int


def read_fpl_stats_from_path(path_like: str | Path) -> Optional[FplStats]:
    csvs = _list_csvs(Path(path_like))
    if not csvs:
        return None

    op_start: Optional[datetime] = None
    op_end: Optional[datetime] = None
    pax_total = 0
    ports: set[str] = set()
    ac_nums: List[int] = []
    uam_nums: Set[int] = set()

    for csv_path in csvs:
        with _open_csv_any(csv_path) as f:
            rdr = csv.DictReader(f)
            for row in rdr:
                for k in list(row.keys()):
                    if not k:
                        continue
                    if k.strip().lower() in _TIME_KEYS:
                        t = _parse_time_any(str(row[k]))
                        if t:
                            op_start = t if (op_start is None or t < op_start) else op_start
                            op_end = t if (op_end is None or t > op_end) else op_end
                pax_key = _first_key(row, _PAX_KEYS)
                if pax_key:
                    try:
                        pax_total += int(float(str(row[pax_key]).strip() or "0"))
                    except Exception:
                        pass
                dep_key = _first_key(row, _DEP_KEYS)
                if dep_key and str(row.get(dep_key, "")).strip():
                    ports.add(str(row[dep_key]).strip())
                arr_key = _first_key(row, _ARR_KEYS)
                if arr_key and str(row.get(arr_key, "")).strip():
                    ports.add(str(row[arr_key]).strip())
                ac_key = _first_key(row, _AIRCRAFT_KEYS)
                candidates: List[str] = []
                if ac_key and str(row.get(ac_key, "")).strip():
                    candidates.append(str(row[ac_key]).strip())
                for val in row.values():
                    if isinstance(val, str) and "uam" in val.lower():
                        candidates.append(val.strip())
                for raw in candidates:
                    if not raw:
                        continue
                    m_uam = re.search(r"uam[^\d]*(\d+)", raw, re.IGNORECASE)
                    if m_uam:
                        try:
                            uam_nums.add(int(m_uam.group(1)))
                            continue
                        except Exception:
                            pass
                    n = _num_token(raw)
                    if n is not None:
                        ac_nums.append(n)

    aircraft_estimate = (
        max(uam_nums)
        if uam_nums
        else (max(ac_nums) if ac_nums else len(ac_nums))
    )

    return FplStats(
        op_start=op_start,
        op_end=op_end,
        pax_total=pax_total,
        ports_count=len(ports),
        aircraft_estimate=aircraft_estimate,
    )


# ---------- Gantt structures -------------------------------------------------
@dataclass
class Bar:
    start_min: float
    end_min: float
    row_key: str
    label: str
    meta: dict | None = None


@dataclass
class GanttBundle:
    base0: datetime
    ports: List[str]
    gate_bars_by_port: Dict[str, List[Bar]]
    fato_bars_by_port: Dict[str, List[Bar]]
    ac_bars_by_id: Dict[str, List[Bar]]
    ac_bars_by_port: Dict[str, Dict[str, List[Bar]]]
    time_span_min: float


def _parse_dt_base(rows: List[dict]) -> datetime:
    stds: List[datetime] = []
    for r in rows:
        k = _first_key(r, {"STD", "std"})
        if k:
            t = _parse_time_any(str(r.get(k, "")).strip())
            if t:
                stds.append(t)
    base_d = (stds[0].date() if stds else date.today())
    return datetime.combine(base_d, dtime(0, 0, 0))


def _pick(row: dict, *names: str) -> Optional[str]:
    for n in names:
        for k in row.keys():
            if k and k.strip().lower() == n.lower():
                v = str(row[k]).strip()
                if v:
                    return v
    return None


def build_gantt_from_folder(path_like: str | Path) -> Optional[GanttBundle]:
    csvs = _list_csvs(Path(path_like))
    if not csvs:
        return None

    rows: List[dict] = []
    for p in csvs:
        with _open_csv_any(p) as f:
            rdr = csv.DictReader(f)
            for r in rdr:
                rows.append({k.strip(): (v.strip() if isinstance(v, str) else v) for k, v in r.items() if k})

    if not rows:
        return None

    base0 = _parse_dt_base(rows)
    gate_bars_by_port: Dict[str, List[Bar]] = {}
    fato_bars_by_port: Dict[str, List[Bar]] = {}
    ac_bars_by_id: Dict[str, List[Bar]] = defaultdict(list)
    ac_bars_by_port: Dict[str, Dict[str, List[Bar]]] = defaultdict(lambda: defaultdict(list))
    flights_by_acid: Dict[str, List[dict]] = defaultdict(list)

    t_min = float("inf")
    t_max = float("-inf")

    for r in rows:
        origin = _pick(r, "From", "Origin", "depart_vertiport", "origin", "dep_port", "dep_vp")
        dest = _pick(r, "To", "Destination", "arrive_vertiport", "destination", "arr_port", "arr_vp")
        acid = _pick(r, "id", "LocalID", "VehicleID", "uam", "aircraft", "acid", "uav", "aircraft_id") or "-"

        dep_fato = _pick(r, *list(_DEP_FATO_KEYS)) or "-"
        arr_fato = _pick(r, *list(_ARR_FATO_KEYS)) or "-"
        dep_gate = _pick(r, *list(_DEP_GATE_KEYS)) or "-"
        arr_gate = _pick(r, *list(_ARR_GATE_KEYS)) or "-"

        t_std = _parse_time_any(_pick(r, "STD", "std") or "")
        t_etot = _parse_time_any(_pick(r, "ETOT", "etot") or "")
        t_sta = _parse_time_any(_pick(r, "STA", "sta") or "")
        t_eldt = _parse_time_any(_pick(r, "ELDT", "ALDT") or "")

        t_dep = t_etot or t_std
        t_arr = t_sta or t_eldt

        dep_gate_in = _time_field(r, "DepGateIn", "dep_gate_in")
        dep_gate_out = _time_field(r, "DepGateOut", "dep_gate_out")
        arr_gate_in = _time_field(r, "ArrGateIn", "arr_gate_in")
        arr_gate_out = _time_field(r, "ArrGateOut", "arr_gate_out")
        dep_fato_in = _time_field(r, "DepFATOIn", "dep_fato_in")
        dep_fato_out = _time_field(r, "DepFATOOut", "dep_fato_out")
        arr_fato_in = _time_field(r, "ArrFATOIn", "arr_fato_in")
        arr_fato_out = _time_field(r, "ArrFATOOut", "arr_fato_out")

        flights_by_acid[acid].append({
            "acid": acid,
            "origin": origin or "",
            "destination": dest or "",
            "dep_port": origin or "",
            "arr_port": dest or "",
            "dep_gate": dep_gate,
            "arr_gate": arr_gate,
            "dep_fato": dep_fato,
            "arr_fato": arr_fato,
            "dep_gate_in": dep_gate_in,
            "dep_gate_out": dep_gate_out,
            "dep_fato_in": dep_fato_in,
            "dep_fato_out": dep_fato_out,
            "arr_gate_in": arr_gate_in,
            "arr_gate_out": arr_gate_out,
            "arr_fato_in": arr_fato_in,
            "arr_fato_out": arr_fato_out,
            "t_dep": t_dep,
            "t_arr": t_arr,
        })

        for tx in (t_dep, t_arr, dep_gate_in, dep_gate_out, arr_gate_in, arr_gate_out,
                    dep_fato_in, dep_fato_out, arr_fato_in, arr_fato_out):
            if tx:
                mm = _mins(base0, tx)
                if mm is not None:
                    t_min = min(t_min, mm)
                    t_max = max(t_max, mm)

        # Gate departures
        if origin and dep_gate_in and dep_gate_out:
            gate_bars_by_port.setdefault(origin, []).append(
                Bar(_mins(base0, dep_gate_in), _mins(base0, dep_gate_out), f"Gate {dep_gate}",
                    "GATE_DEP", {
                        "acid": acid, "from": origin, "to": dest, "gate": dep_gate,
                        "phase": "GATE_DEP", "start_dt": dep_gate_in, "end_dt": dep_gate_out,
                    })
            )
        elif origin and t_dep:
            gate_bars_by_port.setdefault(origin, []).append(
                Bar(_mins(base0, t_dep - timedelta(minutes=1)), _mins(base0, t_dep), f"Gate {dep_gate}",
                    "GATE_DEP", {
                        "acid": acid, "from": origin, "to": dest, "gate": dep_gate,
                        "phase": "GATE_DEP", "start_dt": t_dep - timedelta(minutes=1), "end_dt": t_dep,
                    })
            )

        # Gate arrivals
        if dest and arr_gate_in and arr_gate_out:
            gate_bars_by_port.setdefault(dest, []).append(
                Bar(_mins(base0, arr_gate_in), _mins(base0, arr_gate_out), f"Gate {arr_gate}",
                    "GATE_ARR", {
                        "acid": acid, "from": origin, "to": dest, "gate": arr_gate,
                        "phase": "GATE_ARR", "start_dt": arr_gate_in, "end_dt": arr_gate_out,
                    })
            )
        elif dest and t_arr:
            gate_bars_by_port.setdefault(dest, []).append(
                Bar(_mins(base0, t_arr), _mins(base0, t_arr + timedelta(minutes=6)), f"Gate {arr_gate}",
                    "GATE_ARR", {
                        "acid": acid, "from": origin, "to": dest, "gate": arr_gate,
                        "phase": "GATE_ARR", "start_dt": t_arr, "end_dt": t_arr + timedelta(minutes=6),
                    })
            )

        # FATO departures / arrivals
        if origin and dep_fato_in and dep_fato_out:
            fato_bars_by_port.setdefault(origin, []).append(
                Bar(_mins(base0, dep_fato_in), _mins(base0, dep_fato_out), f"FATO {dep_fato}",
                    "FATO_DEP", {
                        "acid": acid, "from": origin, "to": dest, "fato": dep_fato,
                        "phase": "FATO_DEP", "start_dt": dep_fato_in, "end_dt": dep_fato_out,
                    })
            )
        elif origin and t_dep:
            fato_bars_by_port.setdefault(origin, []).append(
                Bar(_mins(base0, t_dep), _mins(base0, t_dep + timedelta(seconds=30)), f"FATO {dep_fato}",
                    "FATO_DEP", {
                        "acid": acid, "from": origin, "to": dest, "fato": dep_fato,
                        "phase": "FATO_DEP", "start_dt": t_dep, "end_dt": t_dep + timedelta(seconds=30),
                    })
            )

        if dest and arr_fato_in and arr_fato_out:
            fato_bars_by_port.setdefault(dest, []).append(
                Bar(_mins(base0, arr_fato_in), _mins(base0, arr_fato_out), f"FATO {arr_fato}",
                    "FATO_ARR", {
                        "acid": acid, "from": origin, "to": dest, "fato": arr_fato,
                        "phase": "FATO_ARR", "start_dt": arr_fato_in, "end_dt": arr_fato_out,
                    })
            )
        elif dest and t_arr:
            fato_bars_by_port.setdefault(dest, []).append(
                Bar(_mins(base0, t_arr), _mins(base0, t_arr + timedelta(seconds=30)), f"FATO {arr_fato}",
                    "FATO_ARR", {
                        "acid": acid, "from": origin, "to": dest, "fato": arr_fato,
                        "phase": "FATO_ARR", "start_dt": t_arr, "end_dt": t_arr + timedelta(seconds=30),
                    })
            )


    def _sort_key(rec: dict) -> float:
        for key in ("dep_fato_out", "dep_fato_in", "arr_fato_in", "arr_gate_in"):
            dtv = rec.get(key)
            if dtv:
                val = _mins(base0, dtv)
                if val is not None:
                    return val
        return float("inf")

    for acid, flist in flights_by_acid.items():
        if not flist:
            continue
        flist.sort(key=_sort_key)
        for idx, rec in enumerate(flist):
            arr_port = rec.get("arr_port") or ""
            arr_start_dt = rec.get("arr_fato_in") or rec.get("arr_gate_in") or rec.get("arr_fato_out") or rec.get("arr_gate_out")
            if not arr_port or not arr_start_dt:
                continue

            next_rec = None
            for candidate in flist[idx + 1:]:
                dep_port = candidate.get("dep_port") or ""
                if dep_port and dep_port == arr_port:
                    next_rec = candidate
                    break
            if not next_rec:
                continue

            dep_dt = next_rec.get("dep_fato_out") or next_rec.get("dep_fato_in") or next_rec.get("dep_gate_out") or next_rec.get("dep_gate_in")
            if not dep_dt or dep_dt <= arr_start_dt:
                continue

            start_min = _mins(base0, arr_start_dt)
            end_min = _mins(base0, dep_dt)
            if start_min is None or end_min is None or end_min <= start_min:
                continue

            meta = {
                "acid": acid,
                "from": rec.get("origin") or "",
                "to": next_rec.get("destination") or "",
                "arrived_from": rec.get("origin") or "",
                "arrive_port": arr_port,
                "arr_fato_no": rec.get("arr_fato") or "",
                "arr_gate_no": rec.get("arr_gate") or "",
                "arr_fato_in": rec.get("arr_fato_in"),
                "arr_fato_out": rec.get("arr_fato_out"),
                "arr_gate_in": rec.get("arr_gate_in"),
                "arr_gate_out": rec.get("arr_gate_out"),
                "depart_port": next_rec.get("dep_port") or "",
                "depart_to": next_rec.get("destination") or "",
                "dep_gate_no": next_rec.get("dep_gate") or "",
                "dep_fato_no": next_rec.get("dep_fato") or "",
                "dep_gate_in": next_rec.get("dep_gate_in"),
                "dep_gate_out": next_rec.get("dep_gate_out"),
                "dep_fato_in": next_rec.get("dep_fato_in"),
                "dep_fato_out": next_rec.get("dep_fato_out"),
                "start_dt": arr_start_dt,
                "end_dt": dep_dt,
            }
            bar = Bar(start_min, end_min, acid, "AC_GROUND", meta)
            ac_bars_by_id[acid].append(bar)
            ac_bars_by_port[arr_port][acid].append(bar)
            t_min = min(t_min, start_min)
            t_max = max(t_max, end_min)

    # Clean row_keys for gates with missing number: fallback to sequential assignment
    for port, bars in list(gate_bars_by_port.items()):
        if any(not bar.row_key or bar.row_key == '-' for bar in bars):
            sorted_bars = sorted(bars, key=lambda b: (b.start_min, b.end_min))
            assigned: List[Bar] = []
            gate_end: List[float] = []
            for bar in sorted_bars:
                placed = False
                for gi, tend in enumerate(gate_end):
                    if tend <= bar.start_min:
                        gate_end[gi] = bar.end_min
                        new_id = str(gi + 1)
                        meta = {**(bar.meta or {}), "gate": new_id}
                        assigned.append(Bar(bar.start_min, bar.end_min, new_id, bar.label, meta))
                        placed = True
                        break
                if not placed:
                    gate_end.append(bar.end_min)
                    new_id = str(len(gate_end))
                    meta = {**(bar.meta or {}), "gate": new_id}
                    assigned.append(Bar(bar.start_min, bar.end_min, new_id, bar.label, meta))
            gate_bars_by_port[port] = assigned
        else:
            gate_bars_by_port[port] = [
                Bar(bar.start_min, bar.end_min, str(bar.row_key), bar.label,
                    {**(bar.meta or {}), "gate": str(bar.row_key)})
                for bar in bars
            ]

    ports = sorted(set(gate_bars_by_port.keys()) | set(fato_bars_by_port.keys()) | set(ac_bars_by_port.keys()))
    if not ports:
        ports = ["-"]

    if t_min == float("inf") or t_max == float("-inf"):
        t_min, t_max = 0.0, 60.0
    else:
        span_margin = 15.0
        t_min -= span_margin
        t_max += span_margin

    span = max(10.0, t_max - t_min)

    ac_bars_by_id = {acid: bars for acid, bars in ac_bars_by_id.items()}
    ac_bars_by_port = {
        port: {acid: bars for acid, bars in acids.items()}
        for port, acids in ac_bars_by_port.items()
    }

    return GanttBundle(
        base0=base0,
        ports=ports,
        gate_bars_by_port=gate_bars_by_port,
        fato_bars_by_port=fato_bars_by_port,
        ac_bars_by_id=ac_bars_by_id,
        ac_bars_by_port=ac_bars_by_port,
        time_span_min=span,
    )
