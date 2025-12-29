"""
csvs_to_fpl_json.py  –  여러 FPL CSV ➜ flight_plans.json / ac_list.json
  • DEM 고도: rasterio.sample() 로 직접 샘플링
  • route ↔ speed 길이 일치 (출발 0.0, 도착 0.0 포함)
  • ac_list.json 의 VP = 첫 비행 출발지 [lat, lon, elev]
"""
from __future__ import annotations
import csv, json, math, re
from pathlib import Path
from typing import Iterable, List, Tuple, Dict

import numpy as np
import rasterio
import pandas as pd

# ───────────── 환경 경로 ─────────────
OFFSET_M = 50.0
_DEM_FILENAMES = [
    "n37_e126_1arc_v3.tif",
    "n37_e127_1arc_v3.tif",
]
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEM_SEARCH_DIRS = [
    Path(__file__).resolve().parents[1] / "Sources",
    _PROJECT_ROOT / "Analysis" / "Sources",
    _PROJECT_ROOT / "Analyzer" / "Sources",
]


def _resolve_dem_files() -> tuple[list[Path], list[str]]:
    resolved: list[Path] = []
    missing: list[str] = []
    for name in _DEM_FILENAMES:
        found = None
        for base in _DEM_SEARCH_DIRS:
            candidate = base / name
            if candidate.exists():
                found = candidate
                break
        if found is None:
            missing.append(name)
        else:
            resolved.append(found)
    return resolved, missing


DEM_FILES, _DEM_MISSING = _resolve_dem_files()

_SCHED_DIR   = Path(__file__).resolve().parents[1]
VP_CSV_PATH  = _SCHED_DIR / "Sources" / "vertiport.csv"

# ───────────── 프로파일 & 상수 ─────────────
DEFAULTS = {
    "B": {"target_vertical_speed": 500, "target_horizontal_speed":   0, "ending_altitude":  15},
    "C": {"target_vertical_speed": 500, "target_horizontal_speed":  60, "ending_altitude": 300},
    "D": {"target_vertical_speed":   0, "target_horizontal_speed": 120, "ending_altitude": 300},
    "E": {"target_vertical_speed": 500, "target_horizontal_speed": 125, "ending_altitude":1000},
    "F": {"target_vertical_speed":   0, "target_horizontal_speed": 130, "ending_altitude":1000},
    "G": {"target_vertical_speed": 500, "target_horizontal_speed": 125, "ending_altitude": 300},
    "H": {"target_vertical_speed": 500, "target_horizontal_speed": 120, "ending_altitude": 300},
    "I": {"target_vertical_speed": 400, "target_horizontal_speed":  60, "ending_altitude":  65},
    "J": {"target_vertical_speed": 300, "target_horizontal_speed":   0, "ending_altitude":   0},
}

# DEFAULTS = {
#     "B": {"target_vertical_speed": 120, "target_horizontal_speed":   0, "ending_altitude":  15},
#     "C": {"target_vertical_speed": 120, "target_horizontal_speed":  20, "ending_altitude": 150},
#     "D": {"target_vertical_speed":   0, "target_horizontal_speed": 40, "ending_altitude": 150},
#     "E": {"target_vertical_speed": 120, "target_horizontal_speed": 45, "ending_altitude": 500},
#     "F": {"target_vertical_speed":   0, "target_horizontal_speed": 45, "ending_altitude": 500},
#     "G": {"target_vertical_speed": 120, "target_horizontal_speed": 40, "ending_altitude": 150},
#     "H": {"target_vertical_speed": 120, "target_horizontal_speed": 40, "ending_altitude": 150},
#     "I": {"target_vertical_speed": 110, "target_horizontal_speed":  20, "ending_altitude":  35},
#     "J": {"target_vertical_speed": 100, "target_horizontal_speed":   0, "ending_altitude":   0},
# }

SEGMENT_ORDER = list(DEFAULTS.keys())
FT_TO_M  = 0.3048
KT_TO_MS = 0.514444
FPM_TO_MPS = FT_TO_M / 60

# 한글→영문 VP 이름
VP_TRANSLATION = {
    "영등포·여의도":"Yeongdeungpo-Yeouido","잠실":"Jamsil","상암·수색":"Sangam-Susaek",
    "용산":"Yongsan","목동":"Mokdong","미아":"Mia","봉천":"Bongcheon",
    "사당·이수":"Sadang-Isu","성수":"Seongsu","연신내·불광":"Yeonsinnae-Bulgwang",
    "천호·길동":"Cheonho-Gildong","광화문":"Gwanghwamun","강남":"Gangnam",
    "가산·대림":"Gasan-Daerim","마곡":"Magok","망우":"Mangu","수서·문정":"Suseo-Munjeong",
}

# ───────────── Helper ─────────────
def _ideal_speed(p: dict) -> float:
    vv = p["target_vertical_speed"] * FPM_TO_MPS
    vh = p["target_horizontal_speed"] * KT_TO_MS
    return round(math.hypot(vv, vh), 2)

def _parse_segments(cells: List[str]) -> Dict[str, List[Tuple[float,float]]]:
    segs: Dict[str, List[Tuple[float,float]]] = {}
    for part in ",".join(cells).split(";"):
        part = part.strip()
        m = re.match(r"([A-J])\s*:\s*([0-9.+-]+)\s+([0-9.+-]+)", part)
        if m:
            s, lon, lat = m.group(1), float(m.group(2)), float(m.group(3))
            segs.setdefault(s, []).append((lat, lon))
    return segs

def _sample_elevation(ds_list: List[rasterio.DatasetReader], lat: float, lon: float) -> float:
    """여러 타일 중 첫 유효 고도 반환; 없으면 NaN"""
    for ds in ds_list:
        l, b, r, t = ds.bounds
        if not (l <= lon <= r and b <= lat <= t):
            continue
        val = next(ds.sample([(lon, lat)]))[0]
        if ds.nodata is None or val != ds.nodata:
            return float(val)
    return float("nan")

def _load_vp_info() -> tuple[dict[str,float], dict[str,Tuple[float,float]]]:
    """영문 VP → (elev, (lat,lon)) dict"""
    if _DEM_MISSING:
        searched = ", ".join(str(d) for d in _DEM_SEARCH_DIRS)
        raise FileNotFoundError(
            "DEM raster files not found: "
            + ", ".join(_DEM_MISSING)
            + f".\nSearched in: {searched}"
        )
    tiles = [rasterio.open(str(p)) for p in DEM_FILES]
    df = pd.read_csv(VP_CSV_PATH, encoding="utf-8-sig") \
           .rename(columns={"Vertiport 명":"kor","위도":"lat","경도":"lon"})
    elevs, coords = {}, {}
    for _, row in df.iterrows():
        eng = VP_TRANSLATION.get(row["kor"], row["kor"])
        lat, lon = row["lat"], row["lon"]
        h = _sample_elevation(tiles, lat, lon)
        elevs[eng]  = 0.0 if np.isnan(h) else round(h, 1)
        coords[eng] = (lat, lon)
    for ds in tiles: ds.close()
    return elevs, coords

# ═════════════ main converter ═════════════
def csvs_to_fpl_json(
    csv_paths: Iterable[str | Path] | str | Path,
    json_path: str | Path,
    region: str = "seoul",
    vp_map: dict[str, str] | None = None
) -> None:

    DEPART_SEGS = {"B", "C", "D"}
    CRUISE_SEGS = {"E", "F"}
    ARRIVE_SEGS = {"G", "H", "I", "J"}

    # ── 1. 준비
    vp_map = {**VP_TRANSLATION, **(vp_map or {})}
    vp_elev, vp_coords = _load_vp_info()

    if isinstance(csv_paths, (str, Path)):
        base = Path(csv_paths)
        files = sorted(base.glob("*.csv")) if base.is_dir() else [base]
    else:
        files = [Path(p) for p in csv_paths]

    fpl, ac_dict = [], {}

    # ── 2. CSV → FPL
    for csv_file in files:
        with open(csv_file, newline="", encoding="utf-8-sig") as fp:
            rdr = csv.reader(fp)
            next(rdr, None)
            for row in rdr:
                vid   = row[1]
                dep   = vp_map.get(row[4], row[4])
                arr   = vp_map.get(row[6], row[6])
                dep_t = row[5];  arr_t = row[7]

                lat0, lon0 = vp_coords[dep]
                lat1, lon1 = vp_coords[arr]
                elev0      = vp_elev[dep] + OFFSET_M   # ← 수정: 출발 지면 + 오프셋
                elev1      = vp_elev[arr] + OFFSET_M   # ← 수정: 도착 지면 + 오프셋


                route = [[lat0, lon0, elev0]]   # 출발 지면
                speed = [0.0]                   # 출발 정지

                segs = _parse_segments(row[8:])
                for seg in SEGMENT_ORDER:
                    pts = segs.get(seg)
                    if not pts:
                        continue

                    alt_m = DEFAULTS[seg]["ending_altitude"] * FT_TO_M   # 프로파일 목표고도(m)

                    # 구간별 기준 + OFFSET
                    if seg in CRUISE_SEGS:               # E, F
                        abs_alt = round(alt_m + OFFSET_M, 1)
                    elif seg in DEPART_SEGS:             # B, C, D
                        abs_alt = round(elev0 + alt_m + OFFSET_M, 1)
                    else:                                # G, H, I, J (도착 단계)
                        abs_alt = round(elev1 + alt_m + OFFSET_M, 1)

                    prof_spd = _ideal_speed(DEFAULTS[seg])

                    if seg == "F":                # F 전구간 저장
                        for lat, lon in pts:
                            route.append([lat, lon, abs_alt])
                            speed.append(prof_spd)
                    else:                         # 나머지는 끝점만
                        lat, lon = pts[-1]
                        route.append([lat, lon, abs_alt])
                        speed.append(prof_spd)

                # 도착 지면 + 속도 0
                if not (math.isclose(route[-1][0], lat1, abs_tol=1e-6) and
                        math.isclose(route[-1][1], lon1, abs_tol=1e-6)):
                    route.append([lat1, lon1, elev1])
                    speed.append(0.0)
                else:
                    speed[-1] = 0.0

                fpl.append({
                    "vehicle_id": vid,
                    "depart_vertiport": dep,
                    "arrival_vertiport": arr,
                    "route": route,
                    "speed": speed,
                    "departure_time": dep_t,
                    "arrival_time": arr_t
                })

                # ac_list : 첫 비행 출발 VP LLA
                ac_dict.setdefault(
                    vid,
                    {"ac_id": vid,
                     "init_pos": [lat0, lon0, elev0],  # LLA
                     "model_type": "TR"}
                )

    # ── 3. JSON 출력
    jp = Path(json_path)
    with open(jp, "w", encoding="utf-8") as wf:
        json.dump({
            "fpl": fpl,
            "control_mode": [{"vehicle_id": v, "mode": 1} for v in ac_dict],
            "speed": 2
        }, wf, ensure_ascii=False, indent=4)

    with open(jp.with_name("ac_list.json"), "w", encoding="utf-8") as wf:
        json.dump({"ac_list": list(ac_dict.values()), "region": region},
                  wf, ensure_ascii=False, indent=4)


# ───────────────────── 테스트 실행 ─────────────────────
if __name__ == "__main__":
    csvs_to_fpl_json("FPL_20250702", "flight_plans.json", region="Seoul")
