# GateResources.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Iterable
import math, numpy as np, pandas as pd

# WGS84
_WGS84_A = 6378137.0
_WGS84_F = 1.0/298.257223563
_WGS84_E2 = _WGS84_F*(2.0-_WGS84_F)
# ------------------------ 기본 상수 ------------------------
GATE_COUNT_DEFAULT      = 8            # 기본 게이트 개수
TAXI_IN_MIN_DEFAULT     = 5            # FATO → GATE 이동
TAXI_OUT_MIN_DEFAULT    = 5            # GATE → FATO 이동
TAKEOFF_MIN_DEFAULT     = 2            # FATO 이륙 점유
LANDING_MIN_DEFAULT     = 2            # FATO 착륙 점유
PREP_TIME_MIN_DEFAULT   = 6            # 게이트 서비스(하차+정리+탑승)

# ★ 모든 버티포트에 공통으로 잠글 게이트 번호(1-based). 예: [8]이면 전 포트 GATE#8 잠금
LOCKED_GATES_ALL_PORTS: List[int] = [8]   # 필요시 [8] 로 설정하세요.

# --- 추가 상수 ---
GATE_EXIT_LINGER_SEC    = 30           # 택시아웃 시작 이후 게이트 잠금 유지 (초)
NEW_DEP_PREOCCUPY_MIN   = 1.0          # 새 기체 출발 전 게이트 점유 시간 (분)
GATE_DEP_PRE_LABEL = "GATE_DEP_PRE"   # 출발 직전 1분 사전점유(=UAM 점유)
GATE_ARR_SVC_LABEL = "GATE_ARR_SVC"   # 도착 6분 서비스(=UAM 점유)
GATE_LOCK_LABEL    = "GATE_LOCK"      # 출발 후 릴린저 잠금(=UAM 비점유)

def _geodetic_to_ecef(lat_deg, lon_deg, h=0.0):
    lat=math.radians(lat_deg); lon=math.radians(lon_deg)
    s=math.sin(lat); c=math.cos(lat)
    N=_WGS84_A/math.sqrt(1.0-_WGS84_E2*s*s)
    X=(N+h)*c*math.cos(lon); Y=(N+h)*c*math.sin(lon); Z=((1.0-_WGS84_E2)*N+h)*s
    return np.array([X,Y,Z], float)

def _ecef_to_geodetic(x, y, z):
    a=_WGS84_A; e2=_WGS84_E2; b=a*math.sqrt(1-e2); ep2=(a*a-b*b)/(b*b)
    p=math.hypot(x,y); th=math.atan2(a*z, b*p); lon=math.atan2(y,x)
    lat=math.atan2(z+ep2*b*math.sin(th)**3, p - e2*a*math.cos(th)**3)
    N=a/math.sqrt(1-e2*math.sin(lat)**2); h=p/math.cos(lat)-N
    return math.degrees(lat), math.degrees(lon), h

def _ecef_to_enu(xyz, lat0, lon0, h0=0.0):
    x0,y0,z0=_geodetic_to_ecef(lat0,lon0,h0)
    dx,dy,dz=xyz[0]-x0, xyz[1]-y0, xyz[2]-z0
    lat0,lon0=math.radians(lat0), math.radians(lon0)
    slat, clat = math.sin(lat0), math.cos(lat0)
    slon, clon = math.sin(lon0), math.cos(lon0)   # ← 여기 cos(lon0) 추가
    e = -slon*dx + clon*dy
    n = -slat*clon*dx - slat*slon*dy + clat*dz
    u =  clat*clon*dx + clat*slon*dy + slat*dz
    return np.array([e,n,u], float)

def _enu_to_ecef(e, n, u, lat0, lon0, h0=0.0):
    lat0,lon0=math.radians(lat0), math.radians(lon0)
    slat,clat=math.sin(lat0),math.cos(lat0); slon,clon=math.sin(lon0)
    dx = -slon*e - slat*clon*n + clat*clon*u
    dy =  clon*e - slat*slon*n + clat*slon*u
    dz =            clat*n      + slat*u
    x0,y0,z0=_geodetic_to_ecef(math.degrees(lat0), math.degrees(lon0), h0)
    return np.array([x0+dx, y0+dy, z0+dz], float)

# GCP (coordinateUR.py와 동일)
_GCP = [
    ("SEOUL_CITY_HALL", 37.566831, 126.978445, (  217063.379391, -1013419.868553, -214836.157270)),
    ("GIMPO_VERTIPORT", 37.563535, 126.791805, (-1431694.121809,  -979217.553043, -221157.611723)),
    ("YEOUIDO",         37.525491, 126.921490, ( -285967.804749,  -555936.730254, -221891.377864)),
    ("CHEONHO_4WAY",    37.538658, 127.123442, ( 1498921.351312,  -703218.244679, -223002.980913)),
    ("SUSEO_IC",        37.483396, 127.025980, (  638658.814081,   -89106.174219, -221891.377864)),
    ("IMUN_YARD",       37.603812, 127.068543, ( 1013805.274583, -1424832.145135, -223125.441155)),
    ("YEONSINNAE",      37.618936, 126.921349, ( -287516.699663, -1593208.781614, -222297.078953)),
]
_CITY = next(g for g in _GCP if g[0]=="SEOUL_CITY_HALL")
_LAT0,_LON0 = _CITY[1], _CITY[2]
_CITY_UE = np.array(_CITY[3], float)  # 절대 UE(cm)

# WGS84(ENU) → UE(cm) 아핀: [X;Y;Z] = M[3x3]*[E N U]^T + b
def _fit_affine():
    rows=[]; Xs=[]; Ys=[]; Zs=[]
    for _,lat,lon,ue in _GCP:
        e,n,u = _ecef_to_enu(_geodetic_to_ecef(lat,lon,0.0), _LAT0,_LON0,0.0)
        rows.append([e,n,u,1.0]); Xs.append(ue[0]); Ys.append(ue[1]); Zs.append(ue[2])
    A=np.asarray(rows,float); X=np.asarray(Xs,float); Y=np.asarray(Ys,float); Z=np.asarray(Zs,float)
    Cx, *_ = np.linalg.lstsq(A, X, rcond=None)
    Cy, *_ = np.linalg.lstsq(A, Y, rcond=None)
    Cz, *_ = np.linalg.lstsq(A, Z, rcond=None)
    M = np.vstack([Cx, Cy, Cz])  # 3x4
    return M[:,:3], M[:,3]       # M3, b
_M3, _B = _fit_affine()
_M3_INV = np.linalg.inv(_M3)

def ue_ch_to_lonlat(x_cm: float, y_cm: float, z_cm: float):
    """CH-Origin UE(cm) → (lon, lat)."""
    ue_abs = np.array([x_cm, y_cm, z_cm], float) + _CITY_UE     # 절대 UE(cm)
    enu    = _M3_INV @ (ue_abs - _B)                            # ENU(m)
    ecef   = _enu_to_ecef(*enu, _LAT0, _LON0, 0.0)              # → ECEF
    lat, lon, _ = _ecef_to_geodetic(*ecef)
    return float(lon), float(lat)

# csv 로더 & 조회
def load_resources(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path, encoding="utf-8-sig")

def find_lonlat(df: pd.DataFrame, port: str, kind: str, num: int):
    lab = f"{kind.upper()} {int(num)}"
    row = df[(df["Vertiport"]==port) & (df["Label"]==lab)].iloc[0]

    # 1) pt_lon_deg / pt_lat_deg가 있으면 그대로 사용 (변환 생략)
    if "pt_lon_deg" in df.columns and "pt_lat_deg" in df.columns:
        lon = row.get("pt_lon_deg"); lat = row.get("pt_lat_deg")
        try:
            if pd.notna(lon) and pd.notna(lat):
                return float(lon), float(lat)
        except Exception:
            pass  # 값이 비정상일 때만 폴백

    # 2) 폴백: 기존 UE(cm, CH-origin) → WGS84 역변환
    return ue_ch_to_lonlat(row["X_cm"], row["Y_cm"], row["Z_cm"])

# km 환산(경도/위도 → dx,dy[km] @ 기준위도)
def _km_per_deg_lat(phi): return (111.13209 - 0.56605*math.cos(2*phi) + 0.00120*math.cos(4*phi))
def _km_per_deg_lon(phi): return (111.41513*math.cos(phi) - 0.09455*math.cos(3*phi) + 0.00012*math.cos(5*phi))
def lonlat_to_xy_m(lon, lat, ref_lon, ref_lat, ref_x_km, ref_y_km):
    φ = math.radians(ref_lat)
    dx_km = (lon - ref_lon) * _km_per_deg_lon(φ)
    dy_km = (lat - ref_lat) * _km_per_deg_lat(φ)
    return (ref_x_km + dx_km) * 1000.0, (ref_y_km + dy_km) * 1000.0


# ------------------------ 유틸 ------------------------
def _clipped_length(a: float, b: float, lo: float, hi: float) -> float:
    """구간 [a,b]와 [lo,hi]의 교집합 길이(음수 방지)."""
    lo2 = max(a, lo); hi2 = min(b, hi)
    return max(0.0, hi2 - lo2)

# ------------------------ 데이터 구조 ------------------------
@dataclass
class Interval:
    """단일 리소스 점유 구간(분 단위)."""
    start: float
    end:   float
    flight_id: Optional[str] = None
    label: str = ""  # "GATE" | "FATO_TKO" | "FATO_LDG"

@dataclass
@dataclass
class Unit:
    uid: int
    schedule: List[Interval] = field(default_factory=list)
    next_free: float = 0.0

    def append(self, start: float, duration: float,
               flight_id: Optional[str], label: str) -> Interval:
        """(기존) 마지막 뒤에만 추가 – 과거 삽입 불가"""
        s = max(self.next_free, start)
        end = s + duration
        iv = Interval(s, end, flight_id=flight_id, label=label)
        self.schedule.append(iv)
        self.next_free = end
        return iv

    # ── [NEW] 스케줄을 시작시각 기준으로 정렬
    def _sort(self) -> None:
        if len(self.schedule) > 1 and any(self.schedule[i].start > self.schedule[i+1].start
                                          for i in range(len(self.schedule)-1)):
            self.schedule.sort(key=lambda iv: iv.start)

    # ── [NEW] earliest_start 이후 '겹치지 않는' 가장 이른 시각을 찾는다
    def earliest_slot(self, earliest_start: float, duration: float) -> float:
        self._sort()
        t = max(0.0, earliest_start)
        for iv in self.schedule:
            if t + duration <= iv.start - 1e-9:   # 다음 블록 시작 전이면 OK
                return t
            if t < iv.end:                        # 겹치면 바로 뒤로 민다
                t = iv.end
        return t                                   # 꼬리 뒤로

    # ── [NEW] 계산된 시각에 '삽입' (정렬 유지, next_free는 최대 종료시각으로 유지)
    def insert(self, start: float, duration: float,
               flight_id: Optional[str], label: str) -> Interval:
        t = self.earliest_slot(start, duration)
        iv = Interval(t, t + duration, flight_id=flight_id, label=label)
        self.schedule.append(iv)
        self._sort()
        if iv.end > self.next_free:
            self.next_free = iv.end
        return iv

@dataclass
class Group:
    """동일 기능 리소스 묶음(예: GATE N개, FATO-TKO M개)."""
    name: str
    count: int
    locked: set[int] = field(default_factory=set)   # ★ 0-based uid 집합
    units: List[Unit] = field(init=False)

    def __post_init__(self):
        self.units = [Unit(uid=i) for i in range(self.count)]

    def set_locked(self, uids: Iterable[int]) -> None:
        self.locked = set(int(u)-1 for u in uids if int(u) >= 1)

    def allocate(self, earliest_start: float, duration: float,
                 flight_id: Optional[str], label: str,
                 preferred_uid: Optional[int] = None) -> Tuple[int, Interval]:
        """가장 빨리 '시작할 수 있는' 유닛에 배정 (동률이면 낮은 번호, 선호 유닛 우선)"""
        if preferred_uid is not None and 0 <= preferred_uid < len(self.units) and preferred_uid not in self.locked:
            unit = self.units[preferred_uid]
            start = unit.earliest_slot(earliest_start, duration)
            iv = unit.insert(start, duration, flight_id, label)
            return preferred_uid, iv
        best_uid = None
        best_time = float("inf")
        for u in self.units:
            if u.uid in self.locked:
                continue
            t = u.earliest_slot(earliest_start, duration)   # ← [NEW]
            if t < best_time or (t == best_time and (best_uid is None or u.uid < best_uid)):
                best_time = t
                best_uid  = u.uid
        if best_uid is None:
            best_uid = 0
        iv = self.units[best_uid].insert(best_time, duration, flight_id, label)  # ← [NEW]
        return best_uid, iv

    def next_free_time(self) -> float:
        avail = [u.next_free for u in self.units if u.uid not in self.locked]
        return min(avail) if avail else 0.0


# ------------------------ 포트 상태 ------------------------
@dataclass
class PortState:
    port_id: str
    gates: Group
    fato_tko: Group
    fato_ldg: Group
    taxi_in_min: float
    taxi_out_min: float
    takeoff_min: float
    landing_min: float
    prep_time_min: float

    # flight_id -> (gate_unit_id, last_interval_index) : (JIT과 상관없이) 마지막 GATE 블록 보정용
    _gate_last_interval_idx: Dict[str, Tuple[int, int]] = field(default_factory=dict)

    def schedule_arrival(self, touchdown_time: float, flight_id: Optional[str] = None) -> Dict[str, float]:
        # 1) 착륙 FATO(2m)
        fato_ldg_id, fato_ldg_iv = self.fato_ldg.allocate(
            earliest_start=touchdown_time,
            duration=self.landing_min,
            flight_id=flight_id,
            label="FATO_LDG",
        )
        # 2) Taxi-in(5m)
        taxi_in_start = fato_ldg_iv.end
        taxi_in_end   = taxi_in_start + self.taxi_in_min

        # 3) Gate 서비스(6m) – 게이트가 비는 가장 이른 시각에 시작
        gate_id, gate_iv = self.gates.allocate(
            earliest_start=taxi_in_end,
            duration=self.prep_time_min,
            flight_id=flight_id,
            label="GATE_ARR_SVC",     # ← 도착 6분 라벨(이미 라벨 분리 안했다면 "GATE"여도 무방)
        )
        unit = self.gates.units[gate_id]
        self._gate_last_interval_idx[flight_id or f"f-{id(gate_iv)}"] = (gate_id, len(unit.schedule) - 1)

        # 4) ★ 게이트-아웃 락(30초) 추가 — 도착 6분 종료 ‘직후’부터 잠금
        if GATE_EXIT_LINGER_SEC > 0:
            unit.insert(
                start=gate_iv.end,
                duration=GATE_EXIT_LINGER_SEC / 60.0,   # 30초 → 0.5분
                flight_id=None,
                label="GATE_LOCK"                       # ← UAM 비점유 잠금
            )

        return {
            "fato_ldg_id":    fato_ldg_id,
            "fato_ldg_start": fato_ldg_iv.start,
            "fato_ldg_end":   fato_ldg_iv.end,
            "taxi_in_start":  taxi_in_start,
            "taxi_in_end":    taxi_in_end,
            "gate_id":        gate_id,        # 0-based
            "gate_start":     gate_iv.start,  # Gate-in
            "gate_end":       gate_iv.end,    # Gate-out(6분 종료)
        }


    def schedule_departure(self, etot: float, flight_id: Optional[str] = None,
                       std_min: Optional[float] = None,
                       departure_policy: str = "JIT",
                       preferred_gate: Optional[int] = None) -> Dict[str, float]:
        # 기준 시각
        base_start = (std_min if std_min is not None else (etot - self.taxi_out_min))

        # ① 출발 사전점유(1분) = UAM 점유
        gate_id, pre_iv = self.gates.allocate(
            earliest_start=max(0.0, base_start - NEW_DEP_PREOCCUPY_MIN),
            duration=NEW_DEP_PREOCCUPY_MIN,
            flight_id=flight_id,
            label=GATE_DEP_PRE_LABEL,   # "GATE_DEP_PRE"
            preferred_uid=preferred_gate,
        )

        if departure_policy.upper() == "HOLD":
            # ②A HOLD: STD에 게이트 반납(택시아웃 시작), FATO가 비면 TKO
            taxi_out_start = max(pre_iv.end, base_start)
            earliest_tko   = max(taxi_out_start + self.taxi_out_min,
                                self.fato_tko.next_free_time())
            # ③ FATO 이륙 배정 (← 여기 채워야 했던 부분)
            fato_tko_id, tko_iv = self.fato_tko.allocate(
                earliest_start=earliest_tko,
                duration=self.takeoff_min,
                flight_id=flight_id,
                label="FATO_TKO",
            )

        else:
            # ②B JIT: FATO 가용/ETOT에 맞춰 택시아웃을 늦춤
            earliest_tko   = max(etot,
                                pre_iv.end + self.taxi_out_min,
                                self.fato_tko.next_free_time())
            # ③ FATO 이륙 배정 (← 여기 채워야 했던 부분)
            fato_tko_id, tko_iv = self.fato_tko.allocate(
                earliest_start=earliest_tko,
                duration=self.takeoff_min,
                flight_id=flight_id,
                label="FATO_TKO",
            )
            taxi_out_start = tko_iv.start - self.taxi_out_min

        # ④ 릴린저(출발 후 X초)는 '잠금'으로 별도 interval (UAM 비점유)
        self.gates.units[gate_id].insert(
            start=taxi_out_start,
            duration=GATE_EXIT_LINGER_SEC / 60.0,
            flight_id=None,
            label=GATE_LOCK_LABEL,  # "GATE_LOCK"
        )

        return {
            "gate_id":        gate_id,
            "fato_tko_id":    fato_tko_id,
            "gate_release":   taxi_out_start + GATE_EXIT_LINGER_SEC/60.0,
            "taxi_out_start": taxi_out_start,
            "taxi_out_end":   taxi_out_start + self.taxi_out_min,
            "fato_tko_start": tko_iv.start,
            "fato_tko_end":   tko_iv.end,
        }



# ------------------------ 네트워크 상태 ------------------------
class NetworkState:
    """
    포트별 리소스 상태(게이트/FATO)를 보관하고 고수준 API 제공.
    """
    def __init__(self,
                 gate_count_by_port: Optional[Dict[str, int]] = None,
                 taxi_in_min:  float = TAXI_IN_MIN_DEFAULT,
                 taxi_out_min: float = TAXI_OUT_MIN_DEFAULT,
                 takeoff_min:  float = TAKEOFF_MIN_DEFAULT,
                 landing_min:  float = LANDING_MIN_DEFAULT,
                 prep_time_min:float = PREP_TIME_MIN_DEFAULT,
                 tko_count_by_port: Optional[Dict[str, int]] = None,
                 ldg_count_by_port: Optional[Dict[str, int]] = None,
                 locked_gates_by_port: Optional[Dict[str, Iterable[int]]] = None):
        self.taxi_in_min   = taxi_in_min
        self.taxi_out_min  = taxi_out_min
        self.takeoff_min   = takeoff_min
        self.landing_min   = landing_min
        self.prep_time_min = prep_time_min

        self.ports: Dict[str, PortState] = {}

        # 게이트/FATO 유닛 구성
        def _mk_gates(n: int, locked: Iterable[int]) -> Group:
            g = Group("GATE", n); g.set_locked(locked); return g
        def _mk(name: str, n: int) -> Group:
            return Group(name, n)

        locked_all = (locked_gates_by_port or {}).get("*", [])
        for pid, n_gates in (gate_count_by_port or {}).items():
            gates = _mk_gates(n_gates, locked=locked_all)
            fato_tko = _mk("FATO_TKO", (tko_count_by_port or {}).get(pid, 1))
            fato_ldg = _mk("FATO_LDG", (ldg_count_by_port or {}).get(pid, 1))
            self.ports[pid] = PortState(pid, gates, fato_tko, fato_ldg,
                                        self.taxi_in_min, self.taxi_out_min,
                                        self.takeoff_min, self.landing_min,
                                        self.prep_time_min)

    def _get_port(self, port_id: str) -> PortState:
        if port_id not in self.ports:
            self.ports[port_id] = PortState(
                port_id,
                Group("GATE", GATE_COUNT_DEFAULT),
                Group("FATO_TKO", 1),
                Group("FATO_LDG", 1),
                self.taxi_in_min, self.taxi_out_min,
                self.takeoff_min, self.landing_min,
                self.prep_time_min
            )
        return self.ports[port_id]

    # ---------------------- 고수준 API ----------------------
    def arrival_flow(self, port_id: str, touchdown_time: float, flight_id: Optional[str] = None) -> Dict[str, float]:
        return self._get_port(port_id).schedule_arrival(touchdown_time, flight_id=flight_id)

    def departure_flow(self, port_id: str, etot: float, flight_id: Optional[str] = None,
                       std_min: Optional[float] = None, departure_policy: str = "JIT",
                       preferred_gate: Optional[int] = None) -> Dict[str, float]:
        return self._get_port(port_id).schedule_departure(etot, flight_id=flight_id,
                                                          std_min=std_min, departure_policy=departure_policy,
                                                          preferred_gate=preferred_gate)

    # ---------------------- 리포트 ----------------------
    def gate_utilization(self, port_id: str, until: Optional[float] = None,
                     labels=("GATE_DEP_PRE","GATE_ARR_SVC")) -> float:
        """[0,1] 비율. until이 주어지면 해당 시각까지의 점유율."""
        ps = self._get_port(port_id)
        total = 0.0
        busy  = 0.0
        for u in ps.gates.units:
            total += (until if until is not None else u.next_free)
            for iv in u.schedule:
                if iv.label in labels:
                    busy += _clipped_length(iv.start, iv.end, 0.0, (until if until is not None else u.next_free))
            
        return (busy / total) if total > 0 else 0.0

    def get_gate_schedule(self, port_id: str, labels=("GATE_DEP_PRE","GATE_ARR_SVC","GATE_LOCK")):
        ps = self._get_port(port_id)
        rows = []
        for u in ps.gates.units:
            for iv in u.schedule:
                if iv.label in labels:
                    rows.append({
                        'gate': u.uid + 1,
                        'start': iv.start,
                        'end': iv.end,
                        'flight_id': iv.flight_id or '',
                        'label': iv.label,
                    })
        rows.sort(key=lambda r: (r['start'], r['end']))
        return rows
