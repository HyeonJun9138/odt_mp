# -*- coding: utf-8 -*-
"""
sitl_coord_transform.py
- WGS84(위경도) ↔ UE(cm) 3D 아핀(최소제곱) 변환 (레퍼런스: 시청/김포/여의도/천호/수서/이문/연신내)
- AirSim 연동용 NED[m] 변환까지 한 번에 제공 (Z: Down, AirSim 표준)
- 김포 레이아웃(GATE/FATO/TAXI) 오프셋/회전 유틸 포함
  ※ 이 모듈은 GUI 의존성이 없으며, numpy만 필요합니다.
"""

from __future__ import annotations
from typing import Dict, Tuple, Sequence, Optional
import math
import numpy as np
from pathlib import Path
import csv

# ========================= 상수/보조 =========================
_RES_VP_CACHE: tuple[Path | None, dict] | None = None

WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)

JSON_XY_DIV = 100.0  # cm → m
# AirSim expects metres; UE coordinates are stored in centimetres.
# Earlier this was set to 1000 which effectively divided altitude by 10
# and caused aircraft to remain near the ground. Keep XY/Z scaling consistent.
JSON_Z_DIV  = 100.0   # cm → m   (엑셀/GUI와 동일 스케일로 통일)

def find_resources_vp(csv_path: str | None = None) -> Path | None:
    """
    resources_vp.csv 추정 경로 탐색 (명시 경로 우선).
    - <sandbox>/mnt/data/resources_vp.csv (개발/테스트)
    - 현재 모듈 디렉터리 / 상위 디렉터리 / CWD 에서 검색
    """
    cands: list[Path] = []
    
    # ★ 사용자가 준 절대 경로 우선
    if csv_path:
        cands.append(Path(csv_path))
    # ★ SITL/resource/ 기본 탐색지 추가
    base = Path(__file__).resolve().parent
    cands.append(base / "resource" / "resources_vp.csv")  # ← 추가
    
    if csv_path:
        cands.append(Path(csv_path))
    cands.append(Path("/mnt/data/resources_vp.csv"))                 # 개발 환경
    base = Path(__file__).resolve().parent
    cands.append(base / "resources_vp.csv")                          # SITL/
    cands.append(base.parent / "resources_vp.csv")                   # 프로젝트 루트
    cands.append(Path.cwd() / "resources_vp.csv")
    for p in cands:
        try:
            if p.is_file():
                return p
        except Exception:
            continue
    return None

def load_resources_vp(csv_path: str | None = None) -> dict:
    """
    resources_vp.csv을 읽어 (Vertiport, Label) → dict 를 캐시에 적재 후 반환.
    Label은 대문자·trim, Vertiport는 trim 만 적용.
    """
    global _RES_VP_CACHE
    p = find_resources_vp(csv_path)
    mapping: dict[tuple[str, str], dict] = {}
    if p:
        with p.open("r", encoding="utf-8-sig", newline="") as f:
            rdr = csv.DictReader(f)
            for row in rdr:
                vp = str(row.get("Vertiport", "")).strip()
                lb = str(row.get("Label", "")).strip().upper()
                if not vp or not lb:
                    continue

                def _num(*names):
                    for n in names:
                        v = row.get(n)
                        if v is None:
                            continue
                        s = str(v).strip()
                        if not s:
                            continue
                        try:
                            return float(s)
                        except Exception:
                            pass
                    return None

                mapping[(vp, lb)] = {
                    "Z_m":     _num("Z_m", "Z_M"),
                    "Z_cm":    _num("Z_cm", "Z_CM"),
                    "Yaw_deg": _num("Yaw_deg", "Angle_deg", "Yaw", "Angle"),
                    "X_m":     _num("X_m"),
                    "Y_m":     _num("Y_m"),
                }
    _RES_VP_CACHE = (p, mapping)
    return mapping


def lookup_vp_label_alt_m(vertiport: str, label: str) -> float | None:
    """
    (Vertiport, Label) 조합의 고도 Z(m)를 반환. 없으면 None.
    """
    global _RES_VP_CACHE
    if _RES_VP_CACHE is None:
        load_resources_vp(None)
    if _RES_VP_CACHE is None:
        return None
    _, mapping = _RES_VP_CACHE
    key = (str(vertiport).strip(), str(label).strip().upper())
    rec = mapping.get(key)
    if not rec:
        return None
    z = rec.get("Z_m")
    return float(z) if z is not None else None


def _deg2rad(d: float) -> float: return d * math.pi / 180.0

def _geodetic_to_ecef(lat_deg: float, lon_deg: float, h_m: float = 0.0):
    lat = _deg2rad(lat_deg); lon = _deg2rad(lon_deg)
    s = math.sin(lat); c = math.cos(lat)
    N = WGS84_A / math.sqrt(1.0 - WGS84_E2 * s * s)
    X = (N + h_m) * c * math.cos(lon)
    Y = (N + h_m) * c * math.sin(lon)
    Z = ((1.0 - WGS84_E2) * N + h_m) * s
    return np.array([X, Y, Z], dtype=float)

def _ecef_to_enu(xyz, lat0_deg, lon0_deg, h0_m=0.0):
    x0, y0, z0 = _geodetic_to_ecef(lat0_deg, lon0_deg, h0_m)
    dx, dy, dz = xyz[0]-x0, xyz[1]-y0, xyz[2]-z0
    lat0 = _deg2rad(lat0_deg); lon0 = _deg2rad(lon0_deg)
    slat, clat = math.sin(lat0), math.cos(lat0)
    slon, clon = math.sin(lon0), math.cos(lon0)
    e = -slon * dx + clon * dy
    n = -slat * clon * dx - slat * slon * dy + clat * dz
    u =  clat * clon * dx + clat * slon * dy + slat * dz
    return np.array([e, n, u], dtype=float)

def _geodetic_to_enu(lat_deg, lon_deg, lat0_deg, lon0_deg, h_m=0.0, h0_m=0.0):
    return _ecef_to_enu(_geodetic_to_ecef(lat_deg, lon_deg, h_m), lat0_deg, lon0_deg, h0_m)

def _enu_to_ecef(e, n, u, lat0_deg, lon0_deg, h0_m=0.0):
    lat0 = _deg2rad(lat0_deg); lon0 = _deg2rad(lon0_deg)
    slat, clat = math.sin(lat0), math.cos(lat0)
    slon, clon = math.sin(lon0), math.cos(lon0)
    R = np.array([
        [-slon, -slat*clon,  clat*clon],
        [ clon, -slat*slon,  clat*slon],
        [  0.0,       clat,        slat]
    ], dtype=float)
    xyz0 = _geodetic_to_ecef(lat0_deg, lon0_deg, h0_m)
    return xyz0 + (R @ np.array([e, n, u], dtype=float))

def _ecef_to_geodetic(x, y, z):
    # Bowring/closed-form 근사
    a = WGS84_A; e2 = WGS84_E2
    b = a * (1.0 - WGS84_F)
    ep2 = (a*a - b*b) / (b*b)
    p = math.hypot(x, y)
    th = math.atan2(a * z, b * p)
    lon = math.atan2(y, x)
    lat = math.atan2(z + ep2 * b * math.sin(th)**3, p - e2 * a * math.cos(th)**3)
    sin_lat = math.sin(lat)
    N = a / math.sqrt(1.0 - e2 * sin_lat * sin_lat)
    h = p / math.cos(lat) - N
    return (math.degrees(lat), math.degrees(lon), h)

def _enu_to_geodetic(e, n, u, lat0_deg, lon0_deg, h0_m=0.0):
    xe, ye, ze = _enu_to_ecef(e, n, u, lat0_deg, lon0_deg, h0_m)
    return _ecef_to_geodetic(xe, ye, ze)

# ========================= GCP / 김포 레이아웃 =========================
GCP_LIST = [
    ("SEOUL_CITY_HALL", 37.566831, 126.978445, (  217063.379391, -1013419.868553, -220649.796452)),
    ("GIMPO_VERTIPORT", 37.563535, 126.791805, (-1431694.121809,  -979217.553043, -221157.611723)),
    ("YEOUIDO",         37.525491, 126.921490, ( -285967.804749,  -555936.730254, -221891.377864)),
    ("CHEONHO_4WAY",    37.538658, 127.123442, ( 1498921.351312,  -703218.244679, -223002.980913)),
    ("SUSEO_IC",        37.483396, 127.025980, (  638658.814081,   -89106.174219, -221891.377864)),
    ("IMUN_YARD",       37.603812, 127.068543, ( 1013805.274583, -1424832.145135, -223125.441155)),
    ("YEONSINNAE",      37.618936, 126.921349, ( -287516.699663, -1593208.781614, -222297.078953)),
]
CITY_HALL_NAME = "SEOUL_CITY_HALL"
def _get_cityhall_ue():
    for name, _, _, ue in GCP_LIST:
        if name == CITY_HALL_NAME:
            return ue
    raise RuntimeError("SEOUL_CITY_HALL 레퍼런스가 GCP_LIST에 없습니다.")
CITY_HALL_UE = _get_cityhall_ue()  # 절대 UE(cm)

# 김포 레이아웃(월드 좌표 → 로컬 오프셋)
GIMPO_ANCHOR_UE = (-1431694.121809, -979217.553043, -221157.611723)
GIMPO_POINTS_WORLD = {
    "GATE 8": (-1427843.924675, -981364.161650, -221057.611655),
    "GATE 7": (-1426086.696826, -981364.161659, -221057.611655),
    "GATE 6": (-1424398.202005, -981364.161686, -221057.611655),
    "GATE 5": (-1422658.476813, -981364.161737, -221057.611655),
    "GATE 4": (-1427798.845719, -977128.216053, -221057.611645),
    "GATE 3": (-1426087.595443, -977128.216058, -221057.611645),
    "GATE 2": (-1424398.577137, -977128.216057, -221057.611645),
    "GATE 1": (-1422701.715256, -977128.216051, -221057.611645),
    "FATO 2": (-1419811.599390, -981259.834764, -221057.611655),
    "FATO 1": (-1419838.218621, -977128.216053, -221057.611645),
    "TAXI 5": (-1427843.924645, -979191.878667, -221057.611655),
    "TAXI 4": (-1426103.859630, -979191.878789, -221057.611655),
    "TAXI 3": (-1424401.935901, -979191.878785, -221057.611655),
    "TAXI 2": (-1422698.043324, -979191.878799, -221057.611655),
    "TAXI 1": (-1419825.336754, -979191.878807, -221057.611655),
}
def _compute_gimpo_local_offsets():
    cx, cy, cz = GIMPO_ANCHOR_UE
    return {k: (x - cx, y - cy, z - cz) for k, (x, y, z) in GIMPO_POINTS_WORLD.items()}
GIMPO_OFFSETS = _compute_gimpo_local_offsets()

def rotate_yaw_ccw(offset_cm, yaw_deg):
    """Z축 기준 반시계(+) 회전 — 엑셀/GUI와 동일 정의."""
    r = math.radians(yaw_deg)
    c, s = math.cos(r), math.sin(r)
    x, y, z = offset_cm
    return (c*x - s*y, s*x + c*y, z)

# ========================= 아핀 매퍼(M) =========================
class AffineGeoToUEMapper:
    """
    - GCP(서울시청 주변 다점)를 ENU(m)로 변환 후 UE(cm)와 선형아핀(3x4)을 최소제곱 적합
    - geodetic_to_ue: WGS84 → UE(cm) 절대좌표
    - ue_ch_to_geodetic: CH-origin(cm)을 절대 UE로 되돌린 후 WGS84로
    """
    def __init__(self, gcps: Sequence[Tuple[str, float, float, Sequence[float]]]):
        city = [g for g in gcps if g[0] == CITY_HALL_NAME][0]
        self.lat0, self.lon0 = city[1], city[2]
        self._fit(gcps)

    def _fit(self, gcps):
        rows = []; Xs, Ys, Zs = [], [], []
        for _, lat, lon, ue in gcps:
            e, n, u = _geodetic_to_enu(lat, lon, self.lat0, self.lon0)
            rows.append([e, n, u, 1.0])
            Xs.append(ue[0]); Ys.append(ue[1]); Zs.append(ue[2])

        A = np.asarray(rows, dtype=float)
        X = np.asarray(Xs, dtype=float); Y = np.asarray(Ys, dtype=float); Z = np.asarray(Zs, dtype=float)
        Cx, *_ = np.linalg.lstsq(A, X, rcond=None)
        Cy, *_ = np.linalg.lstsq(A, Y, rcond=None)
        Cz, *_ = np.linalg.lstsq(A, Z, rcond=None)
        self.M = np.vstack([Cx, Cy, Cz])  # (3x4)

        # 역변환 준비
        self._M3 = self.M[:, :3].copy()
        self._b  = self.M[:, 3].copy()
        self._M3_inv = np.linalg.inv(self._M3)

        # 검증 잔차(절대 UE 기준, cm)
        self._residuals = []
        for name, lat, lon, ue in gcps:
            pred = self.geodetic_to_ue(lat, lon)
            err = pred - np.array(ue, dtype=float)
            self._residuals.append((name, float(err[0]), float(err[1]), float(err[2])))

    def geodetic_to_ue(self, lat_deg: float, lon_deg: float, h_m: float = 0.0) -> np.ndarray:
        e, n, u = _geodetic_to_enu(lat_deg, lon_deg, self.lat0, self.lon0, h_m, 0.0)
        v = np.array([e, n, u, 1.0], dtype=float)
        return self.M @ v  # 절대 UE(cm)

    def ue_abs_to_geodetic(self, x_cm: float, y_cm: float, z_cm: float):
        v = np.array([x_cm, y_cm, z_cm], dtype=float)
        enu = self._M3_inv @ (v - self._b)  # meters
        lat_deg, lon_deg, h_m = _enu_to_geodetic(enu[0], enu[1], enu[2], self.lat0, self.lon0, 0.0)
        return float(lat_deg), float(lon_deg), float(h_m)

    def ue_ch_to_geodetic(self, x_ch_cm: float, y_ch_cm: float, z_ch_cm: float):
        chx, chy, chz = CITY_HALL_UE
        return self.ue_abs_to_geodetic(chx + x_ch_cm, chy + y_ch_cm, chz + z_ch_cm)

    def residuals(self):
        return list(self._residuals)

# ----- 싱글톤 매퍼 -----
_MAPPER: Optional[AffineGeoToUEMapper] = None
def get_mapper() -> AffineGeoToUEMapper:
    global _MAPPER
    if _MAPPER is None:
        _MAPPER = AffineGeoToUEMapper(GCP_LIST)
    return _MAPPER

# ========================= 공개 API =========================
def wgs84_to_ue_ch_cm(lat: float, lon: float, h_m: float = 0.0) -> Tuple[float, float, float]:
    """WGS84 → UE(cm)[CH-origin]. (엑셀 경로와 동일)"""
    m = get_mapper()
    ax_abs, ay_abs, az_abs = m.geodetic_to_ue(lat, lon, h_m)
    chx, chy, chz = CITY_HALL_UE
    return float(ax_abs - chx), float(ay_abs - chy), float(az_abs - chz)

def wgs84_to_ue_ch_m(lat: float, lon: float, h_m: float = 0.0) -> Tuple[float, float, float]:
    """WGS84 → UE_CH(m) 스케일(엑셀/GUI와 동일; 부호 변경 없음)"""
    x_cm, y_cm, z_cm = wgs84_to_ue_ch_cm(lat, lon, h_m)
    return (x_cm/JSON_XY_DIV, y_cm/JSON_XY_DIV, z_cm/JSON_Z_DIV)

def wgs84_to_airsim_ned(lat: float, lon: float, alt_m: float = 0.0) -> Tuple[float, float, float]:
    """
    위·경도(+고도[m]) → AirSim NED [m]
      - N = X_ch_cm/100, E = Y_ch_cm/100, D = -Z_ch_cm/100
    """
    x_cm, y_cm, z_cm = wgs84_to_ue_ch_cm(lat, lon, alt_m)
    n_m = x_cm / JSON_XY_DIV
    e_m = y_cm / JSON_XY_DIV
    d_m = (-z_cm / JSON_Z_DIV) + 3
    return (n_m, e_m, d_m)

def ue_ch_to_geodetic(x_ch_cm: float, y_ch_cm: float, z_ch_cm: float):
    return get_mapper().ue_ch_to_geodetic(x_ch_cm, y_ch_cm, z_ch_cm)

def ue_abs_to_geodetic(x_cm: float, y_cm: float, z_cm: float):
    return get_mapper().ue_abs_to_geodetic(x_cm, y_cm, z_cm)

def gimpo_layout_from_anchor(lat: float, lon: float, yaw_deg: float) -> Dict[str, Tuple[float, float, float]]:
    """앵커(lat, lon) + yaw → 김포 레이아웃 포인트(CH-origin cm)."""
    m = get_mapper()
    ax_abs, ay_abs, az_abs = m.geodetic_to_ue(lat, lon, 0.0)
    pts_abs = {}
    for label, off in GIMPO_OFFSETS.items():
        ofr = rotate_yaw_ccw(off, yaw_deg)
        pts_abs[label] = (ax_abs + ofr[0], ay_abs + ofr[1], az_abs + ofr[2])
    chx, chy, chz = CITY_HALL_UE
    return {k: (vx - chx, vy - chy, vz - chz) for k, (vx, vy, vz) in pts_abs.items()}

# ========================= 스모크 테스트 =========================
if __name__ == "__main__":
    x, y, z = wgs84_to_ue_ch_cm(37.566831, 126.978445, 0.0)
    print("[CH-origin cm] CityHall:", x, y, z)
    n, e, d = wgs84_to_airsim_ned(37.566831, 126.978445, 0.0)
    print("[AirSim NED m] CityHall:", n, e, d)
    for name, dx, dy, dz in get_mapper().residuals():
        xy = (dx*dx + dy*dy) ** 0.5 / 100.0
        print(f"  {name:14s}: dX={dx:7.1f} dY={dy:7.1f} dZ={dz:7.1f}  (XY≈{xy:5.2f} m)")
