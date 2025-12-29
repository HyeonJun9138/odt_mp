#!/usr/bin/env python
"""
terrain_noise.py  ―  DEM·소음 계산 공용 유틸
───────────────────────────────────────────────────────────────────────────────
• 서울·경기 DEM 타일(n37_e126, n37_e127) 모자이크 → numpy 배열
• 단일 항공기 기준:
      slant_distance()           : 공중→지면 기울기 거리
      CruiseNoiseCalc            : 간단한 감쇠 모델 (L0‧d0‧α)
      contours_for_vehicle()     : DEM 높이를 고려한 dB 등고선(Leaflet용)
      compute_noise_for_vehicle(): ThreadPool 병렬용; 컨투어/원형·최대 dB 반환
• GridNoiseLogger                : 1 s 창(fixed window)별 Leq, 일일 Lden 로그

※ 모든 좌표는 WGS84(lat, lon) 기반, 거리계산시 단순 평면 근사 사용
"""

# ── 라이브러리 ────────────────────────────────────────────────────────────────
import math
import csv
import threading
import queue
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
from skimage import measure
import rasterio
from rasterio.merge import merge
from rasterio.windows import from_bounds
import geopandas as gpd
import shapely.geometry as sgeom
import branca.colormap as cm

# 내부 공용 함수 (ENU <-> LLH 변환) ───────────────────────────────────────────
from Monitoring.Functions.geo_utils import lonlat_to_xy, xy_to_lonlat

# ────────────────────────────────────────────────────────────────────────────
# 0) 경로·DEM 초기화
# ────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESOURCE_DIR = PROJECT_ROOT / "Analysis" / "Sources"

DEM_TILES: List[Path] = [
    RESOURCE_DIR / "n37_e126_1arc_v3.tif",
    RESOURCE_DIR / "n37_e127_1arc_v3.tif",
]

# 계산용 파라미터
DS_MIN      = 5.0
RADIUS_M    = 150.0
ANGLE_STEP  = 36
R_SAMPLE_STEP = 25.0                 # 25 m 간격 예시
R_SAMPLES = [DS_MIN] + [              # 5, 30, 55, …, 155 → 7 개
    r for r in np.arange(DS_MIN + R_SAMPLE_STEP,
                         RADIUS_M + R_SAMPLE_STEP, R_SAMPLE_STEP)
]
# DEM 모자이크 (메모리 캐시)
try:
    srcs = [rasterio.open(f) for f in DEM_TILES]
    mosaic, dem_transform = merge(srcs)        # 1개 배열로 합치기
    full_dem = mosaic[0]                       # 2D numpy array
    DEM = True
except Exception as e:
    print("[terrain_noise] DEM load failed → flat terrain:", e)
    full_dem = dem_transform = None
    DEM = False

# ────────────────────────────────────────────────────────────────────────────
# 1) DEM 헬퍼
# ────────────────────────────────────────────────────────────────────────────
def get_dem_window(lat_min: float, lon_min: float,
                   lat_max: float, lon_max: float) -> np.ndarray:
    """위·경도 bbox(좌하↔우상)에 해당하는 DEM 슬라이스 반환."""
    win = from_bounds(lon_min, lat_min, lon_max, lat_max, transform=dem_transform)
    r0, c0 = int(win.row_off), int(win.col_off)
    h, w   = int(win.height),   int(win.width)
    r1, c1 = r0 + h, c0 + w

    # 배열 경계 체크
    r0, c0 = max(0, r0), max(0, c0)
    r1, c1 = min(full_dem.shape[0], r1), min(full_dem.shape[1], c1)
    return full_dem[r0:r1, c0:c1]

def ground_alt(lat: float, lon: float) -> float:
    """단일 지점 지면 고도 read (m)."""
    row, col = rasterio.transform.rowcol(dem_transform, lon, lat)
    return float(full_dem[row, col])

# ────────────────────────────────────────────────────────────────────────────
# 2) 위치·거리 계산
# ────────────────────────────────────────────────────────────────────────────
def project(lat: float, lon: float,
            bearing_deg: float, distance_m: float) -> Tuple[float, float]:
    """(lat, lon)에서 bearing_deg 방위로 distance_m 만큼 이동한 LL 좌표."""
    x0, y0 = lonlat_to_xy(lon, lat)
    θ = math.radians(bearing_deg)
    dx, dy = distance_m * math.sin(θ), distance_m * math.cos(θ)
    lon1, lat1 = xy_to_lonlat(x0 + dx, y0 + dy)
    return lat1, lon1

def slant_distance(lat1: float, lon1: float, alt1: float,
                   lat2: float, lon2: float, alt2: float = 0.0) -> float:
    """두 점(위경도+고도) 간 3-D 경사거리(m) 단순 근사."""
    dx = (lon2 - lon1) * 111_320 * math.cos(
        math.radians((lat1 + lat2) / 2)
    )
    dy = (lat2 - lat1) * 110_540
    return math.hypot(math.hypot(dx, dy), alt1 - alt2)

# 평면 반경 계산용
def hor_radius_for_dB(L: float, alt: float,
                      L0: float = 60.0, d0: float = 300.0) -> float:
    """주어진 레벨 L[dB]에서 평면상 도달 가능 반경(m)."""
    ds = d0 * 10 ** ((L0 - L) / 20)
    return math.sqrt(max(0.0, ds * ds - alt * alt))

# ────────────────────────────────────────────────────────────────────────────
# 3) 소음 모델
# ────────────────────────────────────────────────────────────────────────────
class CruiseNoiseCalc:
    """(거리) ↦ (dB) 함수  –  수학식은 CruiseNoiseCalculator와 동일."""
    def __init__(self, L0: float = 60.0, d0: float = 300.0,
                 alpha: float = 0.0001) -> None:
        self.L0, self.d0, self.alpha = L0, d0, alpha

    def level(self, ds):
        """numpy-vectorized 레벨 계산."""
        ds = np.asarray(ds)
        ds = np.where(ds <= 0, 1e-6, ds)   # 0 div 보호
        return (
            self.L0
            - 20 * np.log10(ds / self.d0)
            - self.alpha * (ds - self.d0)
        )

# Leaflet 색상 매핑 (50→green … 70→red)
_cmap = cm.LinearColormap(['green', 'yellow', 'red'],
                          vmin=50, vmax=70).to_step(20)

# ────────────────────────────────────────────────────────────────────────────
# 4) DEM 기반 등고선
# ────────────────────────────────────────────────────────────────────────────
def contours_for_vehicle(lat0: float, lon0: float, alt_agl: float,
                          thresholds=range(50, 71), reach_m: int = 500) -> list:
    """단일 항공기에 대해 skimage 컨투어 → Leaflet polygon 좌표 반환.

    Returns
    -------
    list[dict] : {poly:[(lat,lon)…], color:'#RRGGBB'}
    """
    if not DEM:
        return []

    # ① 관심 영역 DEM 슬라이스
    dlat = reach_m / 110_540
    dlon = reach_m / (111_320 * math.cos(math.radians(lat0)))
    lat_min, lat_max = lat0 - dlat, lat0 + dlat
    lon_min, lon_max = lon0 - dlon, lon0 + dlon

    arr = get_dem_window(lat_min, lon_min, lat_max, lon_max)
    nrows, ncols = arr.shape
    if nrows < 2 or ncols < 2:
        return []

    # ② 좌표 그리드
    ys = np.linspace(lat_max, lat_min, nrows)
    xs = np.linspace(lon_min, lon_max, ncols)
    lats, lons = np.meshgrid(ys, xs, indexing='ij')
    ground = arr.astype(float)
    alt_diff = alt_agl - ground

    # ③ 거리 맵 & 레벨 맵
    Rx = (lons - lon0) * 111_320 * np.cos(np.radians((lats + lat0) / 2))
    Ry = (lats - lat0) * 110_540
    ds = np.hypot(np.hypot(Rx, Ry), alt_diff)
    level_map = CruiseNoiseCalc().level(ds)

    # ④ 임곗값별 컨투어
    out = []
    for L in thresholds:
        for c in measure.find_contours(level_map, L):
            # pixel → lat/lon 변환
            lat_poly = lat_max - (c[:, 0] / (nrows - 1)) * (lat_max - lat_min)
            lon_poly = lon_min + (c[:, 1] / (ncols - 1)) * (lon_max - lon_min)
            poly = list(zip(lat_poly, lon_poly))
            if len(poly) >= 3:
                out.append({"poly": poly, "color": _cmap(L)})
    return out

# ────────────────────────────────────────────────────────────────────────────
# 5) ThreadPool용 단일 비행체 계산 함수
# ────────────────────────────────────────────────────────────────────────────
def compute_noise_for_vehicle(args):
    """
    Parameters
    ----------
    args : tuple
        (vid, lat, lon, alt_agl, thresholds)

    Returns
    -------
    vid              : str
    poly_list        : list[(coords, color)]  # DEM 등고선
    raw_list         : list[(lat, lon, r, color)]  # fallback 원(circle)
    max_db           : float  # 최대 레벨
    """
    vid, lat, lon, alt_agl, thresholds = args
    cnc = CruiseNoiseCalc()
    max_db = -float("inf")

    # ① 10° 간격으로 DS_MIN·RADIUS_M 두 점 샘플 → 최대 dB 추출
    for ang in range(0, 360, ANGLE_STEP):
        for r in R_SAMPLES:              # (DS_MIN, 30, 55, …)
            plat, plon = project(lat, lon, ang, r)
            grd  = ground_alt(plat, plon)
            ds   = max(slant_distance(lat, lon, alt_agl, plat, plon, grd), DS_MIN)
            lvl  = cnc.level(ds)
            max_db = max(max_db, lvl)

    poly_list, raw_list = [], []

    # ② DEM 유무에 따라 결과 형식 결정
    if DEM:
        for pd in contours_for_vehicle(lat, lon, alt_agl, thresholds):
            poly_list.append([pd["poly"], pd["color"]])
    else:  # DEM 없음 → 평면 반경 원
        for L in thresholds:
            r = hor_radius_for_dB(L, alt_agl)
            col = "#%02x%02x%02x" % (0, 255 - (L - 55) * 10, (L - 55) * 10)
            raw_list.append([lat, lon, r, col])

    # numpy → python list (json 직렬화 대비)
    poly_list = [(np.asarray(c, dtype=np.float32).tolist(), col)
                 for c, col in poly_list]
    return vid, poly_list, raw_list, float(max_db)

# ────────────────────────────────────────────────────────────────────────────
# 6) 인구 격자(GID) 셋업
# ────────────────────────────────────────────────────────────────────────────
_SHAPE = RESOURCE_DIR / "nlsp_020001001.shp"
_GDF = (
    gpd.read_file(_SHAPE, encoding="cp949", errors="ignore")
       .to_crs(epsg=4326)
)
_GDF["gid_num"] = (
    _GDF["gid"].astype(str).str.extract(r"(\d+)", expand=False).astype("Int64")
)
_GDF = _GDF.dropna(subset=["gid_num"])
_ALL_GIDS = _GDF["gid_num"].astype(int).unique().tolist()
_SIDX = _GDF.sindex  # r-tree spatial index

def _gid_for(lat: float, lon: float) -> int | None:
    """Point가 포함된 gid_num 반환 (없으면 None)."""
    pt = sgeom.Point(lon, lat)
    for idx in _SIDX.intersection(pt.bounds):
        if _GDF.geometry.iloc[idx].contains(pt):
            return int(_GDF.gid_num.iloc[idx])
    return None

# ────────────────────────────────────────────────────────────────────────────
# 7) GridNoiseLogger – L_eq(1 s) & 일일 L_den
# ────────────────────────────────────────────────────────────────────────────
class GridNoiseLogger(threading.Thread):
    """
    UDP → NoiseTab에서 호출하여 격자별 L_eq(1 s), L_den를 계산·CSV 기록.

    • push(hms, lat, lon, max_db) : 비동기 큐에 이벤트 누적
    • 내부 스레드:
        – 매 1 s 윈도우 단위로 noise_log.csv append
        – 종료(21:30) 시 den_log.csv 1회 기록
    """
    def __init__(self, path: str, flush_sec: int = 1):
        super().__init__(daemon=True)
        self._q = queue.Queue()
        self._flush = flush_sec        # 윈도우 폭(초)
        self._path = path

        # 파일 헤더 초기화
        with open(self._path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["sim_time", "gid", "Lmax_1s", "Lmax_w"])
        with open("den_log.csv", "w", newline="", encoding="utf-8") as d:
            csv.writer(d).writerow(["time", "gid", "Lden", "count"])

        # 누적 상태
        self._dayE = {gid: 0.0 for gid in _ALL_GIDS}  # Σ10^(Leq_w/10)
        self._total_seconds = 0
        self._last_idx = None          # 직전 윈도우 idx
        self._bucket: Dict[int, float] = {}

        self.start()

    # ---------- 내부 헬퍼 ----------
    def _fmt_time(self, win_idx: int) -> str:
        """윈도우 idx → HH:MM:SS 문자열."""
        total = win_idx * self._flush
        hh = total // 3600
        mm = (total % 3600) // 60
        ss = total % 60
        return f"{hh:02d}:{mm:02d}:{ss:02d}"

    # ---------- API ----------
    def push(self, hms: str, lat: float, lon: float, max_db: float) -> None:
        """NoiseTab에서 호출 – 1개 이벤트 누적."""
        hh, mm, ss = map(int, hms.split(":"))

        # 21:30 이후 데이터는 기록 종료
        if hh > 21 or (hh == 21 and mm >= 30):
            if self._last_idx is not None:
                # 마지막 윈도우 강제 flush
                self._q.put((self._fmt_time(self._last_idx),
                             self._bucket.copy()))
            self._q.put(None)          # 종료 신호
            return

        sec_total = hh * 3600 + mm * 60 + ss
        win_idx = sec_total // self._flush

        # 윈도우 전환 → 큐에 enqueue
        if self._last_idx is not None and win_idx != self._last_idx:
            # 중간에 비어있는 윈도우 채우기
            for missing in range(self._last_idx + 1, win_idx):
                self._q.put((self._fmt_time(missing), {}))
            # 직전 윈도우 기록
            self._q.put((self._fmt_time(self._last_idx), self._bucket.copy()))
            self._bucket.clear()
        self._last_idx = win_idx

        # 버킷 누적
        if max_db > 0:
            gid = _gid_for(lat, lon)
            if gid is not None:
                # 창(1 s) 동안 등장한 값 중 최대치만 유지
                self._bucket[gid] = max(max_db, self._bucket.get(gid, -math.inf))

    # ---------- 스레드 루프 ----------
    def run(self):
        """큐를 소비하여 noise_log.csv / den_log.csv 작성."""
        while True:
            item = self._q.get()
            if item is None:
                break  # 종료

            sim_time, bucket = item

            # ① noise_log.csv: 현재 윈도우 Leq, Leq_w
            with open(self._path, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                for gid in _ALL_GIDS:
                    if gid in bucket:
                        Lmax   = bucket[gid]
                        Lmax_w = Lmax + self._offset(sim_time)
                    else:
                        Lmax = Lmax_w = 0.0
                    w.writerow([sim_time, gid, f"{Lmax:.1f}", f"{Lmax_w:.1f}"])
                    self._dayE[gid] += 10 ** (Lmax_w / 10)

            self._total_seconds += 1
            self._q.task_done()

        # # ② 종료 시 den_log.csv: 하루 L_den
        # with open("den_log.csv", "a", newline="", encoding="utf-8") as d:
        #     w2 = csv.writer(d)
        #     for gid, E in self._dayE.items():
        #         Lden = 10 * math.log10(E / self._total_seconds)
        #         w2.writerow([self._fmt_time(self._last_idx), gid,
        #                      f"{Lden:.1f}", self._total_seconds])

    # ---------- 시간대별 보정계수 ----------
    @staticmethod
    def _offset(hms: str) -> int:
        """시각별 가중 보정(dB) – 야간·저녁 가중치."""
        hh = int(hms[:2])
        if 19 <= hh < 22:
            return 5
        if hh >= 22 or hh < 7:
            return 10
        return 0

# ────────────────────────────────────────────────────────────────────────────
# 8) 싱글턴 인스턴스 + 래퍼
# ────────────────────────────────────────────────────────────────────────────
NOISE_LOGGER = GridNoiseLogger("noise_log.csv", flush_sec=1)

def log_grid_noise(hms: str, lat: float, lon: float, max_db: float) -> None:
    """외부 모듈용 래퍼(쓰레드 세이프)."""
    NOISE_LOGGER.push(hms, lat, lon, max_db)
