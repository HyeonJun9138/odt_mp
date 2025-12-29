# geo_utils.py  ── 모든 모듈이 공유하는 좌표·bearing 함수
from __future__ import annotations   # ← 이 줄만 추가하면 끝!
import math
LON0, LAT0 = 126.978291, 37.566669          # 서울 시청
KM_PER_DEG = 111.32

def _km_per_lon(lat_deg: float) -> float:
    return KM_PER_DEG * math.cos(math.radians(lat_deg))

def lonlat_to_xy(lon: float, lat: float) -> tuple[float, float]:
    """Plate‑Carrée [m]  ( +x=E, +y=N )"""
    dx = (lon - LON0) * _km_per_lon(LAT0) * 1000.0
    dy = (lat - LAT0) * KM_PER_DEG       * 1000.0
    return dx, dy

def xy_to_lonlat(x_m: float, y_m: float) -> tuple[float, float]:
    d_lat = (y_m/1000.0) / KM_PER_DEG
    d_lon = (x_m/1000.0) / _km_per_lon(LAT0)
    return LON0 + d_lon, LAT0 + d_lat     # (lon, lat)

def bearing_deg(lon1: float, lat1: float,
                lon2: float, lat2: float) -> float:
    """진북 0°, 시계방향+  [0‥360)"""
    φ1, φ2 = map(math.radians, (lat1, lat2))
    Δλ      = math.radians(lon2 - lon1)
    x = math.sin(Δλ) * math.cos(φ2)
    y = math.cos(φ1)*math.sin(φ2) - math.sin(φ1)*math.cos(φ2)*math.cos(Δλ)
    return (math.degrees(math.atan2(x, y)) + 360.0) % 360.0
