# LocationDistanceCalculator.py
# ===================================================================
# • vertiport.csv 로부터 좌표를 자동 생성한다.
# • CSV 헤더 탐색 로직
#       - 이름  컬럼 : ‘vertiport’ 포함  또는  name / vp / location
#       - 위도  컬럼 : ‘lat’ 포함
#       - 경도  컬럼 : ‘lon’ 포함
#   (대소문자 무시, 공백 strip)
# • update_from_csv() 를 호출하면 언제든 다른 CSV로 교체 가능
# ===================================================================

from __future__ import annotations
import csv
from pathlib import Path
from typing import Dict, Tuple

# ---------------------------------------------------------------
# 내부 상태 : {name: (lat, lon)}
_location_coords: Dict[str, Tuple[float, float]] = {}


# ---------------------------------------------------------------
def _load_coords(csv_path: Path) -> Dict[str, Tuple[float, float]]:
    """vertiport.csv → {이름: (lat, lon)} 딕셔너리 반환."""
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)

    with csv_path.open(encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        hdr = [h.strip().lower() for h in reader.fieldnames]

        # --- 컬럼 인덱스 탐색 ----------------------------------------
        name_key = next(
            (h for h in reader.fieldnames
             if "vertiport" in h.lower() or h.strip().lower() in
             ("name", "vp", "location", "vertiport_nm")), None)
        lat_key = next((h for h in reader.fieldnames
                        if any(k in h.lower() for k in ("lat", "위도"))), None)
        lon_key = next((h for h in reader.fieldnames
                        if any(k in h.lower() for k in ("lon", "경도"))), None)

        if not (name_key and lat_key and lon_key):
            raise ValueError("vertiport.csv 헤더에 name/lat/lon 컬럼을 찾지 못했습니다.")

        coords = {}
        for row in reader:
            try:
                name = str(row[name_key]).strip()
                lat  = float(row[lat_key])
                lon  = float(row[lon_key])
                coords[name] = (lat, lon)
            except (ValueError, TypeError):
                # 숫자 변환 실패 → 해당 행 skip
                continue
        return coords


# ---------------------------------------------------------------
def update_from_csv(csv_path: str | Path) -> None:
    """다른 vertiport.csv 로 좌표 dict 를 갱신한다 (in-place)."""
    csv_path = Path(csv_path).expanduser().resolve()
    _location_coords.clear()
    _location_coords.update(_load_coords(csv_path))


# ---------------------------------------------------------------
# 공개 인터페이스 -------------------------------------------------
location_coords = _location_coords            # 기존 코드 호환용 별칭


def get(name: str) -> Tuple[float, float] | None:
    """이름으로 (lat, lon) 튜플 반환. 없으면 None."""
    return _location_coords.get(name)


# ------------------------------------------------------------------
#  Haversine distance  (lat, lon in deg  →  km 반환)
# ------------------------------------------------------------------
import math
def haversine(lon1: float, lat1: float,
              lon2: float, lat2: float) -> float:
    R = 6_371.0  # Earth radius [km]
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    dφ     = math.radians(lat2 - lat1)
    dλ     = math.radians(lon2 - lon1)
    a = (math.sin(dφ / 2) ** 2 +
         math.cos(φ1) * math.cos(φ2) * math.sin(dλ / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(a))

__all__ = ["update_from_csv", "location_coords", "get", "haversine"]