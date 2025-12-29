# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Optional
import json, glob, os
import pandas as pd
import geopandas as gpd
from PyQt5.QtWidgets import QInputDialog
from .core import (
    RES, ROOT, DEF_GRID_SHP, DEF_VP_CSV, DEF_WP_CSV, candidate_noise_paths,
    ensure_gid_int, parse_tsec, first_existing
)
# --- 레지스트리 CSV 읽기 ---
def list_noise_from_registry_csv(registry_csv_path: str) -> List[Tuple[str, str]]:
    """
    레지스트리 CSV에서 (표시명, 절대경로) 목록을 만든다.
    요구 최소 컬럼: 'path' (절대/상대 경로)
    선택 컬럼: 'name' 또는 'label' (없으면 파일명 사용)
    """
    if not registry_csv_path or not os.path.exists(registry_csv_path):
        raise FileNotFoundError("레지스트리 CSV 경로가 유효하지 않습니다.")

    df = pd.read_csv(registry_csv_path)
    cols = {str(c).strip() for c in df.columns}

    if "path" not in cols:
        # 딱 한 컬럼만 있으면 그걸 path로 간주 (명시적 동의 없으면 권장X)
        if len(df.columns) == 1:
            df = df.rename(columns={df.columns[0]: "path"})
        else:
            raise ValueError("레지스트리 CSV에는 최소 'path' 컬럼이 필요합니다.")

    name_col = "name" if "name" in cols else ("label" if "label" in cols else None)

    items: List[Tuple[str, str]] = []
    for _, row in df.iterrows():
        p = str(row["path"]).strip()
        if not p:
            continue
        ap = os.path.abspath(p)
        nm = str(row[name_col]).strip() if name_col and pd.notna(row.get(name_col)) else os.path.basename(ap)
        items.append((nm, ap))
    # 중복 제거
    seen = set(); uniq = []
    for nm, ap in items:
        if ap not in seen:
            uniq.append((nm, ap))
            seen.add(ap)
    return uniq

def pick_noise_from_registry_csv(parent, registry_csv_path: str) -> Optional[str]:
    """
    등록된 목록에서 한 개 선택. 취소하면 None
    """
    items = list_noise_from_registry_csv(registry_csv_path)
    if not items:
        return None
    labels = [f"{nm}   ({os.path.basename(p)})" for nm, p in items]
    label, ok = QInputDialog.getItem(parent, "소음 데이터 선택", "등록된 파일:", labels, 0, False)
    if not ok:
        return None
    idx = labels.index(label)
    return items[idx][1]

# --- 노이즈 CSV 스키마 검사 ---
def validate_noise_schema(csv_path: str) -> Tuple[bool, Optional[str]]:
    """
    최소 요건:
      - 'gid' 존재
      - 'Lmax_w' 또는 'Lmax_1s' 중 하나
      - 'tsec' 또는 'time' 또는 'sim_time' 중 하나
    """
    try:
        df = pd.read_csv(csv_path, nrows=2)
    except Exception as e:
        return False, f"CSV를 읽을 수 없습니다: {e}"

    cols = set(map(str, df.columns))
    missing = []
    if "gid" not in cols:
        missing.append("gid")
    if not (("Lmax_w" in cols) or ("Lmax_1s" in cols)):
        missing.append("Lmax_w 또는 Lmax_1s")
    if not (("tsec" in cols) or ("time" in cols) or ("sim_time" in cols)):
        missing.append("tsec 또는 time/sim_time")

    if missing:
        return False, "필수 컬럼 누락: " + ", ".join(missing)
    return True, None

def load_grid(path: Optional[Path] = None) -> Tuple[Optional[gpd.GeoDataFrame], Optional[tuple], Optional[list]]:
    shp = Path(path) if path else DEF_GRID_SHP
    if not shp.exists(): return None, None, None
    gdf = gpd.read_file(shp, encoding="cp949", errors="ignore").to_crs(epsg=4326)
    gdf = ensure_gid_int(gdf)
    bounds = tuple(gdf.total_bounds)
    c = gdf.unary_union.centroid
    center = [float(c.y), float(c.x)]
    return gdf, bounds, center

def load_overlays(vp_csv: Optional[Path] = None, wp_csv: Optional[Path] = None):
    vp_path = Path(vp_csv) if vp_csv else DEF_VP_CSV
    wp_path = Path(wp_csv) if wp_csv else DEF_WP_CSV
    vp_df = pd.read_csv(vp_path) if vp_path.exists() else None
    wp_df = pd.read_csv(wp_path) if wp_path.exists() else None
    return vp_df, wp_df

def find_default_noise() -> Optional[Path]:
    return first_existing(candidate_noise_paths())

def load_noise(path: Optional[Path] = None) -> Optional[pd.DataFrame]:
    csv = Path(path) if path else find_default_noise()
    if not (csv and csv.exists()): return None
    df = pd.read_csv(csv)
    if "gid" in df.columns:
        df["gid"] = pd.to_numeric(df["gid"], errors="coerce").astype("Int64")
    for c in ("Lmax_w", "Lmax_1s"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "tsec" not in df.columns:
        key = "sim_time" if "sim_time" in df.columns else ("time" if "time" in df.columns else None)
        if key:
            df["tsec"] = df[key].apply(parse_tsec)
    if "tsec" in df.columns:
        df["hour"] = (pd.to_numeric(df["tsec"], errors="coerce").fillna(0).astype(int) // 3600) % 24
    return df
