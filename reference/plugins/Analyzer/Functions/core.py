# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path
from typing import Optional, List, Tuple

import datetime as dt
import numpy as np
import pandas as pd
import branca.colormap as cm

# --- PyQt JumpSlider (NoiseTab 외부에 두되, 파일 수 줄이기 위해 core에 포함) ---
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QSlider, QStyle, QStyleOptionSlider

class JumpSlider(QSlider):
    """클릭 지점으로 점프 + 핸들 히트박스 확장 (UI 동일)"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._dragging = False
    def _handle_rect(self):
        opt = QStyleOptionSlider(); self.initStyleOption(opt)
        return self.style().subControlRect(QStyle.CC_Slider, opt, QStyle.SC_SliderHandle, self)
    def mousePressEvent(self, ev):
        r = self._handle_rect().adjusted(-6, -6, +6, +6)
        if r.contains(ev.pos()):
            self._dragging = True; return super().mousePressEvent(ev)
        if ev.button() == Qt.LeftButton:
            opt = QStyleOptionSlider(); self.initStyleOption(opt)
            groove = self.style().subControlRect(QStyle.CC_Slider, opt, QStyle.SC_SliderGroove, self)
            if groove.contains(ev.pos()):
                vmin, vmax = self.minimum(), self.maximum()
                x = ev.pos().x(); ratio = (x - groove.left()) / max(1, groove.width())
                self.setValue(int(vmin + (vmax - vmin) * min(max(ratio, 0), 1))); return
        return super().mousePressEvent(ev)
    def mouseMoveEvent(self, ev):
        if self._dragging or self.isSliderDown(): return super().mouseMoveEvent(ev)
        ev.ignore()
    def mouseReleaseEvent(self, ev):
        self._dragging = False; return super().mouseReleaseEvent(ev)

# --- 경로/상수 ---
FUNCTIONS_DIR = Path(__file__).resolve().parent
MOD3 = FUNCTIONS_DIR.parent
ROOT = MOD3.parent
RES  = MOD3 / "Sources"

# 기본 데이터 디렉터리(모든 파일 대화상자의 시작 폴더)
DB = MOD3 / "database"

def get_db_dir() -> Path:
    p = DB
    try:
        p.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    return p


SLD_MAX_SEC      = 24 * 3600
TIME_STEP_SEC    = 60
TIME_PAGE_SEC    = 600
FILTER_WINDOW_S  = 60

TABLE_MAX_ROWS   = 600

DEF_GRID_SHP = RES / "nlsp_020001001.shp"
DEF_VP_CSV   = RES / "vertiport.csv"
DEF_WP_CSV   = RES / "waypoint.csv"
DEF_NOISE    = None

PADDING_FACTOR     = 0.10
NOISE_COL_PRIORITY = ("Lmax_w", "Lmax_1s")
NOISE_VMIN         = 40.0
NOISE_VMAX         = 100.0
POP_ZOOM_BUMP      = 1.5

LDEN_DAY_RANGE   = (7, 19)
LDEN_EVE_RANGE   = (19, 23)
LDEN_NIGHT_RNGS  = ((23, 24), (0, 7))
LDEN_PEN_DB_DAY  = 0.0
LDEN_PEN_DB_EVE  = 5.0
LDEN_PEN_DB_NGT  = 10.0

def candidate_noise_paths():
    return [
        ROOT / "noise_log.csv",
        ROOT / "SITL" / "noise_log.csv",
        ROOT / "Analysis" / "output" / "noise_log.csv",
    ]

# --- 유틸 ---
def first_existing(paths: List[Path]) -> Optional[Path]:
    for p in paths:
        if p and p.exists(): return p
    return None

def lden_penalty_db_for_tsec(tsec: int) -> float:
    h = (int(tsec) // 3600) % 24
    if LDEN_DAY_RANGE[0] <= h < LDEN_DAY_RANGE[1]: return LDEN_PEN_DB_DAY
    if LDEN_EVE_RANGE[0] <= h < LDEN_EVE_RANGE[1]: return LDEN_PEN_DB_EVE
    for a, b in LDEN_NIGHT_RNGS:
        if a <= h < b: return LDEN_PEN_DB_NGT
    return LDEN_PEN_DB_NGT

def parse_tsec(x) -> Optional[int]:
    try:
        s = str(x).strip()
        for fmt in ("%H:%M:%S", "%H:%M"):
            try:
                t = dt.datetime.strptime(s, fmt).time()
                return t.hour*3600 + t.minute*60 + t.second
            except ValueError: pass
        e = float(s)
        t = dt.datetime.fromtimestamp(e).time()
        return t.hour*3600 + t.minute*60 + t.second
    except Exception: return None

def ensure_gid_int(gdf):
    if "gid" not in gdf.columns:
        for c in gdf.columns:
            if c.lower() == "gid": gdf = gdf.rename(columns={c: "gid"}); break
    gdf["gid"] = gdf["gid"].astype(str).str.extract(r"(\d+)", expand=False).astype("Int64")
    return gdf.dropna(subset=["gid"])

def linear_cmap(colors, vmin, vmax, caption):
    return cm.LinearColormap(colors=colors, vmin=vmin, vmax=vmax, caption=caption)

def pick_any(row, *names, default=None):
    for n in names:
        if n in row and pd.notna(row[n]): return row[n]
    return default

# --- 소음 지표 계산 ---
def compute_daily_level_from_noise(noise: pd.DataFrame, use_lden: bool=True) -> Optional[pd.DataFrame]:
    if noise is None or noise.empty: return None
    nd = noise.copy()
    col = next((c for c in NOISE_COL_PRIORITY if c in nd.columns), None)
    if col is None or "gid" not in nd.columns: return None
    if "tsec" not in nd.columns:
        key = "sim_time" if "sim_time" in nd.columns else ("time" if "time" in nd.columns else None)
        if key: nd["tsec"] = nd[key].apply(parse_tsec)
    base = nd.dropna(subset=[col, "gid"]).copy()
    base["gid"] = pd.to_numeric(base["gid"], errors="coerce").astype("Int64")
    base["L"]   = pd.to_numeric(base[col], errors="coerce")
    base = base.dropna(subset=["gid","L"])
    if use_lden:
        base["pen"] = base.get("tsec", 0).apply(lden_penalty_db_for_tsec)
        base["lin"] = 10.0 ** ((base["L"] + base["pen"]) / 10.0)
    else:
        base["lin"] = 10.0 ** (base["L"] / 10.0)
    wcol = next((c for c in ["dur","duration","dt","tau","w","weight"] if c in base.columns), None)
    if wcol:
        base["w"]   = pd.to_numeric(base[wcol], errors="coerce").fillna(0.0)
        num = (base["lin"] * base["w"]).groupby(base["gid"]).sum()
        den = base["w"].groupby(base["gid"]).sum().replace(0, np.nan)
        ratio = (num / den).replace([0, np.inf, -np.inf], np.nan)
        out = (10.0 * np.log10(ratio)).rename("L").reset_index()
        out["L"] = pd.to_numeric(out["L"], errors="coerce")
        return out
    inst = base.groupby(["gid","tsec"], as_index=False)["lin"].sum()
    def _per_gid(g):
        t = g["tsec"].to_numpy(dtype=np.int64)
        E = g["lin"].to_numpy(dtype=float)
        if t.size == 0:
            return pd.DataFrame({"gid":[g["gid"].iloc[0]], "L":[np.nan]})
        ord_idx = np.argsort(t); t = t[ord_idx]; E = E[ord_idx]
        if t.size == 1:
            dtv = np.array([86400.0], dtype=float)
        else:
            dtv = np.empty_like(t, dtype=float)
            dtv[:-1] = (t[1:] - t[:-1]).astype(float)
            dtv[-1]  = float((t[0] + 86400) - t[-1])
        W = float((E * dtv).sum())
        T = float(dtv.sum()) if dtv.sum() > 0 else 1.0
        L = 10.0 * np.log10(W / T) if W > 0 else np.nan
        return pd.DataFrame({"gid":[g["gid"].iloc[0]], "L":[L]})
    out = inst.groupby("gid", group_keys=False).apply(_per_gid).reset_index(drop=True)
    out["L"] = pd.to_numeric(out["L"], errors="coerce")
    return out

def pick_noise_df(noise: pd.DataFrame, tsec: int, hour: int, agg: str="auto") -> Tuple[Optional[pd.DataFrame], Optional[str], Optional[str]]:
    if noise is None or noise.empty: return None, None, None
    nd = noise
    if "tsec" in nd.columns:
        t0 = int(tsec)
        def circ_diff(x):
            d = abs(int(x) - t0); return min(d, SLD_MAX_SEC - d)
        sel = nd[nd["tsec"].apply(circ_diff) <= FILTER_WINDOW_S]
    else:
        hh = hour if "hour" in nd.columns else 0
        sel = nd[nd.get("hour", 0) == hh]
    if sel.empty: return None, None, None
    col = next((c for c in NOISE_COL_PRIORITY if c in sel.columns), None)
    if col is None: return None, None, None
    if agg == "leq":
        wcol = next((c for c in ["dur","duration","dt","tau","w","weight"] if c in sel.columns), None)
        tmp = sel.dropna(subset=[col]).copy()
        if tmp.empty: return None, col, None
        if wcol:
            tmp["w"]   = pd.to_numeric(tmp[wcol], errors="coerce").fillna(0.0)
            tmp["lin"] = 10.0 ** (tmp[col] / 10.0)
            num = (tmp["lin"] * tmp["w"]).groupby(tmp["gid"]).sum()
            den = tmp["w"].groupby(tmp["gid"]).sum().replace(0, np.nan)
            L = 10.0 * np.log10((num/den).replace([0,np.inf,-np.inf], np.nan))
            out = pd.DataFrame({"gid": num.index.astype("Int64"), "L": L})
            return out.dropna(subset=["L"]), col, wcol
        else:
            out = (tmp.groupby("gid")[col].apply(lambda s: 10.0*np.log10(np.mean(10.0**(s/10.0)))).rename("L").reset_index())
            out["gid"] = out["gid"].astype("Int64")
            return out, col, None
    if "Lmax" in col:
        out = sel.groupby("gid")[col].max().rename("L").reset_index()
    else:
        out = sel.groupby("gid")[col].mean().rename("L").reset_index()
    out["gid"] = out["gid"].astype("Int64")
    return out, col, None
