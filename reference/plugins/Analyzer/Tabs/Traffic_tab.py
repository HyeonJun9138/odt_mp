# -*- coding: utf-8 -*-
from __future__ import annotations
import sys, os, json, math, tempfile
from pathlib import Path
from typing import Optional, Dict, Tuple
from Analyzer.Functions.core import get_db_dir


# 패키지 경로 보정 (Noise_tab와 동일 컨벤션)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import pandas as pd
import geopandas as gpd
import folium
import branca.colormap as cm
from shapely.geometry import LineString
from collections import defaultdict
from PyQt5.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QFrame, QLabel, QPushButton, QTableWidget,
    QTableWidgetItem, QAbstractItemView, QHeaderView, QFileDialog,
    QSpinBox, QAbstractSpinBox, QSlider, QTextBrowser
)
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtCore import Qt, QUrl, QTimer, QThread, pyqtSignal

from Tabs.base_tab import BaseTab
from Functions.core import (
    JumpSlider, RES, ROOT,
    SLD_MAX_SEC, TIME_STEP_SEC, TIME_PAGE_SEC,FILTER_WINDOW_S   
)
from Functions.io import load_overlays
from Functions.mapkit import run_js, build_js_api_traffic, update_traffic

LANES = ("L-1000","L-2000","R-1000","R-2000")

def _offset_latlon(lat: float, lon: float, east_km: float, north_km: float) -> Tuple[float,float]:
    dlat = north_km / 110.574
    dlon = east_km  / (111.320 * math.cos(math.radians(lat)) + 1e-9)
    return lat + dlat, lon + dlon

class _ExportCongestionThread(QThread):
    progressed = pyqtSignal(int, int)   # (현재, 전체)
    done = pyqtSignal(str)              # out_path
    error = pyqtSignal(str)

    def __init__(self, assign_func, edges_df, track_df, out_path: Path, sld_max: int, win_sec: int = 0, parent=None):
        super().__init__(parent)
        self.assign_func = assign_func      # TrafficTab._assign_points_to_edges 바운드 함수
        self.edges = edges_df               # a_x,a_y,b_x,b_y 만 있어도 됨
        self.track = track_df               # tsec,x_km,y_km,lane
        self.out_path = out_path
        self.sld_max = int(sld_max)
        self.win_sec = int(win_sec)

    def run(self):
        try:
            df = self.track.copy()
            lanes_ok = {"L-1000","R-1000","L-2000","R-2000"}
            df["lane"] = df.get("lane","").astype(str).str.strip().str.upper()
            df = df[df["lane"].isin(lanes_ok)].copy()

            t = pd.to_numeric(df.get("tsec"), errors="coerce")
            df = df[t.notna()].copy()
            df["t"] = (t.loc[df.index].astype(int) % self.sld_max).astype(int)

            # 최근접 에지는 한 번만 계산
            pts = df[["x_km","y_km"]].to_numpy()
            idx = self.assign_func(pts, self.edges).astype(int)
            df["e"] = idx
            df["is1"] = df["lane"].isin(["L-1000","R-1000"]).astype(np.int32)
            df["is2"] = df["lane"].isin(["L-2000","R-2000"]).astype(np.int32)

            grp = df.groupby(["t","e"], sort=True).agg(is1=("is1","sum"), is2=("is2","sum"))

            import csv, datetime as dt
            with self.out_path.open("w", newline="", encoding="utf-8-sig") as f:
                w = csv.writer(f)
                w.writerow(["time","tsec","edge_id","lane","density"])

                times = sorted(grp.index.get_level_values(0).unique().tolist())
                total = len(times)
                for i, ts in enumerate(times, 1):
                    sub = grp.loc[ts]              # index: edge_id
                    ref = np.concatenate([sub["is1"].to_numpy(), sub["is2"].to_numpy()])
                    ref = ref[ref > 0]
                    p95 = float(np.quantile(ref, 0.95)) if ref.size else 1.0
                    if p95 <= 0: p95 = 1.0

                    d1 = (10.0 * sub["is1"].to_numpy() / p95).clip(0, 10)
                    d2 = (10.0 * sub["is2"].to_numpy() / p95).clip(0, 10)
                    eids = sub.index.to_numpy(dtype=int)

                    h = (ts // 3600) % 24; m = (ts % 3600) // 60; s = ts % 60
                    tstr = f"{h:02d}:{m:02d}:{s:02d}"

                    for eid, v1, v2 in zip(eids, d1, d2):
                        w.writerow([tstr, ts, eid, "L-1000", float(v1)])
                        w.writerow([tstr, ts, eid, "R-1000", float(v1)])
                        w.writerow([tstr, ts, eid, "L-2000", float(v2)])
                        w.writerow([tstr, ts, eid, "R-2000", float(v2)])

                    if (i % 20) == 0:
                        self.progressed.emit(i, total)

            self.done.emit(str(self.out_path))
        except Exception as e:
            self.error.emit(str(e))


class TrafficTab(BaseTab):
    """
    시간 슬라이더로 선택한 창(±분) 또는 전체 평균 모드로 링크×레인 혼잡(0–10)을 시각화.
    웨이포인트×레인 기준 로그와 지역별 TO/LD 집계 표를 함께 제공.
    """
    def __init__(self, parent=None):
        super().__init__(parent, logo_path=RES / "kada_logo.png",
                         version_text="v0.0.1", show_back=True)
        self.setWindowTitle("Analyzer – Traffic Viewer")

        # 데이터 경로(기본: Sources)
        self.vp_csv = (RES / "vertiport.csv" if (RES / "vertiport.csv").exists() else RES / "vertiport.csv")
        self.wp_csv = (RES / "waypoint_vipp.csv"  if (RES / "waypoint_vipp.csv").exists()  else RES / "waypoint.csv")
        self.track_csv = (RES / "track_log.csv") if (RES / "track_log.csv").exists() else None

        # 데이터 캐시
        self._vp: Optional[pd.DataFrame] = None
        self._wp: Optional[pd.DataFrame] = None
        self._edges: Optional[pd.DataFrame] = None
        self._lanes_gdf: Optional[gpd.GeoDataFrame] = None
        self._bounds: Optional[tuple] = None
        self._track: Optional[pd.DataFrame] = None

        # 시간/모드 상태
        self._tsec: Optional[int] = None
        self._win_sec: int = int(FILTER_WINDOW_S)                # 기본 20분 창 (다이얼은 UI에서 제거)
        self._mode: str = "window"              # 'window' | 'avg'
        self._throttle = QTimer(self); self._throttle.setSingleShot(True); self._throttle.setInterval(200)
        self._throttle.timeout.connect(self._recompute_and_repaint)

        # 지도/JS 상태
        self._map_js_name = None
        self._lanes_js_name = None
        self._html = None
        self._web_ready = False
        self._base_ready = False
        self._last_vals: Dict[str,float] = {}
        self._last_cols: Dict[str,str]   = {}

        self._export_done = False  # 혼잡 CSV 1회 자동 저장 플래그
        

        self._build_ui()
        self._load_all()
        self._build_map_base()

    # ───────────────────────────── UI ─────────────────────────────
    def _build_ui(self):
        root = self.body

        # ── 상단: 슬라이더 + 시/분/초 + 모드 버튼
        top = QHBoxLayout(); top.setSpacing(12)

        # 시간 슬라이더
        slider_col = QVBoxLayout()
        self.slider = JumpSlider(Qt.Horizontal); self.slider.setObjectName("TimeSlider")
        self.slider.setRange(0, SLD_MAX_SEC); 
        self.slider.setSingleStep(1)     # ← 1초 단위
        self.slider.setPageStep(60)      # ← 1분 점프(페이지업/다운)
        self.slider.setTickInterval(3600)
        self.slider.setTickPosition(QSlider.TicksBelow); self.slider.setTracking(False)
        self.slider.valueChanged.connect(self._on_time_changed)
        slider_col.addWidget(self.slider)
        marks = QHBoxLayout()
        lbl00 = QLabel("00"); lbl00.setFixedWidth(24)
        lbl24 = QLabel("24."); lbl24.setFixedWidth(24); lbl24.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        marks.addWidget(lbl00); marks.addStretch(1); marks.addWidget(lbl24)
        slider_col.addLayout(marks)
        top.addLayout(slider_col, 3)

        # 시/분/초 스핀 (Noise와 동일 UX)
        self.spin_h = QSpinBox(objectName="HSpin"); self.spin_h.setRange(0, 23); self.spin_h.setSuffix(" h")
        self.spin_m = QSpinBox(objectName="MSpin"); self.spin_m.setRange(0, 59); self.spin_m.setSuffix(" m")
        self.spin_s = QSpinBox(objectName="SSpin"); self.spin_s.setRange(0, 59); self.spin_s.setSuffix(" s")

        step_s = max(1, TIME_STEP_SEC if TIME_STEP_SEC < 60 else 1)
        step_m = max(1, TIME_STEP_SEC // 60 if TIME_STEP_SEC >= 60 else 1)
        self.spin_h.setSingleStep(1); self.spin_m.setSingleStep(step_m); self.spin_s.setSingleStep(step_s)
        for sp in (self.spin_h, self.spin_m, self.spin_s):
            sp.setButtonSymbols(QAbstractSpinBox.UpDownArrows); sp.setFixedWidth(90)
        self.spin_h.valueChanged.connect(self._on_spins_changed)
        self.spin_m.valueChanged.connect(self._on_spins_changed)
        self.spin_s.valueChanged.connect(self._on_spins_changed)

        spbox = QHBoxLayout(); spbox.addWidget(self.spin_h); spbox.addWidget(self.spin_m); spbox.addWidget(self.spin_s)
        top.addLayout(spbox, 2)


        root.addLayout(top)

        # 지도
        mapPanel = QFrame(objectName="Panel"); mapL = QVBoxLayout(mapPanel); mapL.setContentsMargins(6,6,6,6)
        self.web = QWebEngineView(); self.web.setMinimumHeight(520); mapL.addWidget(self.web)
        self.web.loadFinished.connect(self._on_web_loaded)

        # ── 우측: 로그(스크롤) + 표
        right = QFrame(objectName="Panel")
        rL = QVBoxLayout(right); rL.setContentsMargins(6,6,6,6); rL.setSpacing(8)

        self.log_title = QLabel("Log")
        self.log_title.setStyleSheet("color:#9aa7b1; font-size:12px;")
        rL.addWidget(self.log_title, 0)

        self.log_view = QTextBrowser()
        self.log_view.setReadOnly(True)
        self.log_view.setOpenExternalLinks(False)
        self.log_view.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.log_view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.log_view.setStyleSheet("QTextBrowser{background:transparent; color:#d9dce4; border:none;}")
        self.log_view.setMinimumHeight(150)
        self.log_view.setMaximumHeight(240)
        rL.addWidget(self.log_view, 0)

        sep = QFrame(); sep.setFrameShape(QFrame.HLine); sep.setStyleSheet("background:#3a4050;"); sep.setFixedHeight(1)
        rL.addWidget(sep, 0)

        self.tbl_title = QLabel("표: 지역별 TO/LD")
        rL.addWidget(self.tbl_title, 0)

        self.tbl = QTableWidget(objectName="DataTable")
        self.tbl.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tbl.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tbl.setAlternatingRowColors(True); self.tbl.setShowGrid(False)
        self.tbl.verticalHeader().setVisible(False)
        self.tbl.horizontalHeader().setFixedHeight(32)
        self.tbl.horizontalHeader().setDefaultAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.tbl.horizontalHeader().setStretchLastSection(True)
        self.tbl.setSortingEnabled(True)
        rL.addWidget(self.tbl, 1)

        # 중앙 배치
        center = QHBoxLayout(); center.setContentsMargins(0,0,0,0); center.setSpacing(8)
        center.addWidget(mapPanel, 3)
        center.addWidget(right,   2)
        root.addLayout(center, 1)


    def _start_export_thread(self):
        """혼잡 CSV 저장을 백그라운드 스레드로 시작."""
        try:
            out_dir = get_db_dir()
            import datetime as dt
            stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = out_dir / f"traffic_congestion_{stamp}.csv"

            edges_small = self._edges[["a_x","a_y","b_x","b_y"]].copy()
            trk_small   = self._track[["tsec","x_km","y_km","lane"]].copy()

            self._export_thr = _ExportCongestionThread(
                self._assign_points_to_edges, edges_small, trk_small,
                out_path, SLD_MAX_SEC, 0, parent=self
            )
            self._export_thr.progressed.connect(lambda i, t: self._log(f"CSV 저장 중… {i}/{t}"))
            self._export_thr.done.connect(self._on_export_done)
            self._export_thr.error.connect(self._on_export_error)
            self._export_thr.start()
            self._log("CSV 저장 시작… (백그라운드)")
        except Exception as e:
            self._log(f"CSV 저장 시작 실패: {e}")

    def _on_export_done(self, path: str):
        self._export_done = True
        self._log(f"✔ 혼잡 결과 CSV 저장 완료: {path}")

    def _on_export_error(self, msg: str):
        self._log(f"⚠ 혼잡 결과 저장 실패: {msg}")
    
    def _get_earliest_tsec(self, df: Optional[pd.DataFrame]) -> int:
        """track df에서 가장 이른 tsec(초)을 돌려준다. 없으면 0(00:00:00)."""
        if df is None:
            return 0
        try:
            s = pd.to_numeric(df.get("tsec"), errors="coerce").dropna()
            if not s.empty:
                return int(s.min()) % SLD_MAX_SEC
        except Exception:
            pass
        return 0
    
    def _load_all(self):
        # VP/WP
        self._vp, self._wp = load_overlays(self.vp_csv, self.wp_csv)
        if self._vp is None or self._wp is None:
            self._log("⚠ Vertiport/Waypoint CSV를 찾지 못했습니다. Sources 경로를 확인하세요.")
            return

        # 에지(무방향) 생성 + 좌측(진행방향 기준) 법선
        self._edges = self._build_edges(self._wp)
        self._lanes_gdf = self._build_lane_geoms(self._edges)
        self._bounds = tuple(self._lanes_gdf.total_bounds)

        # 트랙
        if self.track_csv and Path(self.track_csv).exists():
            self._track = self._clean_track(pd.read_csv(self.track_csv))
        else:
            self._track = None

        # ▶ 기본 시각을 '가장 이른 tsec'으로
        self._tsec = self._get_earliest_tsec(self._track)
        self._update_time_hint()
        self._recompute_and_repaint()

        # ▶ 탭 초기 로딩 후 자동 저장 (이벤트 루프 다음 턴에 실행)
        if self._track is not None and self._edges is not None and not self._export_done:
            QTimer.singleShot(0, self._start_export_thread)

    # ───────────────────────── 지도 ─────────────────────────
    def _base_map(self):
        # Noise 탭과 유사한 초기도; 이후 fit_bounds로 네트워크에 맞춤
        return folium.Map([37.5665, 126.9780], zoom_start=12, tiles="CartoDB positron", control_scale=True)

    def _build_map_base(self):
        if self._lanes_gdf is None: return
        fmap = self._base_map()

        # 배경: VP/WP/링크
        self._add_vp_wp_and_links(fmap)

        # Halo(아래 깔리는 밝은 두꺼운 선)
        halo = folium.GeoJson(
            self._lanes_gdf[["lane_id","edge_id","lane","geometry"]].to_json(),
            name="lanes_halo",
            style_function=lambda f: {
                "color":"#ECEFF1","weight":7.0,"opacity":0.95,
                "lineCap":"round","lineJoin":"round"
            }
        )
        halo.add_to(fmap)

        # 본 레이어(혼잡 색칠 대상)
        lanes_layer = folium.GeoJson(
            self._lanes_gdf[["lane_id","edge_id","lane","geometry"]].to_json(),
            name="lanes",
            style_function=lambda f: {
                "color":"#B0BEC5","weight":4.0,"opacity":0.90,
                "lineCap":"round","lineJoin":"round"
            },
            tooltip=folium.GeoJsonTooltip(fields=["edge_id","lane"], aliases=["Edge","Lane"])
        )
        lanes_layer.add_to(fmap)

        # 컬러바
        self._cmap = cm.LinearColormap(
            # 낮음 -> 높음: 초록 -> 노랑 -> 오렌지 -> 빨강 계열
            ["#1a9850", "#66bd63", "#a6d96a", "#d9ef8b", "#fee08b",
            "#fdae61", "#f46d43", "#d73027", "#a50026", "#7f0000", "#4d0000"],
            vmin=0, vmax=10, caption="Link Congestion (0–10)"
        ).to_step(11)

        # JS 변수명
        self._map_js_name   = fmap.get_name()
        self._lanes_js_name = lanes_layer.get_name()

        # 경계 맞춤
        if self._bounds:
            minx, miny, maxx, maxy = [float(v) for v in self._bounds]
            fmap.fit_bounds([[miny, minx], [maxy, maxx]])

        # 렌더
        if self._html and os.path.exists(self._html):
            try: os.remove(self._html)
            except Exception: pass
        self._html = tempfile.NamedTemporaryFile(suffix=".html", delete=False).name
        fmap.save(self._html)
        self.web.load(QUrl.fromLocalFile(self._html))

    def _inject_js_api(self):
        if not (self._map_js_name and self._lanes_js_name): return
        js = build_js_api_traffic(self._map_js_name, self._lanes_js_name)
        run_js(self.web, js)

    def _on_web_loaded(self, ok: bool):
        self._web_ready = bool(ok)
        if not ok: self._log("페이지 로드 실패"); return
        self._inject_js_api()
        self._base_ready = True
        if self._last_vals or self._last_cols:
            update_traffic(self.web, self._last_vals, self._last_cols)

    # ───────────────────────── 계산/표 ─────────────────────────
    def _recompute_and_repaint(self):
        if self._track is None or self._edges is None:
            self._build_region_table(None, None, None, None)
            self._log('✅ <b>All Clear</b> 현재 과밀 구간 없음')
            return

        vals = self._compute_congestion(self._track, self._edges, self._tsec, 0)

        if self._lanes_gdf is not None:
            for lid in self._lanes_gdf["lane_id"].astype(str):
                if lid not in vals:
                    vals[lid] = 0.0

        cols = {k: self._cmap(v) for k, v in vals.items()} if vals else {}
        self._last_vals, self._last_cols = vals, cols
        if self._web_ready and self._base_ready:
            update_traffic(self.web, self._last_vals, self._last_cols)

        # 표도 동일 정책(평균은 전체 / 스냅샷은 해당 초)
        self._build_region_table(self._track, self._vp, self._tsec, 0)

        self._update_wp_lane_log(vals)

    # ───────────────────────── 편의 함수 ─────────────────────────
    def _update_time_hint(self):
        if self._tsec is None:
            self._tsec = self._get_earliest_tsec(self._track)  # ▶ 변경

        h = (self._tsec // 3600) % 24
        m = (self._tsec % 3600) // 60
        # 스핀/슬라이더 동기화 (기존 그대로)
        self.spin_h.blockSignals(True); self.spin_h.setValue(h); self.spin_h.blockSignals(False)
        self.spin_m.blockSignals(True); self.spin_m.setValue(m); self.spin_m.blockSignals(False)
        self.spin_s.blockSignals(True); self.spin_s.setValue(int(self._tsec % 60)); self.spin_s.blockSignals(False)
        self.slider.blockSignals(True); self.slider.setValue(int(self._tsec)); self.slider.blockSignals(False)


    def _on_time_changed(self, v:int):
        step = 1
        v = int(round(v/step)*step) % SLD_MAX_SEC
        if v == self._tsec: return
        self._tsec = v; self._update_time_hint(); self._throttle.start()

    def _on_spins_changed(self, *_):
        tsec = int(self.spin_h.value())*3600 + int(self.spin_m.value())*60 + int(self.spin_s.value())
        step = 1
        tsec = int(round(tsec/step)*step) % SLD_MAX_SEC
        if tsec != self._tsec: self.slider.setValue(tsec)
        

    def _pick_track(self):
        caption = "트랙 데이터 선택 (track_log.csv)"
        start = str(get_db_dir())
        path, _ = QFileDialog.getOpenFileName(self, caption, start, "CSV (*.csv)")
        if not path:
            return
        self.track_csv = Path(path)
        self._track = self._clean_track(pd.read_csv(self.track_csv))
        self._tsec = self._get_earliest_tsec(self._track)
        self._rebuild_and_repaint()

    # ───────────────────────── 데이터 처리 ─────────────────────────
    @staticmethod
    def _clean_track(df: pd.DataFrame) -> pd.DataFrame:
        from Functions.core import parse_tsec  # 이미 core에 있음

        df = df.copy()
        df.columns = [str(c).strip() for c in df.columns]

        # 좌표(km)
        df["x_km"] = pd.to_numeric(df.get("x"), errors="coerce") / 1000.0
        df["y_km"] = pd.to_numeric(df.get("y"), errors="coerce") / 1000.0

        # 레인 표기 정규화: ' l-2000 ' → 'L-2000'
        if "lane" in df.columns:
            df["lane"] = df["lane"].astype(str).str.strip().str.upper()

        # tsec 생성/정규화
        if "tsec" in df.columns:
            df["tsec"] = pd.to_numeric(df["tsec"], errors="coerce").astype("Int64")
        else:
            tcol = None
            for c in ["sim_time", "time", "timestamp", "ts", "t"]:
                if c in df.columns:
                    tcol = c; break
            if tcol is not None:
                df["tsec"] = df[tcol].apply(parse_tsec).astype("Int64")
            else:
                df["tsec"] = pd.Series([None]*len(df), dtype="Int64")

        return df


    @staticmethod
    def _build_edges(wp: pd.DataFrame) -> pd.DataFrame:
        """무방향 링크를 만들되, 진행방향은 남(y↓)→북(y↑)으로 정규화하고 좌측 법선을 저장."""
        wp_map = {str(r["Waypoint 명"]).strip(): r for _, r in wp.iterrows()}
        seen = set(); rows = []
        for _, r in wp.iterrows():
            s = str(r["Waypoint 명"]).strip()
            links = str(r.get("Link","")).split(",")
            for t in map(str.strip, links):
                if not t or t not in wp_map: 
                    continue
                undirected = frozenset((s,t))
                if undirected in seen: 
                    continue
                seen.add(undirected)

                a_row, b_row = wp_map[s], wp_map[t]
                # y(km) 작은 쪽을 a(남), 큰 쪽을 b(북)
                if float(a_row["y (km)"]) <= float(b_row["y (km)"]):
                    a_name, b_name = s, t; a, b = a_row, b_row
                else:
                    a_name, b_name = t, s; a, b = b_row, a_row

                rows.append(dict(
                    edge_id=len(rows),
                    a=a_name, b=b_name,
                    a_lat=float(a["위도"]), a_lon=float(a["경도"]),
                    b_lat=float(b["위도"]), b_lon=float(b["경도"]),
                    a_x=float(a["x (km)"]), a_y=float(a["y (km)"]),
                    b_x=float(b["x (km)"]), b_y=float(b["y (km)"]),
                ))
        ed = pd.DataFrame(rows)
        dx = ed["b_x"].to_numpy() - ed["a_x"].to_numpy()
        dy = ed["b_y"].to_numpy() - ed["a_y"].to_numpy()
        L  = np.sqrt(dx*dx + dy*dy) + 1e-9
        ed["uxp"] = -dy / L   # 진행방향 기준 '좌측' 단위법선
        ed["uyp"] =  dx / L
        return ed

        # XY 평면에서 오프셋 선을 만든 후, 각 웨이포인트×레인 공통 앵커로 스냅해 끊김 제거
    # Traffic_tab.py 안
    def _build_lane_geoms(self, edges: pd.DataFrame) -> gpd.GeoDataFrame:
        wp_map = {str(r["Waypoint 명"]).strip(): r for _, r in self._wp.iterrows()}

        rec = []
        for _, e in edges.iterrows():
            a_name = str(e["a"]).strip()
            b_name = str(e["b"]).strip()
            for lane in LANES:
                (ax, ay), (bx, by) = self._lane_offsets_xy(e, lane)
                rec.append({
                    "edge_id": int(e["edge_id"]), "lane": lane,
                    "a_node": a_name, "b_node": b_name,
                    "a_xy": (ax, ay), "b_xy": (bx, by),
                })

        def xy_to_ll(node_name: str, x_km: float, y_km: float):
            base = wp_map.get(node_name)
            if base is None:
                return (y_km, x_km)  # fallback
            bx, by = float(base["x (km)"]), float(base["y (km)"])
            blat, blon = float(base["위도"]), float(base["경도"])
            dE, dN = (x_km - bx), (y_km - by)
            lat, lon = _offset_latlon(blat, blon, dE, dN)
            return (lat, lon)

        rows = []
        for r in rec:
            a_lat, a_lon = xy_to_ll(r["a_node"], r["a_xy"][0], r["a_xy"][1])
            b_lat, b_lon = xy_to_ll(r["b_node"], r["b_xy"][0], r["b_xy"][1])
            line = LineString([(a_lon, a_lat), (b_lon, b_lat)])
            rows.append({
                "lane_id": f"{r['edge_id']}|{r['lane']}",
                "edge_id": r["edge_id"], "lane": r["lane"],
                "geometry": line
            })

        return gpd.GeoDataFrame(rows, crs="EPSG:4326")


    def _lane_offsets_xy(self, e_row, lane: str, base_m: float=65.0, sep_m: float=45.0):
        """(edge, lane)을 XY(km)에서 오프셋한 '확장' 선분(양 끝을 소폭 연장). 1000=안쪽, 2000=바깥쪽; L=왼쪽, R=오른쪽."""
        is_right = lane.startswith("R")
        is2000   = ("2000" in lane)

        dx_km = float(e_row["b_x"]) - float(e_row["a_x"])
        dy_km = float(e_row["b_y"]) - float(e_row["a_y"])
        L_km  = math.hypot(dx_km, dy_km) + 1e-9

        max_off_m = max(20.0, 0.35 * L_km * 1000.0)
        base_m    = min(base_m, max_off_m)
        sep_m     = min(sep_m,  max_off_m * 0.6)
        d_km      = (base_m + (sep_m if is2000 else 0.0)) / 1000.0

        nx, ny = float(e_row["uxp"]), float(e_row["uyp"])        # 좌측 법선
        if is_right: nx, ny = -nx, -ny                           # 오른쪽은 부호 반전

        ax = float(e_row["a_x"]) + nx * d_km
        ay = float(e_row["a_y"]) + ny * d_km
        bx = float(e_row["b_x"]) + nx * d_km
        by = float(e_row["b_y"]) + ny * d_km

        # 단절 방지용 링크 방향 ±연장(최소 25m, 길이의 1% 사용)
        ext_km = max(0.025, 0.010 * L_km)
        ux, uy = dx_km / L_km, dy_km / L_km
        ax -= ux * ext_km; ay -= uy * ext_km
        bx += ux * ext_km; by += uy * ext_km
        return (ax, ay), (bx, by)

    @staticmethod
    def _line_intersection_xy(p1, p2, p3, p4, eps=1e-9):
        """무한직선 p1-p2, p3-p4 교점(XY km). 평행/수치불안정 시 None."""
        x1,y1 = p1; x2,y2 = p2; x3,y3 = p3; x4,y4 = p4
        den = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
        if abs(den) < eps: 
            return None
        px = ((x1*y2 - y1*x2)*(x3-x4) - (x1-x2)*(x3*y4 - y3*x4)) / den
        py = ((x1*y2 - y1*x2)*(y3-y4) - (y1-y2)*(x3*y4 - y3*x4)) / den
        return (float(px), float(py))

    def _assign_points_to_edges(self, pts_xy: np.ndarray, edges: pd.DataFrame) -> np.ndarray:
        """점(x_km,y_km) → 최근접 에지 인덱스 (벡터화)"""
        x1 = edges["a_x"].to_numpy(); y1 = edges["a_y"].to_numpy()
        x2 = edges["b_x"].to_numpy(); y2 = edges["b_y"].to_numpy()
        dx = x2 - x1; dy = y2 - y1; L2 = dx*dx + dy*dy + 1e-12
        N = pts_xy.shape[0]; E = edges.shape[0]
        out = np.empty(N, dtype=np.int32)
        bsz = 20000
        for i in range(0, N, bsz):
            P = pts_xy[i:i+bsz]
            px = P[:,0][:,None]; py = P[:,1][:,None]
            t = ((px - x1)*dx + (py - y1)*dy) / L2
            t = np.clip(t, 0, 1)
            projx = x1 + t*dx; projy = y1 + t*dy
            d2 = (px - projx)**2 + (py - projy)**2
            idx = np.argmin(d2, axis=1)
            out[i:i+bsz] = idx
        return out

    def _compute_congestion(self, track: pd.DataFrame, edges: pd.DataFrame,
                            tsec: Optional[int], win_sec: int) -> Dict[str,float]:
        df = track.copy()

        # lane 정규화
        if "lane" in df.columns:
            df["lane"] = df["lane"].astype(str).str.strip().str.upper()

        df = df[df["lane"].isin(LANES)].dropna(subset=["x_km","y_km"])
        if df.empty: return {}

   
        # 시간 필터 (교체)
        if tsec is not None and "tsec" in df.columns:
            # 숫자화 + 하루 모듈러(0..86399)
            t = pd.to_numeric(df["tsec"], errors="coerce")
            df = df[t.notna()].copy()
            t24 = (t.loc[df.index] % SLD_MAX_SEC).astype(int)

            width = int(win_sec or 0)
            ts = int(tsec) % SLD_MAX_SEC
            if width <= 0:
                mask = (t24 == ts)
            else:
                half = width // 2
                lo = (ts - half) % SLD_MAX_SEC
                hi = (ts + half) % SLD_MAX_SEC
                mask = ((t24 >= lo) & (t24 < hi)) if lo <= hi else ((t24 >= lo) | (t24 < hi))
            df = df[mask]
        if df.empty: return {}
        
        # 최근접 에지 배정
        idx = self._assign_points_to_edges(df[["x_km","y_km"]].to_numpy(), edges)
        E = edges.shape[0]; vals: Dict[str, float] = {}

        lane_arr = df["lane"].to_numpy()
        m1000 = (lane_arr == "L-1000") | (lane_arr == "R-1000")
        m2000 = (lane_arr == "L-2000") | (lane_arr == "R-2000")
        c1000 = np.bincount(idx[m1000], minlength=E).astype(int) if m1000.any() else np.zeros(E, int)
        c2000 = np.bincount(idx[m2000], minlength=E).astype(int) if m2000.any() else np.zeros(E, int)

        ref = np.concatenate([c1000[c1000>0], c2000[c2000>0]])
        p95 = float(np.quantile(ref, 0.95)) if ref.size else 1.0
        if p95 <= 0: p95 = 1.0

        for eid in range(E):
            d1 = max(0.0, min(10.0, 10.0 * (c1000[eid] / p95)))
            d2 = max(0.0, min(10.0, 10.0 * (c2000[eid] / p95)))
            vals[f"{eid}|L-1000"] = d1; vals[f"{eid}|R-1000"] = d1
            vals[f"{eid}|L-2000"] = d2; vals[f"{eid}|R-2000"] = d2

        return vals


    def _build_region_table(self, track: Optional[pd.DataFrame], vp: Optional[pd.DataFrame],
                            tsec: Optional[int], win_sec: Optional[int]):
        self.tbl.setSortingEnabled(False)
        self.tbl.clear()
        self.tbl.setColumnCount(3)
        self.tbl.setHorizontalHeaderLabels(["지역(Vertiport)", "TO (B–E)", "LD (G–J)"])
        self.tbl.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.tbl.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.tbl.verticalHeader().setVisible(False)

        if track is None or vp is None or track.empty or vp.empty:
            self.tbl.setRowCount(0); 
            self.tbl_title.setText("표: 지역별 TO/LD (데이터 없음)")
            return

        # 좌표 유효치만
        df = track.copy()
        df = df[df["x_km"].notna() & df["y_km"].notna()].copy()
        if tsec is not None and "tsec" in df.columns:
            t = pd.to_numeric(df["tsec"], errors="coerce")
            df = df[t.notna()].copy()
            t24 = (t.loc[df.index] % SLD_MAX_SEC).astype(int)

            width = int(win_sec or 0)
            ts = int(tsec) % SLD_MAX_SEC
            if width <= 0:
                mask = (t24 == ts)
            else:
                half = width // 2
                lo = (ts - half) % SLD_MAX_SEC
                hi = (ts + half) % SLD_MAX_SEC
                mask = ((t24 >= lo) & (t24 < hi)) if lo <= hi else ((t24 >= lo) | (t24 < hi))
            df = df[mask]

        # ▶ phase 기반 분류 (lane 조건 제거)
        df["phase"] = df.get("phase", "").astype(str).str.strip().str.upper()
        to_set, ld_set = set("BCDE"), set("GHIJ")
        def _cls(p): return "TO" if p in to_set else ("LD" if p in ld_set else None)
        df["cls"] = df["phase"].map(_cls)
        df = df[df["cls"].notna()]
        if df.empty:
            h = (tsec // 3600) % 24 if tsec is not None else 0
            m = (tsec % 3600) // 60 if tsec is not None else 0
            s = (tsec % 60) if tsec is not None else 0
            self.tbl.setRowCount(0)
            self.tbl_title.setText(f"표: 지역별 TO/LD (분류 결과 없음 – {h:02d}:{m:02d}:{s:02d})")
            return

        # VP 최근접 + MTR 이내
        vx, vy = vp["x (km)"].to_numpy(), vp["y (km)"].to_numpy()
        vr = vp.get("MTR(km)", pd.Series([1.0]*len(vp))).fillna(1.0).to_numpy()
        names = vp["Vertiport 명"].astype(str).tolist()

        P = df[["x_km","y_km"]].to_numpy()
        N = P.shape[0]; near = np.full(N, -1, dtype=int)
        bsz = 50000
        for i in range(0, N, bsz):
            Q = P[i:i+bsz]; px = Q[:,0][:,None]; py = Q[:,1][:,None]
            d2 = (px - vx)**2 + (py - vy)**2
            near[i:i+bsz] = np.argmin(d2, axis=1)
        dist = np.sqrt((P[:,0] - vx[near])**2 + (P[:,1] - vy[near])**2)
        df = df.assign(vp_idx=near, within=(dist <= vr[near]))
        df = df[df["within"]]
        if df.empty:
            h = (tsec // 3600) % 24 if tsec is not None else 0
            m = (tsec % 3600) // 60 if tsec is not None else 0
            s = (tsec % 60) if tsec is not None else 0
            self.tbl.setRowCount(0)
            self.tbl_title.setText(f"표: 지역별 TO/LD (MTR 범위 내 데이터 없음 – {h:02d}:{m:02d}:{s:02d})")
            return

        # ▶ 초 단위 중복 제거
        id_col = "reg" if "reg" in df.columns else ("id" if "id" in df.columns else None)
        if id_col:
            df = df.drop_duplicates(subset=["vp_idx", id_col, "cls"])
        elif "tsec" in df.columns:
            df["_tb"] = df["tsec"].fillna(0).astype(int)
            df = df.drop_duplicates(subset=["vp_idx", "_tb", "cls"])

        g = df.groupby(["vp_idx","cls"]).size().unstack(fill_value=0)
        rows = []
        for i, name in enumerate(names):
            to = int(g.loc[i]["TO"]) if i in g.index and "TO" in g.columns else 0
            ld = int(g.loc[i]["LD"]) if i in g.index and "LD" in g.columns else 0
            rows.append((name, to, ld))

        self.tbl.setRowCount(len(rows))
        for r, (nm, a, b) in enumerate(rows):
            self.tbl.setItem(r, 0, QTableWidgetItem(nm))
            self.tbl.setItem(r, 1, QTableWidgetItem(f"{a:,d}"))
            self.tbl.setItem(r, 2, QTableWidgetItem(f"{b:,d}"))
        # ▶ 표 제목에 선택 시각 표시
        h = (tsec // 3600) % 24 if tsec is not None else 0
        m = (tsec % 3600) // 60 if tsec is not None else 0
        s = (tsec % 60) if tsec is not None else 0
        self.tbl_title.setText(f"표: 지역별 TO/LD – {h:02d}:{m:02d}:{s:02d}")
        self.tbl.setSortingEnabled(True)


    # 배경 요소
    def _add_vp_wp_and_links(self, fmap: folium.Map):
        if self._vp is not None:
            for _, r in self._vp.iterrows():
                name = str(r.get("Vertiport 명",""))
                lat  = float(r.get("위도")); lon = float(r.get("경도"))
                folium.CircleMarker([lat, lon], radius=5, color="#1a237e",
                                    fill=True, fill_color="#448aff", fill_opacity=.9,
                                    popup=f"Vertiport {name}").add_to(fmap)
                inr = float(r.get("INR(km)", 0) or 0); mtr = float(r.get("MTR(km)", 0) or 0)
                if inr > 0: folium.Circle([lat, lon], radius=inr*1000.0, color="#2e7d32", weight=1.5, fill=False, opacity=0.8).add_to(fmap)
                if mtr > 0: folium.Circle([lat, lon], radius=mtr*1000.0, color="#8e24aa", weight=1.0, fill=False, opacity=0.5, dash_array="5,5").add_to(fmap)

        dgeo = {}
        for _, r in self._wp.iterrows():
            name = str(r.get("Waypoint 명","")).strip()
            if not name: continue
            dgeo[name] = (float(r.get("위도")), float(r.get("경도")))
        drawn = set()
        for _, r in self._wp.iterrows():
            s = str(r.get("Waypoint 명","")).strip()
            for t in map(str.strip, str(r.get("Link","")).split(",")):
                if not t or s not in dgeo or t not in dgeo: continue
                key = tuple(sorted([s,t]))
                if key in drawn: continue
                drawn.add(key)
                folium.PolyLine([dgeo[s], dgeo[t]], color="#1565c0", weight=1.5, opacity=0.25).add_to(fmap)

    # 로그 출력(QTextBrowser)
    def _log(self, msg: str):
        if "<" not in msg:
            msg = msg.replace("\n","<br>")
        self.log_view.setHtml(msg)

    # 웨이포인트×레인 로그
    def _update_wp_lane_log(self, vals: Dict[str, float]) -> None:
        # ▶ 선택 시각 문자열
        if self._tsec is not None:
            h = (self._tsec // 3600) % 24
            m = (self._tsec % 3600) // 60
            s = self._tsec % 60
            t_exact = f"{h:02d}:{m:02d}:{s:02d}"
        else:
            t_exact = "--:--:--"

        if not vals:
            self._log(f'✅ <b>All Clear</b> {t_exact} 현재 과밀 구간 없음')
            return

        edges_idx = getattr(self, "_edges_idx", None)
        if edges_idx is None:
            try:
                edges_idx = self._edges.set_index("edge_id")
                self._edges_idx = edges_idx
            except Exception:
                self._log(f'✅ <b>All Clear</b> {t_exact} 현재 과밀 구간 없음')
                return
            
        wp_lane_max = {}  # {(wp_name, lane): max_density}
        for k, v in vals.items():
            try:
                eid_str, lane = k.split("|")
                eid = int(eid_str)
                row = edges_idx.loc[eid]
                for wp in (str(row["a"]), str(row["b"])):
                    key = (wp, lane)
                    wp_lane_max[key] = max(wp_lane_max.get(key, 0.0), float(v))
            except Exception:
                continue

        if not wp_lane_max:
            self._log('✅ <b>All Clear</b> 현재 과밀 구간 없음')
            return

        lv1, lv2, lv3 = [], [], []
        for (wp, lane), d in sorted(wp_lane_max.items(), key=lambda kv: -kv[1]):
            if d > 8.0:   lv3.append((wp, lane, d))
            elif d > 5.0: lv2.append((wp, lane, d))
            elif d > 2.0: lv1.append((wp, lane, d))

        TOPN = 12
        parts = []
        def add_block(icon, title, rows):
            if not rows: return
            parts.append(f'<p style="margin:4px 0;"><b>{icon} {title}</b></p>')
            for wp, lane, d in rows[:TOPN]:
                parts.append(f'<p style="margin:0 0 0 1.0em;">• {wp} '
                             f'<span style="color:#9aa7b1;">({lane})</span> : <b>{d:.2f}</b></p>')

        # Traffic_tab.py 안 _update_wp_lane_log() 헤더 생성부 교체
        if self._tsec is not None:
            h = (self._tsec // 3600) % 24
            m = (self._tsec % 3600) // 60
            s = self._tsec % 60
            t_exact = f"{h:02d}:{m:02d}:{s:02d}"
        else:
            t_exact = "--:--:--"

        add_block("🔴", f"혼잡도 Lv3 (고밀도) {t_exact}", lv3)
        add_block("🟠", f"혼잡도 Lv2 (중밀도) {t_exact}", lv2)
        add_block("🟢", f"혼잡도 Lv1 (저밀도) {t_exact}", lv1)

                
        if not parts:
            parts.append(f'<p>✅ <b>All Clear</b> {t_exact} 현재 과밀 구간 없음</p>')

        self._log("".join(parts))
