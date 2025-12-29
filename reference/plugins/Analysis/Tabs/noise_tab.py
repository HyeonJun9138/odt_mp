#!/usr/bin/env python
"""
NoiseTab  ―  UAM Traffic GUI ▸ 소음(Noise) 분석 탭
────────────────────────────────────────────────────────────
• 실시간 비행 데이터(UDP) → 소음 레벨 산출(최대 dB)
• Leaflet 지도 위에
    1) 소음 원(circle) 컨투어
    2) DEM-기반 소음 등고선(polygons)
    3) NLSP 인구 격자(SHP)를 GeoJSON으로 토글
• 결과 테이블: {VID, Max dB @ RADIUS_M}
────────────────────────────────────────────────────────────
구조
    ├─ CruiseNoiseCalculator     : 단순 소음 감소 모델
    ├─ NoiseContourWidget        : Folium+Leaflet 지도 위젯
    ├─ NoiseTab(Tab)             : GUI 탭(컨투어+테이블)
    └─ _NoiseWorker(QThread)     : 백그라운드 풀(Pool) 계산
"""

from __future__ import annotations

# ── 표준 라이브러리 ───────────────────────────────────────────
import json
import math
import os
import tempfile
import csv
import atexit
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Tuple

from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool  # (I/O 가벼워서 ThreadPool 사용)

# ── 외부 라이브러리 ───────────────────────────────────────────
from PyQt5.QtCore    import Qt, QTimer, QUrl, pyqtSignal, pyqtSlot, QThread
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QCheckBox, QHeaderView
)
from PyQt5.QtWebEngineWidgets import QWebEngineView
import folium
import branca.colormap as cm
import geopandas as gpd
import numpy as np

from Analysis.Functions.terrain_noise import *          # ▸ RADIUS_M, compute_noise_for_vehicle, …

from Tabs.base_tab import Tab


# ─────────────────────────────────────────────────────────────
# 0) 경로 설정 ― 프로젝트 루트/Analysis/resource
#    (Path 객체를 쓰면 OS·PC 달라도 경로 꼬임 방지)
# ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESOURCE_DIR = PROJECT_ROOT / "Analysis" / "Sources"


# ─────────────────────────────────────────────────────────────
# 1) 프로세스 풀 관리 (전역 1회만 생성)
# ─────────────────────────────────────────────────────────────
_GLOBAL_POOL: ThreadPool | None = ThreadPool(
    processes=max(1, cpu_count() - 2)  # CPU 2개 남겨두고 사용
)

def _cleanup_pool() -> None:
    """애플리케이션 종료 시 ThreadPool 안전 종료."""
    pool = globals().get("_GLOBAL_POOL")
    if pool is not None:
        pool.close(); pool.join()

atexit.register(_cleanup_pool)


# ─────────────────────────────────────────────────────────────
# 2) 소음 계산 유틸
# ─────────────────────────────────────────────────────────────
class CruiseNoiseCalculator:
    """간단한 거리 감쇠(dSPL) 모델.

    L(d) = L0 - 20·log10(d/d0) - α(d-d0)
    • L0   : 기준 거리(d0)에서 레벨 [dB]
    • d0   : 기준 거리 [m]
    • α    : 추가 감쇠 계수 [dB/m]
    """

    def __init__(self, L0: float = 60.0, d0: float = 300.0,
                 alpha: float = 0.0001) -> None:
        self.L0, self.d0, self.alpha = L0, d0, alpha

    def get_level(self, ds: float) -> float:
        """거리 ds(m)의 소음 레벨[dB] 반환."""
        return (
            self.L0
            - 20 * math.log10(ds / self.d0)
            - self.alpha * (ds - self.d0)
        )


# ─────────────────────────────────────────────────────────────
# 3) 지도 위젯 (Folium → Leaflet)
# ─────────────────────────────────────────────────────────────
class NoiseContourWidget(QWebEngineView):
    """Folium 지도를 렌더링한 뒤 JS 함수를 통해 실시간 오버레이 갱신."""

    # ---------- 초기화 ----------
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        # temp HTML 파일: reload 없이 JS 갱신만 하므로 delete=False
        self._html = tempfile.NamedTemporaryFile(
            suffix=".html", delete=False
        ).name
        self._js_ready = False  # JS 주입 완료 flag
        self._build_base_map()
        self.loadFinished.connect(self._init_js)

    # ---------- Public API ----------
    def update_contours(self, raw: list) -> None:
        """소음 원(circle) 리스트 JS로 전달."""
        if self._js_ready:
            self.page().runJavaScript(
                f"updateContoursJS({json.dumps(raw)})"
            )

    def update_polygons(self, payload: list) -> None:
        """DEM 기반 등고선 폴리곤 리스트 JS로 전달."""
        if self._js_ready:
            self.page().runJavaScript(
                f"updatePolygonsJS({json.dumps(payload)})"
            )

    def show_shp(self, on: bool,
                 geojson: dict | None = None,
                 legend_html: str = "") -> None:
        """인구 격자(SHP) 온/오프."""
        if self._js_ready:
            gjs = json.dumps(geojson) if geojson else "null"
            lhtml = json.dumps(legend_html)
            self.page().runJavaScript(
                f"toggleShpLayer({str(on).lower()}, {gjs}, {lhtml});"
            )

    # ---------- 내부: Folium 뼈대 생성 ----------
    def _build_base_map(self) -> None:
        """Folium으로 기본 지도(html) 생성."""
        center = [37.5665, 126.9780]  # 서울 시청
        fmap = folium.Map(
            center,
            zoom_start=12,
            tiles="CartoDB positron",
            prefer_canvas=True
        )

        # 컬러바(50~70dB, green→yellow→red)
        thresholds = list(range(50, 71))
        cmap = cm.LinearColormap(
            ["green", "yellow", "red"], vmin=50, vmax=70
        ).to_step(len(thresholds) - 1)
        self.cmap = cm.StepColormap(
            cmap.colors, index=thresholds, vmin=50, vmax=70,
            caption="소음 레벨 (dB)"
        ).add_to(fmap)

        self._map_name = fmap.get_name()  # JS 객체명
        fmap.save(self._html)
        self.load(QUrl.fromLocalFile(self._html))

    # ---------- 내부: JS 주입 ----------
    def _init_js(self, ok: bool) -> None:
        """HTML load 완료 후 Leaflet 커스텀 JS 삽입."""
        if not ok:
            return

        # ※ Python f-string으로 JS 코드 작성
        js = f"""
            /* ------------ 1. 소음 원(Circle) 레이어 ------------ */
            window.contourGroup = L.layerGroup().addTo({self._map_name});
            function updateContoursJS(raw) {{
                contourGroup.clearLayers();
                raw.forEach(function(it) {{
                    var lat = it[0], lon = it[1], r = it[2], col = it[3];
                    L.circle([lat, lon], {{
                        radius: r,
                        color: col, fill: false,
                        weight: 2, opacity: 0.6
                    }}).addTo(contourGroup);
                }});
            }}

            /* ---- 2. DEM 기반 소음 등고선(Polygon) 레이어 ---- */
            window.polygonGroup = L.layerGroup().addTo({self._map_name});
            function updatePolygonsJS(raw) {{
                /* 기존 레이어 제거 & 컨테이너 제거 */
                if (window.polygonGroup) {{
                    {self._map_name}.removeLayer(polygonGroup);
                    const c = window.polygonGroup._container;
                    if (c && c.parentNode) L.DomUtil.remove(c);
                }}
                window.polygonGroup = L.layerGroup().addTo({self._map_name});

                raw.forEach(function(it) {{
                    var coords = it[0], col = it[1];
                    L.polygon(coords, {{
                        color: col, weight: 1.2,
                        fill: false, opacity: 0.9
                    }}).addTo(polygonGroup);
                }});
            }}

            /* ------------ 3. NLSP 인구 격자 + 범례 ------------ */
            window.popLayer  = null;
            window.popLegend = null;
            function toggleShpLayer(show, geojson, legendHTML) {{
                if (show) {{
                    if (window.popLayer === null) {{
                        /* 3-1. GeoJSON 레이어 */
                        function styleFn(feat) {{
                            return {{
                                fillColor: feat.properties.fillColor || '#cccccc',
                                fillOpacity: 0.45,
                                stroke: false,
                                color: '#4a79ff',
                                weight: 0.8
                            }};
                        }}
                        window.popLayer = L.geoJSON(geojson, {{style: styleFn}})
                                          .addTo({self._map_name});

                        /* 3-2. 범례 추가 */
                        window.popLegend = L.control({{position: 'bottomleft'}});
                        popLegend.onAdd = function(m) {{
                            var div = L.DomUtil.create('div', 'info legend');
                            div.innerHTML = legendHTML;
                            return div;
                        }};
                        popLegend.addTo({self._map_name});
                    }}
                }} else {{
                    if (window.popLayer)  {{
                        {self._map_name}.removeLayer(popLayer);
                        popLayer = null;
                    }}
                    if (window.popLegend) {{
                        {self._map_name}.removeControl(popLegend);
                        popLegend = null;
                    }}
                }}
            }}
        """
        self.page().runJavaScript(js, lambda _: self._flag_ready())

    def _flag_ready(self) -> None:
        """JS 삽입 완료 플래그."""
        self._js_ready = True

    # ---------- 정리 ----------
    def closeEvent(self, ev) -> None:
        """temp HTML 파일 삭제."""
        if os.path.exists(self._html):
            os.remove(self._html)
        super().closeEvent(ev)


# ─────────────────────────────────────────────────────────────
# 4) 메인 Noise 탭
# ─────────────────────────────────────────────────────────────
class NoiseTab(Tab):
    """• 지도(좌) + 테이블(우) 2-패널 구성
       • UDP 수신 데이터 누적 → 3s 주기 ThreadPool 계산
    """

    # ---------- 초기화 ----------
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Noise", parent)


        # 테이블 설정
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents
        )
        self.table.horizontalHeader().setStretchLastSection(True)

        # (2) 데이터/로직
        self._data: Dict[str, Tuple[float, float, float]] = {}
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._refresh)
        self._noise_calc = CruiseNoiseCalculator()
        self._nlsp_geojson = None
        self._pool = _GLOBAL_POOL                   # 전역 ThreadPool 재사용
        self._worker: _NoiseWorker | None = None

        # Folium HTML 로딩 완료 후 타이머 시작
        QTimer.singleShot(0, lambda:
            self.contour.loadFinished.connect(self._start_timer))

    # ---------- GUI 헬퍼 ----------
    def create_map_widget(self):
        self.contour = NoiseContourWidget(self)

        wrapper = QWidget(self)
        vbox = QVBoxLayout(wrapper)
        vbox.setContentsMargins(0, 0, 0, 0)

        self.chk_nlsp = QCheckBox("NLSP 인구지도 표시", wrapper)
        self.chk_nlsp.stateChanged.connect(self._on_toggle_nlsp)

        vbox.addWidget(self.chk_nlsp, alignment=Qt.AlignLeft)
        vbox.addWidget(self.contour)

        return wrapper  # BaseTab 이 왼쪽 패널로 사용하는 위젯
    def _start_timer(self, ok: bool) -> None:
        """HTML이 정상 로드되면 3s 주기 타이머 스타트."""
        if ok and not self._timer.isActive():
            self._timer.start(1000)

    # ---------- UDP 콜백 ----------
    def process_new_data_packet(self, vid: str, ac: dict) -> None:
        """SitlSim → UDP 데이터 수신."""
        lat, lon = ac.get("lat"), ac.get("lon")
        alt = ac.get("alt_m", ac.get("z", 0.0))
        if lat is not None and lon is not None:
            self._data[vid] = (lat, lon, alt)
        # 시뮬레이션 시간을 저장해 두면 추후 로그 등에 활용
        if "time" in ac:
            self._curr_time = ac["time"]

    def remove_vehicle(self, vid: str) -> None:
        """비행체 종료 시 데이터 제거."""
        self._data.pop(vid, None)

    # ---------- 주기 업데이트 ----------
    def _refresh(self) -> None:
        """3초마다: 현재 데이터 스냅샷 → 워커에게 위임."""
        if self._worker and self._worker.isRunning():
            return  # 이전 계산이 아직 끝나지 않음

        snapshot = self._data.copy()  # 스레드 안전 복사
        self._worker = _NoiseWorker(snapshot, self)
        self._worker.done.connect(self._apply_results)
        self._worker.start()

    # ---------- 결과 반영 ----------
    @pyqtSlot(list, list, dict)
    def _apply_results(self,
                       poly_js: list,
                       raw_js: list,
                       levels: dict[str, float]) -> None:
        """워커 결과를 지도 + 테이블에 표시."""
        # 지도 (draw throttling: 1Hz)
        if time.time() - getattr(self, "_last_draw", 0) < 1.0:
            return
        self._last_draw = time.time()
        self.contour.update_contours(raw_js)
        self.contour.update_polygons(poly_js)


    # ---------- NLSP 인구 지도 토글 ----------
    def _on_toggle_nlsp(self, state: int) -> None:
        """체크박스 토글 → 최초 로드시 GeoJSON 생성."""
        show = state == Qt.Checked

        # ① 최초 ON: SHP → GeoJSON 변환 + legend HTML 생성
        if show and self._nlsp_geojson is None:
            shp_path = RESOURCE_DIR / "nlsp_020001001.shp"
            try:
                gdf = gpd.read_file(str(shp_path), encoding="cp949")

                # 좌표계 WGS84 변환
                if gdf.crs and gdf.crs.to_epsg() != 4326:
                    gdf = gdf.to_crs(epsg=4326)

                # 인구수 컬럼 → 컬러 매핑
                pop_col = "val"
                vals = gdf[pop_col].fillna(0)
                vmin, vmax = float(vals.min()), float(vals.max())
                cmap = cm.LinearColormap(
                    ['#fff5eb', '#fee6ce', '#fdae6b',
                     '#fd8d3c', '#e6550d', '#a63603'],
                    vmin=vmin, vmax=vmax
                )
                gdf["fillColor"] = vals.apply(cmap)

                # 범례 HTML (branca 0.7+)
                if hasattr(cmap, "caption_html"):
                    legend_html = cmap.caption_html(
                        title="인구수", label_fmt="{:.0f}"
                    )
                else:  # branca 0.5–0.6
                    legend_html = cmap._repr_html_()
                    legend_html = legend_html.replace(
                        "<div style=",
                        "<strong>인구수</strong><br><div style=",
                        1
                    )

                self._nlsp_geojson = json.loads(gdf.to_json())
                self._legend_html = legend_html

            except Exception as e:
                print("[NLSP] load failed:", e)
                self.chk_nlsp.setChecked(False)
                return

        # ② JS 레이어 토글 호출
        self.contour.show_shp(show, self._nlsp_geojson, self._legend_html)


# ─────────────────────────────────────────────────────────────
# 5) 백그라운드 계산 쓰레드
# ─────────────────────────────────────────────────────────────
class _NoiseWorker(QThread):
    """스냅샷된 비행체 목록을 받아 ThreadPool로 병렬 소음 계산."""
    done = pyqtSignal(list, list, dict)  # poly_js, raw_js, levels

    def __init__(self,
                 data_snap: Dict[str, Tuple[float, float, float]],
                 parent: NoiseTab | None = None) -> None:
        super().__init__(parent)
        self._data = data_snap
        self._thresholds = list(range(50, 70, 2))  # 등고선 단계

    def run(self) -> None:
        """QThread 진입점 → ThreadPool.map 병렬 처리."""
        # (1) 입력 변환
        inputs = [
            (vid, lat, lon, alt, self._thresholds)
            for vid, (lat, lon, alt) in self._data.items()
        ]

        poly_js, raw_js, levels = [], [], {}

        # (2) ThreadPool: compute_noise_for_vehicle 함수 호출
        pool = self.parent()._pool
        n_proc = pool._processes or 1
        chunksz = max(32, len(inputs) // (n_proc * 2))

        for vid, poly_list, raw_list, max_db in pool.imap_unordered(
                compute_noise_for_vehicle, inputs, chunksize=chunksz):
            poly_js.extend(poly_list)   # 지도 폴리곤
            raw_js.extend(raw_list)     # 지도 원
            levels[vid] = max_db        # 테이블 값
            
        # ── (추가) 격자 로그 기록 ─────────────────────────
            parent = self.parent()
            hms = getattr(parent, "_curr_time",
                           datetime.now().strftime("%H:%M:%S"))
            lat, lon, _ = self._data[vid]
            log_grid_noise(hms, lat, lon, max_db)

        # (3) GUI 스레드에 결과 전달
        self.done.emit(poly_js, raw_js, levels)
