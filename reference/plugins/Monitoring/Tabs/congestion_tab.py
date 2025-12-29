# -*- coding: utf-8 -*-
"""
CongestionTab
─────────────
index 0 : Folium( OSM ) – Vertiport / Waypoint / 원·부채꼴 / 링크
index 1 : Matplotlib Node-Link ( PathVisualizerGeo )
"""

import math
import os
import tempfile
from pathlib import Path
import csv, datetime, tempfile

import folium
from PyQt5.QtCore import Qt, QUrl, pyqtSignal, QObject

from PyQt5.QtWidgets import (
    QWidget, QLabel, QPushButton, QSlider,
    QHBoxLayout, QVBoxLayout, QFrame,
    QSizePolicy, QStackedWidget
)
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtWidgets import QTableWidget, QTableWidgetItem
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from Monitoring.Functions import PathPlanner, PathVisualizerGeo



# ───────────────────────────────────────────────────────────────
#  MapView : OpenStreetMap + 전체 네트워크 시각화
# ───────────────────────────────────────────────────────────────
class MapView(QWebEngineView):
    def __init__(self, planner: PathPlanner, zoom=11, parent=None):
        super().__init__(parent)
        self._tmp_path = None
        self._create_map(planner, zoom)

    # ----------------------------------------------------------
    def _sector(self, lon0, lat0, radius_m, bearing_deg, spread_deg=10, n_pts=12):
        """중심/반경/방위각 → 부채꼴 좌표 리스트(lon,lat)."""
        pts = []
        for d in (bearing_deg - spread_deg, bearing_deg + spread_deg):
            for t in range(n_pts + 1):
                ang = math.radians(d + (spread_deg*2)*(t/n_pts))
                # 단순 구면 근사: 1 deg lat ≈ 111 km, 1 deg lon ≈ 111 km·cosφ
                dx = radius_m/1000 * math.cos(ang)
                dy = radius_m/1000 * math.sin(ang)
                lat = lat0 + dy/111
                lon = lon0 + dx/(111*math.cos(math.radians(lat0)))
                pts.append((lat, lon))
        pts.append((lat0, lon0))
        return pts

    # ----------------------------------------------------------
    def _create_map(self, planner: PathPlanner, zoom):
        # 지도 중심 : 첫 번째 Vertiport
        first = next(iter(planner.iport_names))
        lon0, lat0 = planner.nodes_geo[first]

        fmap = folium.Map(location=[lat0, lon0], zoom_start=zoom, tiles=None)

        # 기본 OSM 타일을 투명도 0.4 로
        folium.TileLayer(
            tiles="OpenStreetMap",
            name="Base",
            control=False,
            opacity=0.4          # ⇦ 0(완전 투명) ~ 1(불투명)
        ).add_to(fmap)

        # ① Vertiport : 마커 + INR/OTR/MTR 원 + 부채꼴
        for v in planner.iport_list:
            lon, lat = planner.nodes_geo[v["name"]]
            folium.CircleMarker([lat, lon], radius=6,
                                color="blue", fill=True,
                                tooltip=v["name"]).add_to(fmap)

            for key, col, ls in (("INR", "green", 2),
                                 ("OTR", "red",   2),
                                 ("MTR", "purple",1)):
                r_km = v.get(key, 0)
                if r_km <= 0: continue
                folium.Circle([lat, lon],
                              radius=r_km*1000,
                              color=col, weight=ls,
                              fill=False, opacity=0.5 if key!="MTR" else 0.3
                              ).add_to(fmap)

            # 부채꼴 ±10°
            for deg_key, col in (("INR_Deg","green"), ("OTR_Deg","red")):
                b = v.get(deg_key)
                if b is None: continue
                poly = self._sector(lon, lat, v["MTR"]*1000, b, 10)
                folium.Polygon(locations=poly,
                               color=None, fill=True,
                               fill_color=col, fill_opacity=0.25).add_to(fmap)

        # ② Waypoint
        for w in planner.waypoint_list:
            lon, lat = planner.nodes_geo[w["name"]]
            folium.CircleMarker([lat, lon], radius=4,
                                color="green", fill=True,
                                tooltip=w["name"]).add_to(fmap)

        # ③ 링크 : VP↔VP(파랑) , WP↔WP(빨강)
        for u, nbrs in planner.vp_graph.items():
            lat1, lon1 = planner.nodes_geo[u][1], planner.nodes_geo[u][0]
            for v, _ in nbrs:
                lat2, lon2 = planner.nodes_geo[v][1], planner.nodes_geo[v][0]
                folium.PolyLine([(lat1, lon1), (lat2, lon2)],
                                color="blue", weight=2, opacity=0.4).add_to(fmap)
        for u, nbrs in planner.wp_graph.items():
            lat1, lon1 = planner.nodes_geo[u][1], planner.nodes_geo[u][0]
            for v, _ in nbrs:
                lat2, lon2 = planner.nodes_geo[v][1], planner.nodes_geo[v][0]
                folium.PolyLine([(lat1, lon1), (lat2, lon2)],
                                color="red", weight=2, opacity=0.9).add_to(fmap)

        # ---- HTML 임시파일 저장 & 로드 ----
        tmp = tempfile.NamedTemporaryFile(suffix=".html", delete=False)
        fmap.save(tmp.name); tmp.close()
        self._tmp_path = tmp.name
        self.load(QUrl.fromLocalFile(str(Path(tmp.name).resolve())))

    # ----------------------------------------------------------
    def closeEvent(self, event):
        if self._tmp_path and os.path.exists(self._tmp_path):
            os.remove(self._tmp_path)
        super().closeEvent(event)

class CongestionOverlay(QObject):
    statsChanged = pyqtSignal(dict)  
    """링크(노드-노드) 단위 혼잡 텍스트 오버레이"""
    def __init__(self, planner, canvas):
        super().__init__()
        self.planner = planner
        self.ax      = canvas.figure.axes[0]
        self._texts  = {}    # link_key → matplotlib.text.Text

        # 모든 링크 꺼내서 (정렬된) 튜플 형태로 저장
        self._edges = set()
        for u, nbrs in planner.vp_graph.items():
            for v, _ in nbrs:
                self._edges.add(tuple(sorted((u, v))))
        for u, nbrs in planner.wp_graph.items():
            for v, _ in nbrs:
                self._edges.add(tuple(sorted((u, v))))
        self.stats = {}

    # ── congestion_tab.py ▸ CongestionOverlay.update  全교체 ─────────
    def update(self, vehicles: dict):
        """링크별 기체 수·평균 분리 간격을 계산해 텍스트·테이블 갱신"""
        # 0) 기존 텍스트 삭제
        for t in self._texts.values():
            t.remove()
        self._texts.clear()

        # ── 도움 함수 -------------------------------------------------
        def _proj_along_m(lon, lat, a, b):
            """점(lon,lat)을 링크 a→b에 투영해 링크 시작점으로부터 m 반환"""
            lon1, lat1 = a; lon2, lat2 = b
            # 경도→m 스케일은 위도 따라 달라서 중간위도 cos 사용
            km_per_deg = 111.32
            km_per_deg_lon = km_per_deg * math.cos(math.radians((lat1 + lat2) * 0.5))
            # 좌표를 평면(km)으로 변환
            ax, ay = (lon1 - lon1) * km_per_deg_lon, (lat1 - lat1) * km_per_deg
            bx, by = (lon2 - lon1) * km_per_deg_lon, (lat2 - lat1) * km_per_deg
            px, py = (lon  - lon1) * km_per_deg_lon, (lat  - lat1) * km_per_deg
            # 투영 비율 t (0~1)
            seg_len2 = bx * bx + by * by
            if seg_len2 == 0:
                return 0.0
            t = max(0.0, min(1.0, (px * bx + py * by) / seg_len2))
            # 투영점까지의 실제 거리(m)
            return math.hypot(bx * t, by * t) * 1000.0

        # 1) 링크별 기체 모음  {link:[(vid, along_m, heading), ...]}
        link_veh = {}
        for v in vehicles.values():
            node = v.get("curr_node")
            hd   = v.get("heading_deg")
            if node is None or hd is None:
                continue
            nbrs = self.planner.vp_graph.get(node, []) + self.planner.wp_graph.get(node, [])
            # heading과 가장 가까운 이웃 노드 선택
            best = None; best_d = 361
            for nb, _ in nbrs:
                lon1, lat1 = self.planner.nodes_geo[node]
                lon2, lat2 = self.planner.nodes_geo[nb]
                bearing = math.degrees(
                    math.atan2(
                        math.sin(math.radians(lon2 - lon1)) * math.cos(math.radians(lat2)),
                        math.cos(math.radians(lat1)) * math.sin(math.radians(lat2)) -
                        math.sin(math.radians(lat1)) * math.cos(math.radians(lat2)) *
                        math.cos(math.radians(lon2 - lon1))
                    )
                ) % 360
                diff = abs((bearing - hd + 180) % 360 - 180)
                if diff < best_d:
                    best_d, best = diff, nb
            if best:
                link = tuple(sorted((node, best)))
                lon, lat = v["lon"], v["lat"]
                a_pt = self.planner.nodes_geo[link[0]]
                b_pt = self.planner.nodes_geo[link[1]]
                along = _proj_along_m(lon, lat, a_pt, b_pt)
                link_veh.setdefault(link, []).append((v["id"], along, hd))

        # 2) 링크별 분리 간격 계산 & 텍스트 표시
        result = {}
        for link, lst in link_veh.items():
            #   (A,B) 정중앙 좌표
            lon_mid = (self.planner.nodes_geo[link[0]][0] + self.planner.nodes_geo[link[1]][0]) * 0.5
            lat_mid = (self.planner.nodes_geo[link[0]][1] + self.planner.nodes_geo[link[1]][1]) * 0.5

            # 진행 방향으로 두 그룹
            fwd  = [p for p in lst if p[2] < 180]
            back = [p for p in lst if p[2] >= 180]

            def _avg_gap(arr):
                """
                arr: [(vid, along_m, heading), …]  (along_m는 링크 기점부터 m)
                0 m(노드)에서 겹친 경우는 의미 없으므로 gap = 0 은 제외.
                """
                if len(arr) < 2:
                    return None
                arr.sort(key=lambda x: x[1])                    # along_m 순
                gaps = [arr[i+1][1] - arr[i][1]
                        for i in range(len(arr)-1)
                        if arr[i+1][1] - arr[i][1] > 1.0]       # ★ 1 m 초과만 사용
                return None if not gaps else sum(gaps) / len(gaps)

            g1 = _avg_gap(fwd)
            g2 = _avg_gap(back)
            gaps = [g for g in (g1, g2) if g is not None]
            avg_m = sum(gaps)/len(gaps) if gaps else None
            result[link] = (len(lst), avg_m)

            txt = f"{link[0]}↔{link[1]}\nN={len(lst)}"
            if avg_m is not None:
                txt += f" / {avg_m:,.0f} m"
            self._texts[link] = self.ax.text(
                lon_mid, lat_mid, txt, color="red",
                ha="center", va="center", fontsize=7, weight="bold", zorder=10
            )

        self.ax.figure.canvas.draw_idle()
        self.stats = result
        self.statsChanged.emit(result)



# congestion_tab.py  ───────────────────────────────────────────
from pathlib import Path
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QStackedWidget, QLabel,
    QPushButton, QSlider, QFrame
)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# 이미 프로젝트에 있는 헬퍼 모듈
#   from .path_planner       import PathPlanner
#   from .path_visualizer_geo import PathVisualizerGeo
#   from .map_view           import MapView
#   from .congestion_overlay import CongestionOverlay   # 앞서 만든 클래스
# -------------------------------------------------------------

class CongestionTab(QWidget):
    """
    Node-Link 네트워크 + 실시간 혼잡 오버레이 탭
    외부에서 PathPlanner 인스턴스를 주입해 사용한다.
    """
    def __init__(self, planner, parent=None):
        super().__init__(parent)
        self._planner = planner        # (중요) 외부에서 넘겨준 planner
        self._link_lines = {}   # (A,B) → matplotlib Line2D
        self._sel_link   = None  # 선택된 링크 (A,B) 튜플
        self._init_ui()
        self._overlay.statsChanged.connect(self._on_stats)

        self._max_hist = {}   # link → (최대 N, gap)
        self._min_hist = {}   # link → (최소 N>0, gap)


    # ------------------------------------------------------------------
    def _init_ui(self):
        # ── ① 시간 슬라이더 -------------------------------------------
        slider = QSlider(Qt.Horizontal, self)
        slider.setRange(0, 24 * 60 - 1)           # 00:00 ~ 23:59 (분 단위)
        slider.setPageStep(60)                    # ⇱/⇲ 한 시간씩
        h_sld = QHBoxLayout()
        h_sld.addWidget(QLabel("00:00"))
        h_sld.addWidget(slider)
        h_sld.addWidget(QLabel("23:59"))
        f_sld = QFrame(self)
        f_sld.setLayout(h_sld)
        f_sld.setFrameShape(QFrame.Panel)
        f_sld.setFrameShadow(QFrame.Raised)

        # ── ② 스택: (0) 지도+노드 ⟷ (1) 순수 노드-링크 ----------------
        stack = QStackedWidget(self)
        stack.addWidget(MapView(self._planner))                 # index 0
        canvas = FigureCanvas(PathVisualizerGeo(self._planner).fig)
        self.ax = canvas.figure.axes[0]
        stack.addWidget(canvas)                                 # index 1

        # ── ③ 혼잡 오버레이 (노드 위 텍스트) ---------------------------
        self._overlay = CongestionOverlay(self._planner, canvas)

        # ── ④ 우측: Link Table + Log -----------------------------------------
        self.table = self._make_table()
        right_up   = self.table                     # 위쪽: 테이블
        self.log   = QLabel("—", self)              # 아래쪽: 실시간 요약 로그
        self.log.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.log.setFrameShape(QFrame.Panel)
        self.log.setWordWrap(True)
        self.log.setMinimumHeight(120)

        v_right = QVBoxLayout()
        v_right.addWidget(right_up, 2)
        v_right.addWidget(self.log,   1)            # ← 3-D 자리 대신 로그

        # ── ⑤ 전환 버튼 -----------------------------------------------
        btn_map  = QPushButton("With Map", self)
        btn_node = QPushButton("Only Node-Link", self)
        btn_map.clicked.connect(lambda: stack.setCurrentIndex(0))
        btn_node.clicked.connect(lambda: stack.setCurrentIndex(1))
        h_btn = QHBoxLayout()
        h_btn.addWidget(btn_map)
        h_btn.addWidget(btn_node)

        # ── ⑥ 메인 레이아웃 -------------------------------------------
        h_main = QHBoxLayout()
        h_main.addWidget(stack, 3)
        h_main.addLayout(v_right, 2)

        root = QVBoxLayout(self)
        root.addWidget(f_sld)      # 위: 시간 슬라이더
        root.addLayout(h_main, 1)  # 중간: 지도/노드 + 우측 정보
        root.addLayout(h_btn)      # 아래: 전환 버튼

        # 필요하면 외부에서 슬라이더에 접근하도록 보관
        self.time_slider = slider


    # ------------------------------------------------------------------
    # MonitoringTab.trafficUpdated 신호를 연결하기 위한 헬퍼
    def bind_traffic_source(self, traffic_signal):
        """외부 MonitoringTab 의 trafficUpdated → 혼잡 오버레이 연결"""
        traffic_signal.connect(self._overlay.update)

    def _make_table(self):
        """시작할 때 모든 링크를 고정 행으로 생성한다."""
        # ① 시뮬 네트워크 모든 링크 집합
        edges = set()
        for u, nbrs in self._planner.vp_graph.items():
            for v, _ in nbrs:
                edges.add(tuple(sorted((u, v))))
        for u, nbrs in self._planner.wp_graph.items():
            for v, _ in nbrs:
                edges.add(tuple(sorted((u, v))))

        self._all_links = sorted(edges)          # 유지용 리스트

        # ② 테이블 생성 & 행 고정
        tbl = QTableWidget(len(self._all_links), 3, self)
        tbl.setHorizontalHeaderLabels(["Link", "N", "Avg Gap (m)"])
        tbl.horizontalHeader().setStretchLastSection(True)
        tbl.setEditTriggers(QTableWidget.NoEditTriggers)
        tbl.cellClicked.connect(self._on_table_clicked)

        # ③ 링크 이름만 미리 채워둔다
        for r, link in enumerate(self._all_links):
            a, b = link
            tbl.setItem(r, 0, QTableWidgetItem(f"{a}↔{b}"))
            tbl.setItem(r, 1, QTableWidgetItem("0"))
            tbl.setItem(r, 2, QTableWidgetItem("-"))
            tbl.item(r, 0).setData(Qt.UserRole, link)   # 링크 key 저장

        return tbl
        
    # congestion_tab.py ▸ CongestionTab._on_stats  전체 교체
    def _on_stats(self, stats: dict):
        # 1) 테이블 값 실시간 갱신  (행 고정)
        for r, link in enumerate(self._all_links):
            cnt, gap = stats.get(link, (0, None))
            self.table.item(r, 1).setText(str(cnt))
            self.table.item(r, 2).setText("-" if gap is None else f"{gap:.0f}")

        # 2) 누적 최고 혼잡 기록 갱신
        for link, (cnt, gap) in stats.items():
            prev = self._max_hist.get(link)
            if prev is None or cnt > prev[0]:
                self._max_hist[link] = (cnt, gap)

        if not self._max_hist:
            self.log.setText("—")
            return

        # 3) TOP 5 추출 (N 내림차순)
        top5 = sorted(self._max_hist.items(),
                    key=lambda x: x[1][0], reverse=True)[:5]

        def _fmt(item):
            (a, b), (n, g) = item
            gtxt = "-" if g is None else f"{g:.0f} m"
            return f"{a}-{b} : N={n}, Gap={gtxt}"

        self.log.setText(
            "▲ 누적 혼잡 TOP 5\n" +
            "\n".join(_fmt(i) for i in top5)
        )



    def _on_table_clicked(self, row, col):
        link = self.table.item(row,0).data(Qt.UserRole)   # (A,B)
        if not link: return

        # ① 이전 강조 복구
        if self._sel_link and self._sel_link in self._link_lines:
            ln = self._link_lines[self._sel_link]
            ln.set_color("black"); ln.set_linewidth(1)

        # ② 새 강조
        a,b = link
        ln = self._link_lines.get(link)
        if not ln:
            # 최초 호출이면 PathVisualizerGeo 의 Line2D 찾아서 캐시
            for l in self._planner.vp_graph.get(a,[])+self._planner.wp_graph.get(a,[]):
                if l[0]==b or l[0]==a:          # 인접선 찾기
                    # PathVisualizerGeo 가 생성한 Line2D 배열은 canvas.axes[0].lines
                    for obj in self.ax.lines:
                        if tuple(sorted(obj.get_label().split("↔"))) == link:
                            self._link_lines[link]=obj; ln=obj; break
        if ln:
            ln.set_color("yellow"); ln.set_linewidth(3)
            self.ax.figure.canvas.draw_idle()
        self._sel_link = link