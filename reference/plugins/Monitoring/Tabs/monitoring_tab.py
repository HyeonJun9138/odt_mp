"""
monitoring_tab.py  ―  UAM Traffic GUI ▸ Monitoring 탭
"""

from __future__ import annotations

# ── 표준 라이브러리 ─────────────────────────────────────────────
import math, os, json, socket, tempfile, shutil, copy, time
from pathlib import Path
from typing import Dict, List
from datetime import datetime, timedelta
# ── 외부 라이브러리 ────────────────────────────────────────────
import folium
import pandas as pd
import re   # 파일 머리부분 import 위치에 이미 있을 수도 있음
from PyQt5.QtCore    import Qt, QUrl, QTimer, QObject, pyqtSignal, pyqtSlot
from PyQt5.QtGui     import QPixmap
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QGridLayout,
    QLabel, QFrame, QProgressBar
)
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtWebChannel        import QWebChannel
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# ── 프로젝트 로컬 모듈 ─────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR  = ROOT_DIR / "Monitoring" / "Sources"

from Monitoring.Functions.PathPlanning   import PathPlanner
from Monitoring.Functions                import UAM_Path2Sim as p2s
from Monitoring.Functions.AircraftAgent  import AircraftAgent
from Monitoring.Functions.MissionProfile import MissionSegment

# ══════════════════════════════════════════════════════════════
# 0.  좌표/시간 유틸
# ══════════════════════════════════════════════════════════════
LON0, LAT0 = 126.978291, 37.566669   # Folium 기본 센터(서울 시청 인근)
KM_PER_DEG = 111.32                  # 위도 1°

# ── 상단 import 근처에 추가 ─────────────────────────────
from datetime import datetime
import re

def _latest_fpl_folder(*roots: Path, prefix: str="FPL_") -> Path|None:
    """
    주어진 루트들에서 FPL_YYYYMMDD 형식 폴더 중 가장 최신 경로 반환
    """
    cand = []
    pat  = re.compile(rf"^{re.escape(prefix)}(\d{{8}})$")
    for root in roots:
        if not root.exists():       # 안전 장치
            continue
        for p in root.iterdir():
            m = pat.match(p.name)
            if m and p.is_dir():
                try:
                    cand.append((datetime.strptime(m[1], "%Y%m%d"), p))
                except ValueError:
                    pass
    return max(cand, default=(None, None))[1] if cand else None


def km_per_lon(lat_deg: float) -> float:
    return KM_PER_DEG * math.cos(math.radians(lat_deg))

def lonlat_to_xy(lon: float, lat: float) -> tuple[float, float]:
    dx = (lon - LON0) * km_per_lon(LAT0) * 1000
    dy = (lat - LAT0) * KM_PER_DEG       * 1000
    return dx, dy

def xy_to_lonlat(x_m: float, y_m: float) -> tuple[float, float]:
    d_lat = (y_m / 1000) / KM_PER_DEG
    d_lon = (x_m / 1000) / km_per_lon(LAT0)
    return LON0 + d_lon, LAT0 + d_lat

def time_to_sec(hms: str) -> int:
    h, m, *s = map(int, hms.split(":"))
    return h*3600 + m*60 + (s[0] if s else 0)

def sec_to_str(sec: int) -> str:
    sec %= 86_400
    return f"{sec//3600:02}:{(sec%3600)//60:02}:{sec%60:02}"


# ══════════════════════════════════════════════════════════════
# 1.  DetailPanel
# ══════════════════════════════════════════════════════════════
def _mk(text: str, css: str | None = None) -> QLabel:
    lbl = QLabel(text)
    if css:
        lbl.setProperty("class", css)
    return lbl


class DetailPanel(QWidget):
    """오른쪽 상단: 선택 기체 세부 정보"""

    def __init__(self, parent=None):
        super().__init__(parent)

        # ── 스타일
        self.setStyleSheet("""
            QWidget#Header      { background:#003366; }
            QLabel.hd           { font:900 28px 'Segoe UI'; color:#ffffff; }

            QLabel.bigkey       { font:700 15px 'Segoe UI'; }
            QLabel.bigval       { font:600 15px 'Segoe UI'; }

            QLabel.midkey       { font:700 14px 'Segoe UI'; }
            QLabel.rowkey       { font:600 13px 'Segoe UI'; color:#707070; }
            QLabel.rowval       { font:600 13px 'Segoe UI'; }

            QLabel.small        { font:500 12px 'Segoe UI'; }
            QProgressBar        { border:1px solid #aaa; height:14px; }
            QProgressBar::chunk { background:#ffb400; }
        """)

        root = QVBoxLayout(self); root.setContentsMargins(2,2,2,2); root.setSpacing(6)

        # ── (1) Header
        hd = QFrame(objectName="Header")
        hb = QHBoxLayout(hd); hb.setContentsMargins(10,4,10,4)
        self.lbl_dep_name = _mk("출발지","hd")
        self.lbl_arr_name = _mk("도착지","hd")
        plane = QLabel("\u2708"); plane.setStyleSheet("font:900 28px 'Segoe UI'; color:#ffe000;")
        hb.addWidget(self.lbl_dep_name); hb.addStretch(); hb.addWidget(plane); hb.addStretch(); hb.addWidget(self.lbl_arr_name)
        root.addWidget(hd)

        # ── (2) ID + 스케줄 테이블
        self.lbl_id = _mk("ID: --", "midkey")
        tm_lbls = {k: _mk("--", "bigval") for k in ("std","etd","sta","eta")}
        self.tm = tm_lbls

        grid = QGridLayout(); grid.setHorizontalSpacing(4); grid.setVerticalSpacing(3)
        grid.addWidget(_mk("SCHEDULED","bigkey"), 0,1, Qt.AlignCenter)
        grid.addWidget(_mk("ACTUAL",   "bigkey"), 0,2, Qt.AlignCenter)
        grid.addWidget(_mk("DEP","midkey"), 1,0); grid.addWidget(tm_lbls["std"], 1,1); grid.addWidget(tm_lbls["etd"], 1,2)
        grid.addWidget(_mk("ARR","midkey"), 2,0); grid.addWidget(tm_lbls["sta"], 2,1); grid.addWidget(tm_lbls["eta"], 2,2)

        box1 = QVBoxLayout(); box1.setContentsMargins(0,0,0,0); box1.setSpacing(4)
        box1.addWidget(self.lbl_id); box1.addLayout(grid)
        w1 = QWidget(); w1.setLayout(box1)          # ← 레이아웃을 담을 위젯
        root.addWidget(self._boxed(w1))

        # ── (3) Phase / 좌표 / 승객
        keys = ("Phase","heading","Lon","Lat","Alt (m)","Remain km","Pax")
        self.val = {}
        info_g = QGridLayout(); info_g.setHorizontalSpacing(2); info_g.setVerticalSpacing(1)
        for r,k in enumerate(keys):
            info_g.addWidget(_mk(k,"rowkey"), r,0)
            self.val[k] = _mk("-", "rowval")
            info_g.addWidget(self.val[k], r,1)
        w2 = QWidget(); w2.setLayout(info_g)
        root.addWidget(self._boxed(w2))

        # ── (4) Progress bar
        self.pb = QProgressBar(maximum=100); self.pb.setTextVisible(False)
        self.lbl_tr = _mk("0 km, – ago","small")
        self.lbl_rm = _mk("0 km, in –","small")
        vb = QVBoxLayout(); vb.addWidget(self.pb)
        hb2 = QHBoxLayout(); hb2.addWidget(self.lbl_tr); hb2.addStretch(); hb2.addWidget(self.lbl_rm)
        vb.addLayout(hb2)
        w3 = QWidget(); w3.setLayout(vb)
        root.addWidget(self._boxed(w3, radius=4, padding=4))
        root.addStretch()
        self.current_id = None   # 선택된 기체 ID

    def _boxed(self, widget: QWidget, *, radius: int = 6,
        padding: int = 6) -> QFrame:
        frame = QFrame()
        frame.setStyleSheet(f"""
        QFrame {{
            border:1px solid #bbb;
            border-radius:{radius}px;
            background:#ffffff;
                }}
            """)
        lay = QVBoxLayout(frame)
        lay.setContentsMargins(padding, padding, padding, padding)
        lay.addWidget(widget)
        return frame
    
    # ── helper: 하얀 Box
    def update(self, v: Dict):
        self.current_id = v.get("_key", v.get("id", None))
        self.lbl_id.setText(f"ID : {v['id']}")
        self.lbl_dep_name.setText(v["departure"])
        self.lbl_arr_name.setText(v["destination"])

        for k in ("std","etd","sta","eta"):
            self.tm[k].setText(v.get(k, "–"))

        self.val["Phase"].setText(v["phase"])
        self.val["heading"].setText(f"{v.get('heading_deg', 0):.1f}°")
        self.val["Lon"].setText(f"{v['lon']:.5f}")
        self.val["Lat"].setText(f"{v['lat']:.5f}")
        self.val["Alt (m)"].setText(f"{v['z']:.0f}")
        self.val["Remain km"].setText(f"{v['remain']:.1f}")
        self.val["Pax"].setText(str(v["pax"]))

        pct = int(v["traveled"] / v["total_km"] * 100) if v["total_km"] else 0
        self.pb.setValue(max(0, min(pct, 100)))        # 0 – 100 안쪽 클램프

        # ── 경과 / 남은 시간 ────────────────────────────────────────
        def _fmt_dur(sec: int) -> str:
            """0 → '0s', 67 → '1m07s', 3920 → '1h05m' 식 포맷"""
            if sec < 60:
                return f"{sec:d}s"
            if sec < 3600:
                return f"{sec//60:d}m{sec%60:02d}s"
            return f"{sec//3600:d}h{(sec%3600)//60:02d}m"

        ago_txt = "–"
        rm_txt  = "–"
        now = v.get("_now_sec")
        if now is not None:
            if v.get("etd_sec") is not None:
                ago_txt = _fmt_dur(max(0, now - v["etd_sec"]))
            if v.get("eta_sec") is not None:
                rm_txt  = _fmt_dur(max(0, v["eta_sec"] - now))

        self.lbl_tr.setText(f"{v['traveled']:.0f} km, {ago_txt} ago")
        self.lbl_rm.setText(f"{v['remain']:.0f} km, in {rm_txt}")



# ══════════════════════════════════════════════════════════════
# 2.  MapView  (folium + Leaflet)
# ══════════════════════════════════════════════════════════════
class MapBridge(QObject):
    clicked = pyqtSignal(str)
    @pyqtSlot(str)
    def emit(self, vid): self.clicked.emit(vid)

_STALE_SEC  = 5          # 60 초 이상 패킷이 없으면 ‘끊김’으로 간주
_ARRIVED_SEC = 10         # ARRIVED 후 30 초 보여주고 제거

class MapView(QWebEngineView):
    """좌측: 실시간 지상/공역 지도"""


    def __init__(self, planner: PathPlanner,
                 center: tuple[float,float] | None = None,
                 zoom: int = 11, parent=None):
        super().__init__(parent)
        self.p           = planner
        self._loaded     = False
        self._pending_js: List[str] = []
        self._callbacks  = {}
        

        # 센터 지정 없으면 VP 평균
        if center is None:
            lons,lats = zip(*(planner.nodes_geo[n] for n in planner.iport_names))
            center = (sum(lats)/len(lats), sum(lons)/len(lons))
        self._build_map(center, zoom)
        self.loadFinished.connect(self._on_load)

        # 시계 오버레이 라벨
        from PyQt5.QtWidgets import QLabel
        self._clock = QLabel("00:00:00", self)
        self._clock.setStyleSheet("""
            QLabel { background:rgba(0,0,0,0.55); color:#fff;
                     font:600 14px 'Segoe UI'; padding:3px 8px; border-radius:4px; }""")
        self._clock.adjustSize(); self._place_clock()
        self.resizeEvent = self._wrap_resize_evt(self.resizeEvent)

    def remove_all(self, vid: str):
        cmd = f"removeAll('{vid}');"
        if self._loaded:
            self.page().runJavaScript(cmd)
        else:
            self._pending_js.append(cmd)

    def _sector(self, lon0: float, lat0: float,
                radius_m: float,
                bearing_deg: float,
                half_angle_deg: float,
                n_pts: int = 30) -> list[tuple[float, float]]:
        """
        반환 좌표 (lat, lon)  – Folium 순서에 맞춤
        """
        import math
        R = 6_371_000.0                     # 지구 반경 (m)
        lat0_rad = math.radians(lat0)
        lon0_rad = math.radians(lon0)
        brg_rad  = math.radians(bearing_deg)

        start = brg_rad - math.radians(half_angle_deg)
        end   = brg_rad + math.radians(half_angle_deg)
        step  = (end - start) / n_pts

        poly: list[tuple[float, float]] = [(lat0, lon0)]
        for i in range(n_pts + 1):
            θ = start + i * step
            lat = math.asin(
                math.sin(lat0_rad) * math.cos(radius_m / R) +
                math.cos(lat0_rad) * math.sin(radius_m / R) * math.cos(θ)
            )
            lon = lon0_rad + math.atan2(
                math.sin(θ) * math.sin(radius_m / R) * math.cos(lat0_rad),
                math.cos(radius_m / R) - math.sin(lat0_rad) * math.sin(lat)
            )
            poly.append((math.degrees(lat), math.degrees(lon)))

        return poly
    
    # ── 시계 관련
    def _place_clock(self): m=10; self._clock.move(self.width()-self._clock.width()-m, m)
    def _wrap_resize_evt(self, prev_fn):
        def _res(ev): prev_fn(ev); self._place_clock()
        return _res
    def update_clock(self, hms:str): self._clock.setText(hms); self._clock.adjustSize(); self._place_clock()

    # ── 지도·마커 JS
    def _build_map(self, center: tuple[float,float], zoom: int):
        lat0,lon0 = center
        fmap = folium.Map(location=[lat0,lon0],
                          zoom_start=zoom,
                          tiles="CartoDB Dark_Matter")
        folium.TileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
                         attr="© OpenStreetMap", name="OSM", opacity=0.65).add_to(fmap)
        map_js = fmap.get_name()
        self._map_js = map_js

        # Vertiport / Waypoint / 링크 … (PathPlanner 정보 활용) ──────────
        # ── Vertiport 마커 + 반경·부채꼴 ──────────────────────────────
        for vp in self.p.iport_list:
            lon, lat = self.p.nodes_geo[vp["name"]]

            # VP 원점 마커
            icon_html = f"""
            <div style="
                display: flex;
                align-items: center;
                justify-content: center;
                width: 18px;            /* 아이콘의 너비 */
                height: 18px;           /* 아이콘의 높이 */
                font-size: 12px;        /* 'V' 글자 크기 */
                font-weight: bold;      /* 글자 굵기 */
                color: Orange;           /* 'V' 글자 색상 */
                background-color: #003366; /* 배경색 (원하시는 색으로 변경 가능) */
                border: 1.5px solid Orange; /* 테두리 색상 및 두께 */
                border-radius: 50%;     /* 원형 모양 만들기 */
                box-shadow: 1px 1px 3px rgba(0,0,0,0.4); /* 간단한 그림자 효과 (선택 사항) */
                ">
                V  </div>
            """
            
            # HTML 내용을 기반으로 DivIcon 생성
            vertiport_icon = folium.DivIcon(
                icon_size=(24, 24),     
                icon_anchor=(12, 12),   
                html=icon_html
            )
            
            # 생성된 DivIcon을 사용하여 마커 추가
            folium.Marker(
                location=[lat, lon],    
                icon=vertiport_icon,    
                tooltip=vp["name"]      
            ).add_to(fmap)

            # INR / OTR / MTR 동심원
            for key, col, width in (("INR", "green", 2),
                                    ("OTR", "red",   2),
                                    ("MTR", "purple", 1)):
                r_km = vp.get(key, 0)
                if r_km > 0:
                    folium.Circle([lat, lon],
                                radius=r_km * 1000,
                                color=col, weight=width,
                                fill=False,
                                opacity=0.5 if key != "MTR" else 0.3
                                ).add_to(fmap)

            # ±10° 부채꼴 (bearing 값 있으면)
            for deg_key, col in (("INR_Deg", "green"), ("OTR_Deg", "red")):
                bearing = vp.get(deg_key)
                if bearing is None or vp.get("MTR", 0) <= 0:
                    continue
                poly = self._sector(lon, lat, vp["MTR"] * 1000,
                                    bearing_deg=bearing, half_angle_deg=10)
                folium.Polygon(locations=poly,
                            color=None, fill=True,
                            fill_color=col, fill_opacity=0.25
                            ).add_to(fmap)


        for wp in self.p.waypoint_list:
            lon,lat = self.p.nodes_geo[wp["name"]]
            folium.CircleMarker([lat,lon], radius=4, color="green", fill=True,
                                tooltip=wp["name"]).add_to(fmap)

        for u,nbrs in self.p.wp_graph.items():
            lat1,lon1 = self.p.nodes_geo[u][1], self.p.nodes_geo[u][0]
            for v,_ in nbrs:
                lat2,lon2 = self.p.nodes_geo[v][1], self.p.nodes_geo[v][0]
                folium.PolyLine([(lat1,lon1),(lat2,lon2)],
                                color="#ff0033", weight=1.8, opacity=.9).add_to(fmap)

        # plane 아이콘 복사( tmp 폴더 ), JS 함수 정의
        plane_candidates = ["plane.svg"] + [f"plane{i}.svg" for i in range(1, 5)]
        plane_files: list[str] = []
        tmp_dir = Path(tempfile.gettempdir())
        for fn in plane_candidates:
            src = (SRC_DIR / fn)
            if not src.exists():
                continue
            try:
                shutil.copy(src, tmp_dir / fn)
                plane_files.append(fn)
            except Exception:
                pass
        if not plane_files:
            fallback = SRC_DIR / "plane.svg"
            if fallback.exists():
                try:
                    shutil.copy(fallback, tmp_dir / fallback.name)
                except Exception:
                    pass
                plane_files = [fallback.name]
            else:
                plane_files = ["plane.svg"]
        plane_files_js = ",".join([f"'{f}'" for f in plane_files])

        js = f"""
    // ───────────────────────────────
    //   Leaflet helper functions
    // ───────────────────────────────

    // 전역 캐시
    window.vMarkers      = {{}};   // 비행체 마커
    window.routeLayers   = {{}};   // 남은 경로(Polyline)
    window.pendingRoutes = {{}};   // Python → JS 전달용 경로 버퍼

    const planeFiles = [{plane_files_js}];
    const iconCache  = {{}};       // DivIcon 캐싱

    // ── 아이콘 로더
    function getIcon(fn) {{
        if (!iconCache[fn]) {{
            iconCache[fn] = L.icon({{
                iconUrl:        fn,
                iconSize:       [30, 30],
                iconAnchor:     [15, 15],
                tooltipAnchor:  [0, 0]
            }});
        }}
        return iconCache[fn];
    }}

    // ── 마커 추가 / 갱신
    function addOrMove(id, lat, lon, info, heading) {{
        if (window.vMarkers[id]) {{
            // 위치·툴팁·방향만 업데이트
            window.vMarkers[id].setLatLng([lat, lon]);
            const tip = window.vMarkers[id].getTooltip();
            if (tip) tip.setContent(info);
            window.vMarkers[id].setRotationAngle(heading);
        }} else {{
            // 새 마커 생성
            const fn = planeFiles[Math.floor(Math.random() * planeFiles.length)];
            window.vMarkers[id] = L.marker([lat, lon], {{
                icon: getIcon(fn),
                rotationAngle:  heading,
                rotationOrigin: 'center center'
            }})
            .addTo({map_js})
            .on('click', () => {{
                // Python 슬롯 호출
                channel.emit(id);

                // 기존 경로 전부 삭제
                for (let k in window.routeLayers) {{
                    {map_js}.removeLayer(window.routeLayers[k]);
                    delete window.routeLayers[k];
                }}

                // 선택된 기체의 경로만 다시 그림
                if (window.pendingRoutes[id])
                    drawRoute(id, window.pendingRoutes[id]);
            }})
            .bindTooltip(info, {{direction:'top', offset:[0,-8]}});
        }}
    }}

    // ── 남은 경로(Polyline) 그리기
    function drawRoute(id, coords) {{
        // 기존 레이어 제거
        if (window.routeLayers[id]) {{
            {map_js}.removeLayer(window.routeLayers[id]);
            delete window.routeLayers[id];
        }}
        // 좌표가 있을 때만 새로 그림
        if (coords && coords.length > 0) {{
            window.routeLayers[id] = L.polyline(coords, {{
                color:   'cyan',
                weight:  4,
                opacity: 0.7
            }}).addTo({map_js});
        }}
    }}

    // ── 마커 · 경로 · 부채꼴 · heading line 삭제
    function removeAll(id) {{
        if (window.vMarkers[id]) {{
            {map_js}.removeLayer(window.vMarkers[id]);
            delete window.vMarkers[id];
        }}
        if (window.routeLayers[id]) {{
            {map_js}.removeLayer(window.routeLayers[id]);
            delete window.routeLayers[id];
        }}
        if (window['sector_' + id]) {{
            {map_js}.removeLayer(window['sector_' + id]);
            delete window['sector_' + id];
        }}
        if (window['direct_heading_line_' + id]) {{
            {map_js}.removeLayer(window['direct_heading_line_' + id]);
            delete window['direct_heading_line_' + id];
        }}
    }}
    """

        fmap.get_root().script.add_child(folium.Element(js))
        fmap.get_root().html.add_child(folium.Element(
            '<script src="https://unpkg.com/leaflet-rotatedmarker/leaflet.rotatedMarker.js"></script>'
            '<script src="https://cdn.jsdelivr.net/npm/qwebchannel@latest/qwebchannel.js"></script>'
            '<script>new QWebChannel(qt.webChannelTransport,function(ch){window.channel=ch.objects.channel;});</script>'
        ))


        tmp = tempfile.NamedTemporaryFile(suffix=".html", delete=False)
        fmap.save(tmp.name); tmp.close(); self._tmp = tmp.name
        self.load(QUrl.fromLocalFile(str(Path(tmp.name).resolve())))
        self.channel = QWebChannel(self.page())
        self.bridge  = MapBridge()
        self.channel.registerObject('channel', self.bridge)
        self.page().setWebChannel(self.channel)
        self.bridge.clicked.connect(self._bridge_clicked)

    def _on_load(self, ok): self._loaded=ok; [self.page().runJavaScript(js) for js in self._pending_js]
    def _bridge_clicked(self, vid): self._callbacks.get(vid, lambda:None)()
    def add_or_move(self, vid, lat, lon, info,heading, cb):
        self._callbacks[vid] = cb
        cmd = (
            f"addOrMove('{vid}', {lat}, {lon}, {info!r}, "
            f"{heading:.1f});"
        )
        if self._loaded:
            self.page().runJavaScript(cmd)
        else:
            self._pending_js.append(cmd)
    def closeEvent(self, e):
        if hasattr(self,"_tmp") and os.path.exists(self._tmp): os.remove(self._tmp)
        super().closeEvent(e)


# ══════════════════════════════════════════════════════════════
# 3.  3-D 궤적 패널
# ══════════════════════════════════════════════════════════════
class Traj3DPanel(FigureCanvas):
    """선택 기체의 3-D 궤적 (CSV Seg 열 그대로)"""

    def __init__(self, parent=None):
        fig = Figure(figsize=(4, 3))
        super().__init__(fig)
        self.ax = fig.add_subplot(111, projection="3d")

        self._init_elev, self._init_azim = 5, -50  
        self._last_traj: list[dict] | None = None # 그려진 전체 경로 데이터 저장
        self._curr_scatter = None                 # 현재 위치 점(scatter) 객체 저장
        self._current_vid_on_3d: str | None = None # 현재 3D 패널에 표시된 항공기 ID

        from PyQt5.QtWidgets import QPushButton
        self._btn = QPushButton("↺", self)
        self._btn.setFixedSize(24, 24)
        self._btn.clicked.connect(self._reset_view)
        self._clear_placeholder()

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self._btn.move(6, 6)

    def _clear_placeholder(self):
        self.ax.cla()
        self.ax.text2D(0.5, 0.5, "3-D Trajectory\n(select)",
                       ha="center", va="center")
        self.ax.set_axis_off()
        self._last_traj = None # 궤적 정보 없음으로 초기화
        self._current_vid_on_3d = None # 표시된 항공기 ID 없음
        if self._curr_scatter:
                # ArtistList에서 인덱스를 찾아 직접 삭제
                try:
                    idx = self.ax.collections.index(self._curr_scatter)
                    del self.ax.collections[idx]
                except ValueError:
                    pass
                self._curr_scatter = None
        self.draw_idle()

    def _reset_view(self):
        """카메라 시점만 초기 위치로 되돌립니다."""
        if self._last_traj:
            # 화면은 지우지 않고, 궤적은 그대로 두고 시점만 리셋
            self.ax.view_init(self._init_elev, self._init_azim)
            self.draw_idle()
        else:
            # 궤적이 없으면 placeholder 유지
            self._clear_placeholder()

    def show_traj(self, traj_points_list: list[dict] | None, current_traj_idx: int | None = None, aircraft_id: str | None = None):
        # 현재 표시하려는 항공기가 다르거나, 아직 아무것도 표시되지 않은 경우 (또는 강제 전체 다시 그리기)
        force_full_redraw = (aircraft_id != self._current_vid_on_3d) or (self._last_traj is None)

        if traj_points_list is None or not traj_points_list: # 표시할 경로가 없는 경우
            self._clear_placeholder()
            return

        # 새 항공기를 표시하거나, 경로 데이터 자체가 변경된 경우 전체 다시 그리기
        if force_full_redraw or self._last_traj != traj_points_list:
            self._current_vid_on_3d = aircraft_id
            self._last_traj = copy.deepcopy(traj_points_list) # 새 전체 경로 저장 (깊은 복사)
            
            self.ax.cla() # 이전 그림 모두 지우기
            self.ax.set_axis_on()

            # ① 전체 경로 그리기
            x0_path, y0_path = self._last_traj[0]["x"], self._last_traj[0]["y"]
            xs_path = [p["x"] - x0_path for p in self._last_traj]
            ys_path = [p["y"] - y0_path for p in self._last_traj]
            zs_path = [p["z"] for p in self._last_traj]

            self.ax.plot(xs_path, ys_path, zs_path, "-", color="magenta", label="Full Path")
            self.ax.scatter(xs_path[0], ys_path[0], zs_path[0], color="green", s=50, label="Path Start")
            if len(xs_path) > 1:
                self.ax.scatter(xs_path[-1], ys_path[-1], zs_path[-1], color="red", s=50, label="Path End")

            # 축 설정 (레이블, 범위, 시점)
            self.ax.set_xlabel("ΔX (m)")
            self.ax.set_ylabel("ΔY (m)")
            self.ax.set_zlabel("Altitude (m)")

            min_x_p, max_x_p = min(xs_path), max(xs_path)
            min_y_p, max_y_p = min(ys_path), max(ys_path)
            dx_p = max_x_p - min_x_p
            dy_p = max_y_p - min_y_p
            cx_p = (max_x_p + min_x_p) / 2
            cy_p = (max_y_p + min_y_p) / 2
            rng_p = max(dx_p, dy_p)
            eps = 10.0 # 최소 범위 (궤적이 점 하나일 경우 등)
            
            plot_display_range = max(rng_p, eps)

            self.ax.set_xlim(cx_p - plot_display_range / 2, cx_p + plot_display_range / 2)
            self.ax.set_ylim(cy_p - plot_display_range / 2, cy_p + plot_display_range / 2)
            self.ax.set_zlim(0, 500) # Z축 고정 또는 데이터 기반으로 조정 가능
            self.ax.view_init(self._init_elev, self._init_azim)
            
            # 이전에 있던 현재 위치 점 삭제 (새로운 전체 경로를 그리므로)
            if self._curr_scatter:
                # ArtistList에서 인덱스를 찾아 직접 삭제
                try:
                    idx = self.ax.collections.index(self._curr_scatter)
                    del self.ax.collections[idx]
                except ValueError:
                    pass
                self._curr_scatter = None
            
            # 새 전체 경로에 대한 현재 위치 점 (재)그리기 (current_traj_idx가 있다면)
            if current_traj_idx is not None and (0 <= current_traj_idx < len(self._last_traj)):
                current_point_data = self._last_traj[current_traj_idx]
                cur_x_rel = current_point_data["x"] - x0_path
                cur_y_rel = current_point_data["y"] - y0_path
                cur_z_abs = current_point_data["z"]
                self._curr_scatter = self.ax.scatter(
                    cur_x_rel, cur_y_rel, cur_z_abs,
                    color="yellow", marker='o', s=100, label="Current Position", depthshade=True, edgecolors='black'
                )
            
            self.ax.legend(loc="upper left") # 범례는 전체 다시 그릴 때만 호출

        # 이미 그려진 전체 경로 위에서 현재 위치 점만 업데이트하는 경우
        elif current_traj_idx is not None and self._last_traj and (0 <= current_traj_idx < len(self._last_traj)):
            if self._curr_scatter: # 이전 현재 위치 점 삭제
                self._curr_scatter.remove()
                self._curr_scatter = None # 확실히 None으로 설정

            x0_path, y0_path = self._last_traj[0]["x"], self._last_traj[0]["y"] # 저장된 경로의 기준점 사용
            current_point_data = self._last_traj[current_traj_idx]
            cur_x_rel = current_point_data["x"] - x0_path
            cur_y_rel = current_point_data["y"] - y0_path
            cur_z_abs = current_point_data["z"]
            
            self._curr_scatter = self.ax.scatter(
                cur_x_rel, cur_y_rel, cur_z_abs,
                color="yellow", marker='o', s=100, label="Current Position", depthshade=True, edgecolors='black'
            )
            # 범례에 "Current Position"이 이미 있다면, legend를 다시 호출할 필요는 없음
            # 또는, legend 핸들 관리를 더 정교하게 할 수 있음. 여기서는 단순화.

        self.draw_idle()


# ══════════════════════════════════════════════════════════════
# 4.  MonitoringTab  ★ UDP 패킷 파서 통합 (멀티/단일)
# ══════════════════════════════════════════════════════════════
class MonitoringTab(QWidget):
    packetReceived = pyqtSignal(str, dict)
    vehicleRemoved = pyqtSignal(str)
    trafficUpdated = pyqtSignal(dict)

    def __init__(self, parent=None, fpl_dir: str | Path | None = None):
        super().__init__(parent)

        # ── (1) 경로·UI
        self.planner = PathPlanner(str(SRC_DIR/"vertiport.csv"), str(SRC_DIR/"waypoint.csv"))
        self.map    = MapView(self.planner)
        self.detail = DetailPanel(); self.traj = Traj3DPanel()
        right = QSplitter(Qt.Vertical); right.addWidget(self.detail); right.addWidget(self.traj); right.setSizes([340,330])
        main  = QSplitter(Qt.Horizontal); main.addWidget(self.map); main.addWidget(right); main.setStretchFactor(0,4)
        QHBoxLayout(self).addWidget(main)

        # ── (2) 차량 캐시 & UDP
        self._vehicles: Dict[str, Dict] = {}
        self._traj_cache: Dict[str, List] = {}

        # 현재 시각 보관
        self._now_sec: int | None = None

        # ★★ ① FPL CSV → 스케줄 DB ---------------------------------
        self._sched_db: Dict[str, Dict] = {}
        selected_fpl: Path | None = None

        if fpl_dir:
            try:
                candidate = Path(fpl_dir).expanduser()
                if candidate.is_file():
                    candidate = candidate.parent
                candidate = candidate.resolve()
                if candidate.exists() and list(candidate.glob("*.csv")):
                    selected_fpl = candidate
                    print(f"[INFO] Using FPL folder from FleetOps: {candidate}")
                else:
                    print(f"[WARN] Provided FPL folder has no CSV files: {candidate}")
            except Exception as exc:
                print(f"[WARN] Failed to interpret provided FPL folder '{fpl_dir}': {exc}")

        if selected_fpl is None:
            hardcoded = (ROOT_DIR / "Scheduler" / "FPL_Result" / "20250815").resolve()
            if hardcoded.exists():
                selected_fpl = hardcoded
                print(f"[INFO] Using hardcoded FPL folder: {selected_fpl}")
            else:
                print(f"[WARN] Hardcoded FPL folder not found: {hardcoded}")
                selected_fpl = _latest_fpl_folder(Path.cwd(), ROOT_DIR)

        if selected_fpl is None:
            print("[WARN] No FPL folder available.")
        else:
            self._load_sched_db(selected_fpl)


        # -----------------------------------------------------------
 
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind(("127.0.0.1", 50051))
        self._sock.setblocking(False)
 
        # 0.3 초 간격으로 수신만 돌림
        self.map.bridge.clicked.connect(self._on_map_select)
        self._timer = QTimer(); self._timer.timeout.connect(self._poll_udp); self._timer.start(300)



    def _cleanup_stale(self):
        now = time.time()
        for vid, v in list(self._vehicles.items()):
            silent_for = now - v["last_update"]
            arrived    = (v["phase"] == "ARRIVED")
            if v.get("pause"):
                continue
            elif (silent_for > _STALE_SEC) or (arrived and silent_for > _ARRIVED_SEC):
                self.vehicleRemoved.emit(vid)
                # 1) 지도 레이어 제거
                self.map.remove_all(vid)
                # 2) 내부 캐시 제거
                self._vehicles.pop(vid, None)
                self._traj_cache.pop(vid, None)
                # 3) Detail 패널이 이 기체를 보고 있었으면 지워준다
                if self.detail.current_id == vid:
                    self.detail.current_id = None
                    self.detail.update({"id":"–","phase":"–", "lon":0,"lat":0,
                                        "z":0,"remain":0,"traveled":0,"total_km":0,
                                        "departure":"–","destination":"–","pax":0})



    # ───────────────────────────────────────────────────────────
    #   UDP 수신 루프 + 브로드캐스트
    # ───────────────────────────────────────────────────────────
    def _poll_udp(self):
        try:
            while True:
                data, addr = self._sock.recvfrom(65_535)
                msg = data.decode("utf-8", errors="ignore").strip()

                # ── 헬스체크 핑/퐁 -----------------------------------
                if not msg:
                    continue
                if msg == "PING":
                    self._sock.sendto(b"PONG", addr)
                    continue
                if msg == "PONG":
                    continue

                # ── JSON 파싱 ----------------------------------------
                try:
                    pkt = json.loads(msg)
                    # print("[DEBUG] incoming pkt keys:", list(pkt.keys()), "   full pkt:", pkt)
                except json.JSONDecodeError:
                    continue

                # ── 시계 문자열 처리 ("HH:MM:SS") ---------------------
                if "time" in pkt:
                    self._now_sec = time_to_sec(pkt["time"])
                    self.map.update_clock(pkt["time"])

                # ── 패킷 처리 ----------------------------------------
                if "fleet" in pkt:           # { vid : subpkt, … }
                    for vid, sub in pkt["fleet"].items():
                        self._process_remote_packet(vid, sub)
                elif "id" in pkt:            # 단일 기체
                    self._process_remote_packet(pkt["id"], pkt)

                # ── NEW: 혼잡 탭용 전체 브로드캐스트 ------------------
                #     CongestionOverlay 에게 현재 기체 dict 전달
                self.trafficUpdated.emit(self._vehicles)

        except BlockingIOError:
            # non-blocking recvfrom()일 경우 반복 탈출
            pass
        finally:
            self._cleanup_stale()            # 오래된 기체 정리

   

    # ────────────────────────────────────────────────────────────
    #   ★ 외부 기체 패킷 반영
    # ────────────────────────────────────────────────────────────

    def _load_sched_db(self, fpl_dir: Path, pattern: str = "*.csv"):
        self._sched_db.clear()
        total = 0
        for csv in fpl_dir.glob(pattern):
            df = pd.read_csv(csv, encoding="utf-8")
            seg_cols = [c for c in df.columns if c.lower().startswith("seg")]
            for _, r in df.iterrows():
                vid      = str(r["ID"]).strip()
                local_id = str(r["LocalID"]).strip()
                key      = f"{vid}:{local_id}"
                traj: list[dict] = []

                for col in seg_cols:
                    cell = str(r[col]).strip()
                    if not cell or cell.lower() == "nan":
                        break

                    # 1) Phase: 맨 앞 글자 (예: "B : ...")
                    phase = cell[0].upper() if re.match(r"^[A-Za-z]\s*:", cell) else None

                    # 2) 숫자(실수)만 골라서 lon/lat
                    nums = re.findall(r"[-+]?\d*\.\d+|\d+", cell)
                    if len(nums) < 2:
                        continue
                    lon, lat = map(float, nums[:2])

                    # 3) 좌표 -> xy(m)
                    x, y = lonlat_to_xy(lon, lat)

                    # 4) z: MissionSegment.DEFAULTS 에서 ft -> m
                    z_ft = MissionSegment.DEFAULTS.get(phase, {}).get("ending_altitude", 0)
                    z_m  = z_ft * 0.3048

                    traj.append({"x": x, "y": y, "z": z_m})

                self._sched_db[key] = dict(
                        departure   = str(r["From"]).strip(),
                        destination = str(r["To"]).strip(),
                        STD         = str(r["STD"]).strip(),
                        STA         = str(r["STA"]).strip(),
                        traj        = traj,
                    )
                total += 1

        print(f"[SchedDB] ✔ {total} plans loaded from {fpl_dir}\\{pattern}")

    # monitoring_tab.py 파일의 MonitoringTab 클래스 내에 위치

    # ─────────────────────────────────────────────────────────
    def _process_remote_packet(self, vid: str, pkt: dict):
        # 1) ID, local_id, 복합 키 생성
        vid = str(pkt.get("id", pkt.get("ID", vid))).strip()
        local_id = str(pkt.get("local_id", "")).strip()
        key = f"{vid}:{local_id}"
        paused = bool(pkt.get("pause", False))
        
        # 2) 위치·상태값 파싱
        lon       = pkt["lon"];            lat       = pkt["lat"]
        alt       = pkt.get("alt_m", pkt.get("alt", 0))
        prog_m    = pkt.get("progress_m", 0.0);      remain_m  = pkt.get("remain_m", 0.0)
        prog_km   = prog_m / 1000.0;                 remain_km = remain_m / 1000.0
        total_km  = (prog_m + remain_m)/1000.0 if (prog_m or remain_m) else 0.0
        now_real  = time.time()

        # 3) 신규 기체라면 초기화
        if key not in self._vehicles:
            sched = self._sched_db.get(key, {})

            # atd/sec, eta/sec 미리 파싱
            atd_val = pkt.get("atd", "–")
            if atd_val and atd_val not in ("-", "–"):
                etd = atd_val
                etd_sec = time_to_sec(atd_val)
            else:
                etd = "–"
                etd_sec = None

            eta_val = pkt.get("eta", "–")
            if eta_val and eta_val not in ("-", "–"):
                eta = eta_val
                eta_sec = time_to_sec(eta_val)
            else:
                eta = "–"
                eta_sec = None

            self._vehicles[key] = dict(
                id           = vid,
                local_id     = local_id,
                departure    = sched.get("departure",   "EXT"),
                destination  = sched.get("destination", "EXT"),
                phase        = pkt.get("phase", "REMOTE"),
                time         = pkt.get("time"),
                agent        = None,
                total_km     = max(total_km, 0.001),
                traveled     = prog_km,
                remain       = remain_km,
                lon          = lon,
                lat          = lat,
                z            = alt,
                std          = sched.get("STD", "–"),
                sta          = sched.get("STA", "–"),
                etd          = etd,
                eta          = eta,
                etd_sec      = etd_sec,
                eta_sec      = eta_sec,
                pax          = pkt.get("pax", 0),
                heading_deg  = float(pkt.get("heading_deg", 0.0)),
                _now_sec     = self._now_sec,
                last_update  = now_real,
                _key         = key
            )

        # 4) 기존 기체 정보 업데이트
        v = self._vehicles[key]
        v.update(
            lon=lon, lat=lat, z=alt,
            phase=pkt.get("phase", v["phase"])
        )
        v["pause"] = paused
        v["last_update"] = now_real
        v["_now_sec"] = self._now_sec

        if prog_m or remain_m:
            v["traveled"], v["remain"], v["total_km"] = prog_km, remain_km, max(total_km, 0.001)

        # ETA/ATD 재파싱 (업데이트된 값 반영)
        atd_val = pkt.get("atd", None)
        if atd_val and atd_val not in ("-", "–"):
            v["etd"] = atd_val
            v["etd_sec"] = time_to_sec(atd_val)
        else:
            v["etd"] = "–"
            v["etd_sec"] = None

        eta_val = pkt.get("eta", None)
        if eta_val and eta_val not in ("-", "–"):
            v["eta"] = eta_val
            v["eta_sec"] = time_to_sec(eta_val)
        else:
            v["eta"] = "–"
            v["eta_sec"] = None

        if v["phase"] != "ARRIVED" and v["remain"] <= 0.05:
            v["phase"] = "ARRIVED"

        # heading 업데이트
        heading = float(pkt.get("heading_deg", v.get("heading_deg", 0.0)))
        v["heading_deg"] = heading

        # ★ 추가: curr_node 자동 추정
        def _find_nearest_node(lon, lat):
            min_dist = float("inf")
            nearest = None
            for name, (n_lon, n_lat) in self.planner.nodes_geo.items():
                d = math.hypot(lon - n_lon, lat - n_lat)
                if d < min_dist:
                    min_dist = d
                    nearest = name
            return nearest

        v["curr_node"] = _find_nearest_node(lon, lat)

        # ★ 혼잡 오버레이용 남은 거리(km) 저장
        v["remain_dist"] = v["remain"]

        # 5) MapView에 마커 추가/이동
        info_key = f"{vid} ({local_id})"
        info = f"{info_key}<br>{v['departure']} ▶ {v['destination']}"
        self.map.add_or_move(
            key, lat, lon, info,
            heading,
            lambda k=key: self._on_map_select(k)
        )

        if self.detail.current_id == key:
            self._on_map_select(key)

        # AnalyticTab 개별 업데이트
        self.packetReceived.emit(key, v)
        # ★ CongestionTab 전체 업데이트
        self.trafficUpdated.emit(self._vehicles)

        # -----------------------------------------------------------------
        # 6) 방향 지시선 그리기 (UDP heading 사용)
        # -----------------------------------------------------------------
        line_length_m = 1000.0    # 지도상 1 km 길이
        lat1_rad = math.radians(lat); lon1_rad = math.radians(lon)
        bearing_rad = math.radians(heading)
        R = 6_371_000.0           # 지구 반경 (m)

        lat2_rad = math.asin(math.sin(lat1_rad) * math.cos(line_length_m / R) +
                             math.cos(lat1_rad) * math.sin(line_length_m / R) * math.cos(bearing_rad))
        lon2_rad = lon1_rad + math.atan2(math.sin(bearing_rad) * math.sin(line_length_m / R) * math.cos(lat1_rad),
                                         math.cos(line_length_m / R) - math.sin(lat1_rad) * math.sin(lat2_rad))

        lat2_deg = math.degrees(lat2_rad); lon2_deg = math.degrees(lon2_rad)
        line_coords = [[lat, lon], [lat2_deg, lon2_deg]]
        line_id = f"direct_heading_line_{key}"

        js_draw_direct_heading_line = f"""
            if (window['{line_id}']) {{
                try {{ {self.map._map_js}.removeLayer(window['{line_id}']); }} catch(e) {{}}
                delete window['{line_id}'];
            }}
            try {{
                window['{line_id}'] = L.polyline({line_coords!r}, {{
                    color: 'yellow',
                    weight: 2,
                    opacity: 0.75,
                    dashArray: '5, 5'
            }}).addTo({self.map._map_js});
            }} catch(e) {{}}
        """
        if self.map._loaded:
            self.map.page().runJavaScript(js_draw_direct_heading_line)

    # ────────────────────────────────────────────────────────────
    #   메인 루프
    # ────────────────────────────────────────────────────────────
    def _step(self):
        # (0) 시계 전진
        self._sim_sec += int(self.DT_SEC*self.SPEED_MULT); self.map.update_clock(sec_to_str(self._sim_sec))
        
        
        # (1) STD 도달 → 스폰
        while self._pending and self._pending[0]["std_sec"]<=self._sim_sec:
            self._spawn_vehicle(self._pending.pop(0))

        # (2) UDP 수신  ★ 멀티/단일 패킷 모두 처리
        try:
            while True:
                data,_ = self._sock.recvfrom(65_535)
                pkt=json.loads(data.decode("utf-8"))
                if "fleet" in pkt:  # 최신 형식
                    for vid, sub in pkt["fleet"].items(): self._process_remote_packet(vid,sub)
                elif "id" in pkt:   # 구형 테스트
                    self._process_remote_packet(pkt["id"], pkt)
        except BlockingIOError:
            pass

        # (3) 내부 기체 스텝
        for vid,v in list(self._vehicles.items()):
            ag=v["agent"]
            if ag is None: continue                 # 외부 기체
            for _ in range(self.SPEED_MULT): ag.step()
            pos=ag.position; dx=pos["x"]-v["_prev_x"]; dy=pos["y"]-v["_prev_y"]
            v.update(x=pos["x"], y=pos["y"], z=pos["z"],
                     traveled=v["traveled"]+math.hypot(dx,dy)/1000)
            v["remain"]=max(v["total_km"]-v["traveled"],0)
            v["_prev_x"],v["_prev_y"]=pos["x"],pos["y"]
            v["lon"],v["lat"]=xy_to_lonlat(v["x"],v["y"])

            # 첫 이동 → ETD
            if v["etd"]=="–" and v["traveled"]>0.01:
                v["etd"]=sec_to_str(self._sim_sec)
                if self.detail.current_id==vid: self.detail.update(v)

            info=f"{vid}<br>{v['departure']} ▶ {v['destination']}"
            self.map.add_or_move(vid,v["lat"],v["lon"],info,lambda vid=vid: self._on_map_select(vid))
            if self.detail.current_id==vid: self.detail.update(v)

            # 도착 판정
            if v["phase"]!="ARRIVED" and v["remain"]<=0.05:
                v["phase"]="ARRIVED"; v["eta"]=sec_to_str(self._sim_sec); self._finished.append(v)

            # (4) 외부 broadcast → 다른 모듈 시청용
            self._sock_out.send(json.dumps({
                "id":vid,"lon":v["lon"],"lat":v["lat"],"alt":v["z"],"phase":v["phase"]
            }).encode("utf-8"))

    # ────────────────────────────────────────────────────────────
    #   지도 마커 클릭
    # ────────────────────────────────────────────────────────────
    def _on_map_select(self, vid: str):
        v = self._vehicles[vid] 
        self.detail.update(v)   
        self.detail.current_id = vid
        full_scheduled_trajectory = self._sched_db.get(vid, {}).get("traj", [])
        
        current_display_idx = None # 3D 패널에 표시할 현재 위치 인덱스
        if full_scheduled_trajectory:
            current_pos_xy = lonlat_to_xy(v["lon"], v["lat"])
            try:
                current_display_idx = min(
                    range(len(full_scheduled_trajectory)),
                    key=lambda i: math.hypot(
                        full_scheduled_trajectory[i]["x"] - current_pos_xy[0],
                        full_scheduled_trajectory[i]["y"] - current_pos_xy[1]
                    )
                )
            except ValueError: # 경로가 비어있는 경우 min에서 에러 발생 방지
                current_display_idx = None
        
        # 3D 패널 업데이트: 전체 경로, 현재 인덱스, 항공기 ID 전달
        self.traj.show_traj(full_scheduled_trajectory, current_traj_idx=current_display_idx, aircraft_id=vid)

        # --- 2D 지도 (Leaflet) 용: 남은 경로 (기존 로직 유지) ---
        leaflet_coords_remaining = [] 
        if full_scheduled_trajectory and current_display_idx is not None: 
            remaining_path_points_for_2d_map = full_scheduled_trajectory[current_display_idx:]
            leaflet_coords_remaining = [
                [ LAT0 + (pt["y"]/1000)/KM_PER_DEG,
                  LON0 + (pt["x"]/1000)/km_per_lon(LAT0) ]
                for pt in remaining_path_points_for_2d_map
            ]
        
        js_set_pending_route = f"window.pendingRoutes[{vid!r}] = {leaflet_coords_remaining!r};"
        self.map.page().runJavaScript(js_set_pending_route)
        js_draw_route = f"drawRoute({vid!r}, {leaflet_coords_remaining!r});"
        self.map.page().runJavaScript(js_draw_route)
