#!/usr/bin/env python
"""
congestion_tab.py ― UAM Traffic GUI ▸ 혼잡(Heat-map) + 버티포트/네트워크 탭
"""
from __future__ import annotations
import json, math, os, tempfile
from typing import Dict, List, Tuple
from datetime import datetime, timedelta

from PyQt5.QtCore    import Qt, QTimer, QUrl
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QSplitter, QTableWidget, QTableWidgetItem


import folium
from folium import features

import branca

import sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))  # …/TrafficSim_System
sys.path.insert(0, ROOT)

from Monitoring.Functions.PathPlanning import PathPlanner
from Tabs.base_tab import Tab
from typing import List

BASE_DIR   = os.path.join(os.path.dirname(__file__), "..", "..")   # TrafficSim_System
VP_CSV     = os.path.join(BASE_DIR, "Monitoring", "Sources", "vertiport.csv")
WP_CSV     = os.path.join(BASE_DIR, "Monitoring", "Sources", "waypoint.csv")
PLANNER    = PathPlanner(VP_CSV, WP_CSV)   # 전역 하나만 만들어 재사용

# ──────────────────────────────────────────────────────────
# 거리 변환 및 단계 상수 (TestTab에서 쓰던 값 동일)
_KM_PER_DEG_LAT   = 111.0               # 위도 1도 ≒ 111 km
_KM_PER_DEG_LON   =  88.9               # 경도 1도 ≒  88.9 km (서울 근방)
_TAKEOFF_PHASES   = set("BCDE")         # 이륙(상승) 단계 코드
_LANDING_PHASES   = set("GHIJ")         # 접근(하강) 단계 코드
# ──────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────
# Heat-map + 네트워크 Folium 위젯
# ──────────────────────────────────────────────────────────
class HeatmapWidget(QWebEngineView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._map_ready      = False
        self._pending        = []                # 점 히트맵 큐
        self._pending_edges  = []                # 스타일(혼잡도) 큐
        self._pending_meta   = []                # 팝업 메타데이터 큐  ← 추가
        self._build_base_map()
        self.loadFinished.connect(self._on_loaded)

    def update_wp_congestion(self, meta: Dict[str, dict]):
        if not meta:
            return
        if self._map_ready:
            self.page().runJavaScript(f"window.updateWpCongestion({json.dumps(meta)});")
        else:
            # 맵 로딩 직후 flush하도록 큐에 쌓아두고, _on_loaded()에서 비워도 OK
            self._pending_meta.append(meta)

    def _build_base_map(self):
        center = [37.5665, 126.9780]                     # 서울
        fmap   = folium.Map(center, zoom_start=12, tiles="CartoDB positron")

        # ── ① Vertiport 마커 ------------------------------------------------
        for vp in PLANNER.iport_list:
            lon, lat = PLANNER.nodes_geo[vp["name"]]
            folium.CircleMarker([lat, lon], radius=6, weight=2,
                                color="#333", fill=True,
                                fill_color="#FFDD00", fill_opacity=0.9,
                                popup=f"Vertiport {vp['name']}").add_to(fmap)

        # ── ② 링크 GeoJSON --------------------------------------------------
        feats, self._key2eid = [], {}
        eid = 0
        for g in (PLANNER.vp_graph, PLANNER.wp_graph):
            for a, nbrs in g.items():
                for b, _ in nbrs:
                    key = tuple(sorted((a, b)))
                    if key in self._key2eid:
                        continue
                    lon1, lat1 = PLANNER.nodes_geo[key[0]]
                    lon2, lat2 = PLANNER.nodes_geo[key[1]]
                    feats.append({
                        "type": "Feature",
                        "geometry": {
                            "type": "LineString",
                            "coordinates": [[lon1, lat1], [lon2, lat2]]
                        },
                        "properties": {"eid": eid}
                    })
                    self._key2eid[key] = eid
                    eid += 1

        gj = features.GeoJson(
            {"type": "FeatureCollection", "features": feats},
            style_function=lambda f: {"color": "#8888FF", "weight": 4, "opacity": 0.3}
        ).add_to(fmap)

        # ── ③ WP 좌표 사전(JSON) --------------------------------------------
        wp_names = set()
        for a, nbrs in PLANNER.wp_graph.items():
            wp_names.add(a)
            for b, _ in nbrs:
                wp_names.add(b)
        wp_data = {
            nm: [PLANNER.nodes_geo[nm][1], PLANNER.nodes_geo[nm][0]]  # [lat, lon]
            for nm in wp_names if nm in PLANNER.nodes_geo
        }
        wp_json = json.dumps(wp_data, ensure_ascii=False)

        # ── ④ JS: 스타일/팝업 + 웨이포인트 혼잡 업데이트 함수 -----------------
        js = f"""
        (function register() {{
            function init() {{
                if (typeof {gj.get_name()} === 'undefined') {{
                    setTimeout(init, 50);
                    return;
                }}
                const edgeLayers = {{}};
                {gj.get_name()}.eachLayer(l => {{
                    const eid = l.feature.properties.eid;
                    edgeLayers[eid] = l;
                    l.on('click', () => {{ if (l.getPopup()) l.openPopup(); }});
                }});

                // 혼잡도(0~10) → Red/Blue 그라데이션
                function weightToColor(w) {{
                    const v = Math.max(0, Math.min(10, w));
                    const r = Math.round(255 * v / 10);
                    const b = 255 - r;
                    return `rgb(${{r}},0,${{b}})`;
                }}

                /* ① 링크 스타일(혼잡도) 갱신 */
                window.updateEdgeCongestion = function(raw) {{
                    for (const eid in raw) {{
                        const w = raw[eid];
                        const layer = edgeLayers[eid];
                        if (!layer) continue;
                        layer.setStyle({{
                            color  : weightToColor(w),
                            weight : 15,
                            opacity: 0.4
                        }});
                    }}
                }};

                /* ② 링크 팝업 메타 갱신 */
                window.updateEdgeMeta = function(meta) {{
                    for (const eid in meta) {{
                        const layer = edgeLayers[eid];
                        if (!layer) continue;
                        const m = meta[eid];   // {{name, density, count, level}}
                        const html = `
                            <b>${{m.name}}</b><br/>
                            혼잡도 : ${{m.density.toFixed(3)}}<br/>
                            비행체 수 : ${{m.count}}<br/>
                            PC Lv : ${{m.level}}
                        `;
                        layer.bindPopup(html);
                    }}
                }};

                /* ③ 웨이포인트 레이어 생성 */
                const wpLayers = {{}};
                const WP_DATA = {wp_json};
                // 맵 핸들
                let theMap = null;
                for (var k in window) {{
                    try {{ if (window[k] && window[k] instanceof L.Map) {{ theMap = window[k]; break; }} }} catch(e){{}}
                }}
                for (const name in WP_DATA) {{
                    const ll = WP_DATA[name];
                    const cm = L.circleMarker([ll[0], ll[1]], {{
                        radius: 4, weight: 1, color: "#666",
                        fillColor: "#888", fillOpacity: 0.35
                    }}).addTo(theMap);
                    wpLayers[name] = cm;
                }}

                /* ④ 웨이포인트 혼잡(점) 갱신
                meta[name] = {{score, count, "L-1000", "R-1000", "L-2000", "R-2000", OTHER}} */
                window.updateWpCongestion = function(meta) {{
                    for (const name in meta) {{
                        const m = meta[name];
                        const layer = wpLayers[name];
                        if (!layer) continue;
                        const clr = weightToColor(Math.min(10, m.score || 0));
                        layer.setStyle({{color: clr, fillColor: clr, weight: 2, fillOpacity: 0.65}});
                        layer.setRadius(4 + Math.min(12, m.count || 0));
                        const html = `
                            <b>${{name}}</b><br/>
                            NodeScore : ${{(m.score||0).toFixed(2)}}<br/>
                            Count     : ${{m.count||0}}<br/>
                            L-1000 : ${{m["L-1000"]||0}} / R-1000 : ${{m["R-1000"]||0}}<br/>
                            L-2000 : ${{m["L-2000"]||0}} / R-2000 : ${{m["R-2000"]||0}}<br/>
                            OTHER  : ${{m["OTHER"]||0}}
                        `;
                        layer.bindPopup(html);
                    }}
                }};
            }}
            init();
        }})();
        """
        fmap.get_root().script.add_child(folium.Element(js))

        # ── ⑤ 컬러맵(Legend) --------------------------------------------
        branca.colormap.LinearColormap(
            colors=['blue', 'red'],
            vmin=0, vmax=10,
            caption='Link Congestion Density (0–10)'
        ).add_to(fmap)

        # ── ⑥ HTML 저장 & 로드 ------------------------------------------
        self._html = tempfile.NamedTemporaryFile(suffix=".html", delete=False).name
        fmap.save(self._html)
        self.load(QUrl.fromLocalFile(self._html))

    # .....................................................

    def show_waypoint_congestion(self, items):
        """
        Leaflet 지도 위에 웨이포인트 혼잡을 원형 마커로 표시한다.

        Parameters
        ----------
        items : Iterable[dict | tuple]
            각 원소는 다음 중 하나:
            • dict: {
                    "lat": float, "lon": float,                # 필수
                    "density": float,                          # 0~10 권장 (0=파랑, 10=빨강)
                    "name": str="",                            # 팝업 제목
                    "count": int=0,                            # 동시 비행체 수(옵션)
                    "level": int|None=None,                    # PC Lv(옵션)
                    "n1000": int=0, "n2000": int=0,            # 고도별 수(옵션)
                    "lane": str|None=None                      # 'L' 또는 'R'(옵션)
                }
            • tuple: (lat, lon, density) 또는 (lat, lon, density, name)
        """
        import json

        # 1) 파이썬 → JS 전달용 정규화
        norm = []
        for it in (items or []):
            if isinstance(it, dict):
                lat = it.get("lat"); lon = it.get("lon")
                den = it.get("density", 0) or 0
                nm  = it.get("name", "")
                cnt = int(it.get("count", 0) or 0)
                lvl = it.get("level", None)
                n1k = int(it.get("n1000", 0) or 0)
                n2k = int(it.get("n2000", 0) or 0)
                lane = (it.get("lane") or it.get("dir") or "")
            else:
                # tuple/list
                lat = it[0] if len(it) > 0 else None
                lon = it[1] if len(it) > 1 else None
                den = it[2] if len(it) > 2 else 0
                nm  = it[3] if len(it) > 3 else ""
                cnt = 0; lvl = None; n1k = 0; n2k = 0; lane = ""
            if lat is None or lon is None:
                continue
            norm.append({
                "lat": float(lat), "lon": float(lon),
                "density": float(den), "name": str(nm),
                "count": int(cnt), "level": (None if lvl is None else int(lvl)),
                "n1000": int(n1k), "n2000": int(n2k), "lane": str(lane or "")
            })

        if not norm:
            return

        payload = json.dumps(norm, ensure_ascii=False)

        # 2) 지도에 그리는 JS (맵 준비가 안 돼 있으면 재시도)
        js = f"""
        (function(data){{
        var tries = 0;
        function color(v){{
            v = Math.max(0, Math.min(10, +v||0));
            var r = Math.round(255 * v / 10), b = 255 - r;
            return 'rgb(' + r + ',0,' + b + ')';    // 0=파랑 → 10=빨강
        }}
        function findMap(){{
            for (var k in window) {{
            try {{ if (window[k] && window[k] instanceof L.Map) return window[k]; }}
            catch(e){{}}
            }}
            return null;
        }}
        function draw(){{
            var map = findMap();
            if (!map) {{
            if (tries++ < 30) return setTimeout(draw, 100);  // 최대 3초 재시도
            console.warn('Leaflet map not found.');
            return;
            }}
            if (!window._wpGroup) window._wpGroup = L.layerGroup().addTo(map);
            window._wpGroup.clearLayers();

            data.forEach(function(d){{
            if (isNaN(d.lat) || isNaN(d.lon)) return;
            var rad = 6 + (d.density||0) * 1.8;       // 밀도에 비례
            var col = color(d.density||0);
            var mk = L.circleMarker([d.lat, d.lon], {{
                radius: rad, color: col, weight: 2,
                fillColor: col, fillOpacity: 0.65
            }}).addTo(window._wpGroup);

            var lines = [];
            if (d.name)  lines.push('<b>'+String(d.name)+'</b>');
            lines.push('혼잡도 : ' + (+(d.density||0)).toFixed(3));
            if (d.count) lines.push('비행체 수 : ' + String(d.count));
            if (d.level!=null) lines.push('PC Lv : ' + String(d.level));
            if (d.n1000 || d.n2000) lines.push('고도 : 1000ft ' + (d.n1000||0) + ' / 2000ft ' + (d.n2000||0));
            if (d.lane) lines.push('방향 : ' + String(d.lane));
            mk.bindPopup(lines.join('<br/>'));
            }});
        }}
        draw();
        }})({payload});
        """

        try:
            self.page().runJavaScript(js)
        except Exception:
            pass



    def update_edge_meta(self, meta: Dict[int, dict]):
        """eid → {name, density, count, level} 사전 전달"""
        if not meta:
            return
        if self._map_ready:
            self.page().runJavaScript(
                f"window.updateEdgeMeta({json.dumps(meta)});")
        else:
            self._pending_meta.append(meta)

    def update_edge_congestion(self, raw: Dict[int, float]):
        if not raw:
            return
        if self._map_ready:
            self.page().runJavaScript(
                f"window.updateEdgeCongestion({json.dumps(raw)});")
        else:
            self._pending_edges.append(raw)

    def _on_loaded(self, ok: bool):
        self._map_ready = ok
        if not ok:
            return

        # 히트맵‧스타일 큐 비우기
        for pts in self._pending:
            self._push_js(pts)
        self._pending.clear()

        for raw in self._pending_edges:
            self.page().runJavaScript(
                f"window.updateEdgeCongestion({json.dumps(raw)});")
        self._pending_edges.clear()

        for meta in self._pending_meta:
            self.page().runJavaScript(
                f"window.updateEdgeMeta({json.dumps(meta)});")
        self._pending_meta.clear()

        # 이후 도착할 큐도 주기적으로 flush
        QTimer.singleShot(100, self._flush_pending_edges)
        QTimer.singleShot(100, self._flush_pending_meta)
            
    def _flush_pending_meta(self):
        if not self._map_ready:
            return
        for meta in self._pending_meta:
            self.page().runJavaScript(
                f"window.updateEdgeMeta({json.dumps(meta)});")
        self._pending_meta.clear()

    def _flush_pending_edges(self):
        """_pending_edges 큐를 한번에 JS로 전달 – QTimer용"""
        if not self._map_ready:
            return
        for raw in self._pending_edges:
            self.page().runJavaScript(
                f"window.updateEdgeCongestion({json.dumps(raw)});")
        self._pending_edges.clear()

    # .....................................................
    def _push_js(self, pts):
        self.page().runJavaScript(
            f"window.updateHeat({json.dumps(pts)});")

    def update_heatmap(self, points: List[Tuple[float,float,float]]):
        clean = [
            (lat, lon, w) for lat, lon, w in points
            if lat is not None and lon is not None and not math.isnan(lat) and not math.isnan(lon)
        ]
        if not clean:
            return
        if self._map_ready:
            self._push_js(clean)
        else:
            self._pending.append(clean)

    def closeEvent(self, ev):
        if os.path.exists(self._html):
            os.remove(self._html)
        super().closeEvent(ev)

# ──────────────────────────────────────────────────────────
# 혼잡 탭
# ──────────────────────────────────────────────────────────
class CongestionTab(Tab):
    def __init__(self, parent=None):
        self._sim_time = None
        self.heatmap_widget = HeatmapWidget(parent)
        # ② 그 다음에 Tab.__init__ → _build_ui() → create_map_widget() 호출
        super().__init__("Congestion", parent)

        # ── 테이블 초기화: 행, 열 수, 헤더, 초기 아이템 삽입 ──────────
        # vertiport 이름 리스트는 미리 준비
        self._vp_names = [vp["name"] for vp in PLANNER.iport_list]

        # 열 개수 2개, 헤더 라벨 설정
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(["TO (B–E)", "LD (G–J)"])

        # 행 개수 = VP 개수
        self.table.setRowCount(len(self._vp_names))
        for row, vp in enumerate(self._vp_names):
            # 세로 헤더에 이름
            self.table.setVerticalHeaderItem(row,
                QTableWidgetItem(vp))
            # 각 셀에 0으로 초기화
            for col in (0, 1):
                self.table.setItem(row, col, QTableWidgetItem("0"))

        # ── 6) 내부 데이터 / 타이머 ───────────────────────────────
        self._data: Dict[str, dict] = {}
        self._vp_names = [vp["name"] for vp in PLANNER.iport_list]
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._update_heat)
        self._timer.start(500)

    def create_map_widget(self) -> QWidget:
        # 3) Tab 레이아웃의 map_view 자리에 HeatmapWidget 연결
        return self.heatmap_widget

    def process_new_data_packet(self, vid: str, ac: dict):
        # 1) 시뮬레이션 타임스탬프 갱신
        ts = ac.get("time") or ac.get("timestamp")
        if ts is not None:
            try:
                # HH:MM:SS 형식이라면
                self._sim_time = datetime.strptime(ts, "%H:%M:%S")
            except ValueError:
                try:
                    # ISO 포맷 문자열이라면
                    self._sim_time = datetime.fromisoformat(ts)
                except ValueError:
                    try:
                        # epoch 초라면
                        self._sim_time = datetime.fromtimestamp(float(ts))
                    except Exception:
                        self._sim_time = None

        # 2) 위치/단계 + lane/alt 저장  (★추가)
        self._data[vid] = {
            "lat":   ac.get("lat"),
            "lon":   ac.get("lon"),
            "phase": str(ac.get("phase","")).upper(),
            "lane":  str(ac.get("lane") or "").upper(),
            "alt_m": ac.get("alt_m"),
            "heading_deg": ac.get("heading_deg")
        }


    def remove_vehicle(self, vid: str):
        self._data.pop(vid, None)

    def _update_heat(self):
        # 0) 준비
        if not self.heatmap_widget._map_ready:
            return
        if not hasattr(self, "_alerted"):
            self._alerted: set[Tuple[str,str]] = set()

        # 1) edge 좌표 캐시 (lon1,lat1,lon2,lat2)
        if not hasattr(self, "_edge_info"):
            self._edge_info = {
                key: (*PLANNER.nodes_geo[key[0]], *PLANNER.nodes_geo[key[1]])
                for key in self.heatmap_widget._key2eid
            }

        # 1-1) waypoint 좌표 캐시 (노드 혼잡 계산용)
        if not hasattr(self, "_wp_geo"):
            names = set()
            for a, nbrs in PLANNER.wp_graph.items():
                names.add(a)
                for b, _ in nbrs:
                    names.add(b)
            self._wp_geo = {nm: PLANNER.nodes_geo[nm] for nm in names if nm in PLANNER.nodes_geo}

        # === A) 링크 혼잡(기존) – 거리 단위 보정 =========================
        SNAP_KM = 0.5
        link_cnt = {k: 0 for k in self._edge_info}

        for d in self._data.values():
            lon, lat = d.get("lon"), d.get("lat")
            if lon is None or lat is None:
                continue

            best, best_dkm = None, SNAP_KM
            for key, (x1, y1, x2, y2) in self._edge_info.items():
                # 선분-점 최소거리(km)
                dx, dy = lon - x1, lat - y1
                ux, uy = x2 - x1, y2 - y1
                seg2 = ux*ux + uy*uy
                if seg2 == 0:
                    continue
                t = max(0.0, min(1.0, (dx*ux + dy*uy) / seg2))
                px, py = x1 + t*ux, y1 + t*uy
                # deg → km 변환 (경도: 위도에 따른 축척 보정)
                dx_km = (lon - px) * _KM_PER_DEG_LON
                dy_km = (lat - py) * _KM_PER_DEG_LAT
                d_km  = math.hypot(dx_km, dy_km)
                if d_km < best_dkm:
                    best_dkm, best = d_km, key
            if best:
                link_cnt[tuple(sorted(best))] += 1

        # 링크 길이(km)로 나눠 밀도 계산
        dens = {}
        for key, cnt in link_cnt.items():
            lon1, lat1, lon2, lat2 = self._edge_info[key]
            dx_km = (lon2 - lon1) * _KM_PER_DEG_LON
            dy_km = (lat2 - lat1) * _KM_PER_DEG_LAT
            km = max(1e-6, math.hypot(dx_km, dy_km))
            dens[key] = cnt / km

        # 0~10 스케일로 JS 전달
        SCALE = 10.0
        js_dens = { self.heatmap_widget._key2eid[key]: round(min(10.0, d * SCALE), 2)
                    for key, d in dens.items() }
        self.heatmap_widget.update_edge_congestion(js_dens)

        # 팝업 메타(링크)
        edge_meta = {}
        for key in dens:
            eid = self.heatmap_widget._key2eid[key]
            density = round(dens[key], 3)
            cnt = link_cnt[key]
            level = 1 if density <= 0.6 else (2 if density <= 0.8 else 3)   # 임계는 추측입니다
            edge_meta[eid] = {"name": f"{key[0]}↔{key[1]}", "density": density, "count": cnt, "level": level}
        self.heatmap_widget.update_edge_meta(edge_meta)

        # === B) 웨이포인트(노드) 혼잡 – L/R × 1000/2000 =================
        # lane 파서
        def lane_key(lane_str, alt_m):
            if not lane_str:
                # alt_m으로 근사 분류(확실하지 않음)
                if alt_m is None:
                    return "U-0000"
                ft = float(alt_m) / 0.3048
                band = "2000" if ft >= 1500 else "1000"
                return "U-" + band
            lane_str = lane_str.strip().upper()
            # 예: "L-2000", "R2000", " L_1000 "
            import re
            m = re.match(r"([LR])[\-_ ]?(\d{3,4})", lane_str)
            if not m:
                return lane_str
            side, alt = m.group(1), m.group(2)
            return f"{side}-{alt}"

        R_NODE_KM = 0.25
        wp_counts = {nm: {"L-1000":0,"R-1000":0,"L-2000":0,"R-2000":0,"OTHER":0}
                    for nm in self._wp_geo}

        for d in self._data.values():
            lon, lat = d.get("lon"), d.get("lat")
            if lon is None or lat is None:
                continue
            # 가장 가까운 WP (R_NODE_KM 이내)
            best_nm, best_dkm = None, R_NODE_KM
            for nm, (wlon, wlat) in self._wp_geo.items():
                dx_km = (lon - wlon) * _KM_PER_DEG_LON
                dy_km = (lat - wlat) * _KM_PER_DEG_LAT
                d_km  = math.hypot(dx_km, dy_km)
                if d_km < best_dkm:
                    best_dkm, best_nm = d_km, nm
            if not best_nm:
                continue
            lk = lane_key(d.get("lane"), d.get("alt_m"))
            if lk in wp_counts[best_nm]:
                wp_counts[best_nm][lk] += 1
            else:
                wp_counts[best_nm]["OTHER"] += 1

        # 노드 점수: lane 초과수 합을 0~10으로
        wp_meta = {}
        for nm, cnts in wp_counts.items():
            excess = sum(max(0, cnts[k]-1) for k in ("L-1000","R-1000","L-2000","R-2000"))
            score  = min(10.0, 5.0 * excess)     # 계수 5는 튜닝 값(추측)
            total  = sum(cnts.values())
            wp_meta[nm] = {
                "name": nm, "score": score, "count": total,
                "L-1000": cnts["L-1000"], "R-1000": cnts["R-1000"],
                "L-2000": cnts["L-2000"], "R-2000": cnts["R-2000"],
                "OTHER": cnts["OTHER"]
            }

        # 지도에 노드 혼잡 반영(원 크기/색 + 팝업)
        self.heatmap_widget.update_wp_congestion(wp_meta)

        # === C) 알림/표(기존) ============================================
        DENS_TH, CLEAR_TH = 0.6, 0.4
        now_alert = {key for key, d in dens.items() if d >= DENS_TH}
        just_cleared = [k for k in self._alerted if dens.get(k,0)<CLEAR_TH]
        for k in just_cleared: self._alerted.remove(k)
        self._alerted |= now_alert

        start = self._sim_time if self._sim_time else datetime.now()
        end   = start + timedelta(minutes=10)
        ts = (start.strftime("%H:%M:%S"), end.strftime("%H:%M:%S"))

        lv1, lv2, lv3 = [], [], []
        for key in sorted(self._alerted):
            d = dens[key]; name = f"{key[0]}↔{key[1]}"
            if 0.5 < d <= 0.6:   lv1.append(name)
            elif 0.6 < d <= 0.8: lv2.append(name)
            elif d > 0.8:        lv3.append(name)

        parts = []
        if lv3:
            parts += [f'🔴 <b>혼잡도 Lv 3 (고밀도)</b> {ts[0]} ~ {ts[1]}'] + [f'    • {n}' for n in lv3]
        if lv2:
            parts += [f'🟠 <b>혼잡도 Lv 2 (중밀도)</b> {ts[0]} ~ {ts[1]}'] + [f'    • {n}' for n in lv2]
        if lv1:
            parts += [f'🟢 <b>혼잡도 Lv 1 (저밀도)</b> {ts[0]} ~ {ts[1]}'] + [f'    • {n}' for n in lv1]
        if not parts:
            parts.append('✅ <b>All Clear</b> 현재 과밀 구간 없음')
        self.msg_box.setHtml("".join(f"<p>{line}</p>" for line in parts))

        try:
            main_win = self.window()
            if hasattr(main_win, 'main_tab'):
                main_win.main_tab.update_congestion_lv3(lv3)
        except:
            pass

        # 버티포트 TO/LD 표(기존)
        to_cnt = {vp: 0 for vp in self._vp_names}
        ld_cnt = {vp: 0 for vp in self._vp_names}
        for d in self._data.values():
            ph = d.get("phase","")
            lon, lat = d.get("lon"), d.get("lat")
            if lon is None or lat is None: continue
            best_vp, best_d = None, 1.0
            for vp in self._vp_names:
                vlon, vlat = PLANNER.nodes_geo[vp]
                dkm = math.hypot((lon-vlon)*_KM_PER_DEG_LON, (lat-vlat)*_KM_PER_DEG_LAT)
                if dkm < best_d: best_d, best_vp = dkm, vp
            if best_vp:
                if ph in _TAKEOFF_PHASES: to_cnt[best_vp] += 1
                elif ph in _LANDING_PHASES: ld_cnt[best_vp] += 1

        for row, vp in enumerate(self._vp_names):
            self.table.item(row, 0).setText(str(to_cnt[vp]))
            self.table.item(row, 1).setText(str(ld_cnt[vp]))

            # ── 0) 지도 준비 체크
            if not self.heatmap_widget._map_ready:
                return
            
            if not hasattr(self, "_alerted"):
                self._alerted: set[Tuple[str,str]] = set()

            # ── 1) edge_info 캐시
            if not hasattr(self, "_edge_info"):
                self._edge_info = {
                    key: (
                        *PLANNER.nodes_geo[key[0]],
                        *PLANNER.nodes_geo[key[1]]
                    )
                    for key in self.heatmap_widget._key2eid
                }

            # ── 2) 링크별 항공기 카운트
            _SNAP_KM = 0.5
            link_cnt = {k: 0 for k in self._edge_info}
            for d in self._data.values():
                lon, lat = d.get("lon"), d.get("lat")
                if lon is None or lat is None:
                    continue

                best, best_d = None, _SNAP_KM
                for (a, b), (x1, y1, x2, y2) in self._edge_info.items():
                    # 선분-점 거리 계산
                    dx, dy = lon - x1, lat - y1
                    ux, uy = x2 - x1, y2 - y1
                    seg2 = ux*ux + uy*uy
                    if seg2 == 0: continue
                    t = max(0., min(1., (dx*ux + dy*uy) / seg2))
                    px, py = x1 + t*ux, y1 + t*uy
                    d0 = math.hypot(lon - px, lat - py)
                    if d0 < best_d:
                        best_d, best = d0, (a, b)
                if best:
                    link_cnt[tuple(sorted(best))] += 1

            # ── 3) raw 밀도(d = ac/km) 계산 & 저장 ────────────────────
            dens: Dict[Tuple[str,str], float] = {}
            for key, cnt in link_cnt.items():
                lon1, lat1, lon2, lat2 = self._edge_info[key]
                km = math.hypot(lon2 - lon1, lat2 - lat1) * _KM_PER_DEG_LAT
                dens[key] = cnt / (km + 1e-6)

            # ── 4) raw dens → 0~10 스케일 JS용 데이터 준비 ─────────────
            SCALE = 10.0
            js_dens: Dict[int, float] = {}
            for key, d in dens.items():
                eid = self.heatmap_widget._key2eid[key]
                js_dens[eid] = round(min(10.0, d * SCALE), 2)
    

            # ── 4) JS 호출
            self.heatmap_widget.update_edge_congestion(js_dens)

            # 팝업용 메타데이터 준비
            edge_meta = {}
            for key in dens:
                eid      = self.heatmap_widget._key2eid[key]
                density  = round(dens[key], 3)
                cnt      = link_cnt[key]
                level    = 1 if density <= 0.6 else (2 if density <= 0.8 else 3)
                edge_meta[eid] = {
                    "name"    : f"{key[0]}↔{key[1]}",
                    "density" : density,
                    "count"   : cnt,
                    "level"   : level
                }
            self.heatmap_widget.update_edge_meta(edge_meta)

            # ── 4.1) 과밀 링크 집합 계산 (TestTab 참고) ─────────────────
            DENS_TH  = 0.6
            CLEAR_TH = 0.4
            now_alert: set[Tuple[str,str]] = {
                key for key, d in dens.items() if d >= DENS_TH
            }

            # (A) 해제된 링크
            just_cleared = [k for k in self._alerted if dens.get(k,0)<CLEAR_TH]
            for k in just_cleared:
                self._alerted.remove(k)

            # (B) 신규/유지 과밀
            self._alerted |= now_alert

            # ── 5) 시뮬레이션 시간 기준 “발생시각 ~ +10분” 계산 ────────────
            if self._sim_time:
                start = self._sim_time
            else:
                start = datetime.now()
            end = start + timedelta(minutes=10)
            ts_fmt = "%H:%M:%S"
            t0 = start.strftime(ts_fmt)
            t1 = end.strftime(ts_fmt)

            # ── 6) HTML 메시지 생성 & 표시 ───────────────────────────────
            lv1, lv2, lv3 = [], [], []
            for key in sorted(self._alerted):
                d = dens[key]
                name = f"{key[0]}↔{key[1]}"
                if 0.5 < d <= 0.6:   lv1.append(name)
                elif 0.6 < d <= 0.8: lv2.append(name)
                elif d > 0.8:        lv3.append(name)

        # 한 줄씩 쌓아서 <p>태그로 분리
            parts = []
            if lv3:
                parts.append(f'🔴 <b>혼잡도 Lv 3 (고밀도)</b> {t0} ~ {t1}')
            for name in lv3:
                parts.append(f'    • {name}')
            if lv2:
                parts.append(f'🟠 <b>혼잡도 Lv 2 (중밀도)</b> {t0} ~ {t1}')
            for name in lv2:
                parts.append(f'    • {name}')
            if lv1:
                parts.append(f'🟢 <b>혼잡도 Lv 1 (저밀도)</b> {t0} ~ {t1}')
            for name in lv1:
                parts.append(f'    • {name}')

            if not parts:
                parts.append('✅ <b>All Clear</b> 현재 과밀 구간 없음')

            # <p>…</p> 태그 씌워서 한 번에 렌더
            html = "".join(f"<p>{line}</p>" for line in parts)
            self.msg_box.setHtml(html)

            # ── MainTab에 과밀 Lv3 알림
            try:
                main_win = self.window()
                if hasattr(main_win, 'main_tab'):
                    main_win.main_tab.update_congestion_lv3(lv3)
            except:
                pass
            # ── 5) TO/LD 테이블 갱신
            to_cnt = {vp: 0 for vp in self._vp_names}
            ld_cnt = {vp: 0 for vp in self._vp_names}
            for d in self._data.values():
                ph = d.get("phase", "")
                lon, lat = d.get("lon"), d.get("lat")
                if lon is None or lat is None:
                    continue
                # 1 km 이내 최단거리 VP 찾기
                best_vp, best_d = None, 1.0
                for vp in self._vp_names:
                    vlon, vlat = PLANNER.nodes_geo[vp]
                    dkm = math.hypot((lon - vlon) * _KM_PER_DEG_LON,
                                    (lat - vlat) * _KM_PER_DEG_LAT)
                    if dkm < best_d:
                        best_d, best_vp = dkm, vp
                if best_vp:
                    if ph in _TAKEOFF_PHASES:      to_cnt[best_vp] += 1
                    elif ph in _LANDING_PHASES:    ld_cnt[best_vp] += 1

            # 테이블에 반영
            for row, vp in enumerate(self._vp_names):
                self.table.item(row, 0).setText(str(to_cnt[vp]))
                self.table.item(row, 1).setText(str(ld_cnt[vp]))