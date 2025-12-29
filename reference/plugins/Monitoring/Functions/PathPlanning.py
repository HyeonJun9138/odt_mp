from __future__ import annotations

import math
import heapq
from typing import Dict, List, Tuple, Iterable

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge
from matplotlib.patches import Ellipse

# ----- 모듈 공개 심볼 --------------------------------------------------------
__all__ = [
    "PathPlanner",
    "PathVisualizer",
    "PathVisualizerGeo",   # ← 추가
    "rebuild_route",
]

# ----- matplotlib 한글 설정 --------------------------------------------------
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

# --------------------------------------------------------------------------- #
# 내부 상수 / 헬퍼
# --------------------------------------------------------------------------- #
_ARC_STEP_DEG = 5  # 원호 보간 간격(도)


def _frange(start: float, end: float, step: float) -> Iterable[float]:
    x = start
    while (step > 0 and x <= end) or (step < 0 and x >= end):
        yield x
        x += step


def _arc_points(
    cx: float,
    cy: float,
    r: float,
    start_ang: float,
    end_ang: float,
    direction: str,
    step_deg: int = _ARC_STEP_DEG,
) -> List[Tuple[float, float]]:
    """시계(R) / 반시계(L) 방향으로 원호 보간 좌표 생성"""
    points: List[Tuple[float, float]] = []

    if direction == "R":
        if end_ang > start_ang:
            end_ang -= 2 * math.pi
        thetas = list(_frange(start_ang, end_ang, -math.radians(step_deg)))
    else:
        if end_ang < start_ang:
            end_ang += 2 * math.pi
        thetas = list(_frange(start_ang, end_ang, math.radians(step_deg)))

    # 첫 점(start_ang)은 기존 노드이므로 제외하고 이어붙인다
    for a in thetas[1:]:
        points.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return points

def _km_per_deg_lat(phi):
    # 111.13209 – 0.56605 cos(2φ) + 0.00120 cos(4φ)
    return (111.13209
            - 0.56605 * math.cos(2*phi)
            + 0.00120 * math.cos(4*phi))

def _km_per_deg_lon(phi):
    # 111.41513 cos φ – 0.09455 cos(3φ) + 0.00012 cos(5φ)
    return (111.41513 * math.cos(phi)
            - 0.09455  * math.cos(3*phi)
            + 0.00012  * math.cos(5*phi))

def _km_to_dlon_dlat(dx_km, dy_km, lat0_deg):
    """km 단위 Δx·Δy → Δlon·Δlat(도).  lat0 = 기준 위도(deg)."""
    φ = math.radians(lat0_deg)
    d_lat = dy_km / _km_per_deg_lat(φ)
    d_lon = dx_km / _km_per_deg_lon(φ)
    return d_lon, d_lat

# --------------------------------------------------------------------------- #
# 1) PathPlanner
# --------------------------------------------------------------------------- #
class PathPlanner:
    """
    • Vertiport/Waypoint 엑셀 로드
    • 그래프 구축 (Vertiport-Vertiport, Waypoint-Waypoint)
    • INR/OTR/CL 접속 지점 계산
    • 최단 경로 (Dijkstra)
    """

    # .............................................................
    def __init__(self, vp_file: str, wp_file: str):
        self.df_vp = pd.read_csv(vp_file,  encoding="utf-8")   # Sources/vertiport.csv
        self.df_wp = pd.read_csv(wp_file, encoding="utf-8")   # Sources/waypoint.csv

        self._rename_columns()

        self.iport_list = self.df_vp.to_dict("records")
        self.waypoint_list = self.df_wp.to_dict("records")

        # 좌표 dict  {name: (x_km, y_km)}
        # ── (a) 내부 계산용: km 평면 좌표
        self.nodes: Dict[str, Tuple[float, float]] = {
            v["name"]: (v["x"], v["y"]) for v in self.iport_list
        }
        self.nodes.update({w["name"]: (w["x"], w["y"]) for w in self.waypoint_list})

        # ── (b) 시각화용: 위도·경도 좌표  (lon, lat 순서로 통일)
        self.nodes_geo: Dict[str, Tuple[float, float]] = {
            v["name"]: (v["lon"], v["lat"]) for v in self.iport_list
        }
        self.nodes_geo.update({w["name"]: (w["lon"], w["lat"])
                               for w in self.waypoint_list})

        self.iport_names = {v["name"] for v in self.iport_list}
        self.waypoint_names = {w["name"] for w in self.waypoint_list}

        # 그래프 {node: [(nbr, dist_km), …]}
        self.vp_graph: Dict[str, List[Tuple[str, float]]] = {n: [] for n in self.iport_names}
        self.wp_graph: Dict[str, List[Tuple[str, float]]] = {n: [] for n in self.waypoint_names}

        # _build_graphs() → vp_graph / wp_graph 채우기
        self._build_graphs()

        # INR/OTR/CL 포인트 생성
        self.io_nodes: List[Dict] = []
        self._add_inout_nodes()

    # .............................................................
    # private helpers
    # .............................................................
    def _rename_columns(self) -> None:
        """한글 컬럼명을 코드 친화적인 영문 컬럼명으로 변경"""
        self.df_vp.rename(
            columns={
                "상세 위치": "detailed_location",
                "Vertiport 명": "name",
                "Class": "class",
                "위도": "lat",
                "경도": "lon",
                "x (km)": "x",
                "y (km)": "y",
                "h (km)": "h",
                "INR(km)": "INR",
                "OTR(KM)": "OTR",
                "INR_Deg": "INR_Deg",
                "OTR_Deg": "OTR_Deg",
                "MTR(km)": "MTR",
                "LINK1": "link1",
                "LINK2": "link2",
                "LINK3": "link3",
                "LINK4": "link4",
                "Circle Turn": "turn_dir",
            },
            inplace=True,
        )
        self.df_wp.rename(
            columns={
                "Waypoint 명": "name",
                "위도": "lat",
                "경도": "lon",
                "x (km)": "x",
                "y (km)": "y",
                "h (km)": "h",
                "Link": "link",
            },
            inplace=True,
        )

    # .............................................................
    def _build_graphs(self) -> None:
        """Vertiport-Vertiport, Waypoint-Waypoint 그래프(무방향) 생성"""
        # Vertiport ↔ Vertiport
        for v in self.iport_list:
            u = v["name"]
            x1, y1 = self.nodes[u]
            for col in ("link1", "link2", "link3", "link4"):
                nbr = v.get(col)
                if isinstance(nbr, str) and nbr in self.nodes:
                    x2, y2 = self.nodes[nbr]
                    d = math.hypot(x2 - x1, y2 - y1)
                    self.vp_graph[u].append((nbr, d))
                    if nbr in self.vp_graph:
                        self.vp_graph[nbr].append((u, d))

        # Waypoint ↔ Waypoint
        for w in self.waypoint_list:
            u = w["name"]
            x1, y1 = self.nodes[u]
            for nbr in map(str.strip, str(w.get("link", "")).split(",")):
                if nbr in self.waypoint_names:
                    x2, y2 = self.nodes[nbr]
                    d = math.hypot(x2 - x1, y2 - y1)
                    self.wp_graph[u].append((nbr, d))
                    self.wp_graph[nbr].append((u, d))

    # .............................................................
    # .............................................................
    def _add_inout_nodes(self) -> None:
        """INR/OTR 회전점(TO/LD) & Port-WP 접속(CL) 노드 생성"""
        to_id, ld_id, cl_id = ({v["name"]: 0 for v in self.iport_list} for _ in range(3))

        # ── (1) INR / OTR 회전 포인트 ─────────────────────────────
        for v in self.iport_list:
            name = v["name"]
            cx, cy = self.nodes[name]                      # km-평면 기준점
            lon0, lat0 = v["lon"], v["lat"]               # 위‧경도 기준점

            for label, deg in (("INR_Deg", v.get("INR_Deg")),
                               ("OTR_Deg", v.get("OTR_Deg"))):
                if deg is None:
                    continue

                angle = math.radians(90 - deg)
                for radius in (v.get("INR", 0), v.get("OTR", 0)):
                    if radius <= 0:
                        continue

                    # km 좌표
                    x = cx + radius * math.cos(angle)
                    y = cy + radius * math.sin(angle)


                    # km → 위‧경도 정밀 변환
                    d_lon, d_lat = _km_to_dlon_dlat(x - cx, y - cy, lat0)
                    lon_new, lat_new = lon0 + d_lon, lat0 + d_lat

                    # 노드 이름 결정  # ★
                    if label == "OTR_Deg":
                        to_id[name] += 1
                        n = f"{name}_TO_{to_id[name]}"
                    else:
                        ld_id[name] += 1
                        n = f"{name}_LD_{ld_id[name]}"

                    # 저장
                    self.nodes[n]      = (x, y)
                    self.nodes_geo[n]  = (lon_new, lat_new)
                    self.io_nodes.append({"name": n, "x": x, "y": y, "type": label})

        # ── (2) Port–Waypoint 접속 CL 포인트 ────────────────────
        for v in self.iport_list:
            vx,  vy  = self.nodes[v["name"]]               # km
            lon0, lat0 = v["lon"], v["lat"]               # 위‧경도 기준점

            for col in ("link1", "link2", "link3", "link4"):
                wp = v.get(col)
                if isinstance(wp, str) and wp in self.waypoint_names:
                    wx, wy = self.nodes[wp]
                    dx, dy = wx - vx, wy - vy
                    dist   = math.hypot(dx, dy)
                    if dist == 0:
                        continue

                    r = v.get("OTR") if v.get("OTR", 0) > 0 else v.get("INR", 0)
                    x = vx + dx / dist * r                # km
                    y = vy + dy / dist * r

                    # km → 위‧경도 근사 변환  # ★
                    d_lat = (y - vy) / 111.0
                    d_lon = (x - vx) / (111.0 * math.cos(math.radians(lat0)))
                    lon_new, lat_new = lon0 + d_lon, lat0 + d_lat

                    cl_id[v["name"]] += 1
                    n = f"CL{cl_id[v['name']]}_{v['name']}_{wp}"

                    self.nodes[n]     = (x, y)
                    self.nodes_geo[n] = (lon_new, lat_new)
                    self.io_nodes.append({"name": n, "x": x, "y": y, "type": "conn"})

    # .............................................................
    # public API
    # .............................................................
    def dijkstra(self, start: str, end: str) -> Tuple[Dict[str, float], Dict[str, str | None]]:
        """Vertiport ↔ Vertiport 최단 경로 (km, prev-table 반환)"""
        dist = {n: math.inf for n in self.nodes}
        prev: Dict[str, str | None] = {n: None for n in self.nodes}

        dist[start] = 0
        pq: List[Tuple[float, str]] = [(0, start)]

        while pq:
            cd, u = heapq.heappop(pq)
            if u == end:
                break
            if cd > dist[u]:
                continue

            if u == start:
                nbrs = self.vp_graph.get(u, [])
            elif u in self.waypoint_names:
                # Waypoint → Waypoint
                nbrs = self.wp_graph.get(u, [])
                # Waypoint → 목적 Vertiport 직접 링크가 있으면 추가
                nbrs += [(end, d) for t, d in self.vp_graph.get(end, []) if t == u]
            else:
                nbrs = []

            for v, d in nbrs:
                nd = cd + d
                if nd < dist[v]:
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(pq, (nd, v))

        return dist, prev

    # .............................................................
    @staticmethod
    def reconstruct(prev: Dict[str, str | None], start: str, end: str) -> List[str]:
        """prev-table → 정방향 경로 리스트"""
        path: List[str] = []
        cur = end
        while cur:
            path.append(cur)
            if cur == start:
                break
            cur = prev[cur]
        return list(reversed(path))


# --------------------------------------------------------------------------- #
# 2) 이륙·착륙 절차 삽입
# --------------------------------------------------------------------------- #
def rebuild_route(planner: PathPlanner, raw_path: List[str]) -> List[str]:
    """
    Vertiport 중심 ↔ Waypoint 최단 경로(raw_path) 를
    ▶ Circle Turn / CL 접속 / INR/OTR Node 로 확장
    """
    full: List[str | Tuple[float, float]] = []

    # ---- 출발 & 도착 Vertiport 메타데이터
    dep_port, arr_port = raw_path[0], raw_path[-1]
    dep_meta = next(v for v in planner.iport_list if v["name"] == dep_port)
    arr_meta = next(v for v in planner.iport_list if v["name"] == arr_port)
    turn_dep = (dep_meta.get("turn_dir") or "R").upper()[0]
    turn_arr = (arr_meta.get("turn_dir") or "R").upper()[0]

    # ---- 출발부 --------------------------------------------------------------
    full.append(dep_port)
    to_inr = f"{dep_port}_TO_1"
    to_otr = f"{dep_port}_TO_2"
    if to_inr in planner.nodes:
        full.append(to_inr)
    if to_otr in planner.nodes:
        full.append(to_otr)

    first_wp = raw_path[1]
    cl_candidates = [n for n in planner.nodes if n.startswith("CL") and n.endswith(f"{dep_port}_{first_wp}")]
    if cl_candidates:
        cl = cl_candidates[0]
        # 원호 보간
        cx, cy = planner.nodes[dep_port]
        r = dep_meta["OTR"] if to_otr in planner.nodes else dep_meta["INR"]
        start_ang = math.atan2(planner.nodes[to_otr if to_otr in planner.nodes else to_inr][1] - cy,
                               planner.nodes[to_otr if to_otr in planner.nodes else to_inr][0] - cx)
        end_ang = math.atan2(planner.nodes[cl][1] - cy, planner.nodes[cl][0] - cx)
        full += _arc_points(cx, cy, r, start_ang, end_ang, turn_dep)
        full.append(cl)

    full.append(first_wp)          # Waypoint 시작점
    full.extend(raw_path[2:-1])    # 중간 Waypoint

    # ----- 착륙부 --------------------------------------------------------------
    # 기본값을 미리 선언해 두면 UnboundLocal 오류 방지
    to_inr_arr = f"{arr_port}_LD_1"
    to_otr_arr = f"{arr_port}_LD_2" if f"{arr_port}_LD_2" in planner.nodes else None

    last_wp = raw_path[-2]
    cl_candidates = [n for n in planner.nodes
                    if n.startswith("CL") and n.endswith(f"{arr_port}_{last_wp}")]

    if cl_candidates:
        cl2 = cl_candidates[0]
        full.append(cl2)

        cx, cy = planner.nodes[arr_port]
        r = arr_meta["OTR"] if to_otr_arr else arr_meta["INR"]

        start_ang = math.atan2(planner.nodes[cl2][1] - cy, planner.nodes[cl2][0] - cx)
        end_ang   = math.atan2(planner.nodes[(to_otr_arr or to_inr_arr)][1] - cy,
                            planner.nodes[(to_otr_arr or to_inr_arr)][0] - cx)
        full += _arc_points(cx, cy, r, start_ang, end_ang, turn_arr)

    # CL이 없더라도 여기까지 오면 변수는 항상 정의돼 있음
    if to_otr_arr:
        full.append(to_otr_arr)
    full.append(to_inr_arr)
    full.append(arr_port)

    return full


# --------------------------------------------------------------------------- #
# 3) PathVisualizer  (클릭-선택 시각화)  ── 옵션
# --------------------------------------------------------------------------- #
class PathVisualizer:
    """
    Vertiport 두 지점을 클릭하면
        1) Dijkstra 최단 경로 → 2) 절차 포함 → 3) 거리·그래프 표시
    """

    def __init__(self, planner: PathPlanner):
        self.p = planner
        self.nodes = planner.nodes
        self.iport_list = planner.iport_list
        self.waypoint_list = planner.waypoint_list
        self.io_nodes = planner.io_nodes

        self._selected: List[str] = []
        self._route_artists: List = []

        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.ax.set_aspect("equal")
        self._draw_base_map()

        self.cid = self.fig.canvas.mpl_connect("button_press_event", self._on_click)

    # .....................................................................
    def _draw_base_map(self) -> None:
        ax = self.ax
        ax.clear()

        # Vertiport & 원
        for v in self.iport_list:
            x, y = self.nodes[v["name"]]
            ax.plot(x, y, "ko", alpha=0.3)
            ax.text(x, y, f" {v['name']}", fontsize=7, va="bottom", color="blue", alpha=0.8)

            # INR / OTR / MTR
            ax.add_patch(Circle((x, y), v["INR"], edgecolor="green", facecolor="none", linestyle="--", alpha=0.5))
            if v.get("OTR", 0) > 0:
                ax.add_patch(Circle((x, y), v["OTR"], edgecolor="red", facecolor="none",
                                    linestyle="--", alpha=0.5))
            ax.add_patch(Circle((x, y), v["MTR"], edgecolor="purple", facecolor="none",
                                linestyle="-.", alpha=0.2))

            inr_ang = 90 - v["INR_Deg"]
            otr_ang = 90 - v["OTR_Deg"]
            ax.add_patch(Wedge((x, y), v["MTR"], inr_ang - 10, inr_ang + 10,
                               facecolor="green", alpha=0.2))
            ax.add_patch(Wedge((x, y), v["MTR"], otr_ang - 10, otr_ang + 10,
                               facecolor="red", alpha=0.2))

        # Waypoint
        for w in self.waypoint_list:
            x, y = self.nodes[w["name"]]
            ax.plot(x, y, "ko", alpha=0.6)
            ax.text(x, y, f" {w['name']}", fontsize=7, va="bottom", color="green", alpha=0.6)

        # 링크
        for v in self.iport_list:
            x1, y1 = self.nodes[v["name"]]
            for col in ("link1", "link2", "link3", "link4"):
                t = v.get(col)
                if t in self.nodes:
                    x2, y2 = self.nodes[t]
                    ax.plot([x1, x2], [y1, y2], "-", color="blue", alpha=0.2)
        for w in self.waypoint_list:
            x1, y1 = self.nodes[w["name"]]
            for t in map(str.strip, str(w.get("link", "")).split(",")):
                if t in self.nodes:
                    x2, y2 = self.nodes[t]
                    ax.plot([x1, x2], [y1, y2], "r-", alpha=1.0)

        # IO 노드
        for io in self.io_nodes:
            x, y = io["x"], io["y"]
            ax.plot(x, y, "x", color="blue", markersize=8, alpha=0.6)

        ax.set_xlabel("x (km)")
        ax.set_ylabel("y (km)")
        ax.set_title("Vertiport Network (click two Vertiports)")
        ax.grid(True)

    # .....................................................................
    def _nearest_vertiport(self, x: float, y: float, th: float = 0.5) -> str | None:
        cand, dmin = None, float("inf")
        for v in self.iport_list:
            vx, vy = self.nodes[v["name"]]
            d = math.hypot(vx - x, vy - y)
            if d < dmin:
                cand, dmin = v["name"], d
        return cand if dmin <= th else None

    # .....................................................................
    def _clear_previous_route(self):
        for art in self._route_artists:
            art.remove()
        self._route_artists.clear()
        self._draw_base_map()
        self.fig.canvas.draw_idle()

    # .....................................................................
    def _on_click(self, event):
        if event.inaxes is not self.ax or event.xdata is None:
            return

        vt = self._nearest_vertiport(event.xdata, event.ydata)
        if not vt:
            print("⚠️  Vertiport 근처를 클릭해 주세요.")
            return

        # 첫 클릭 → 이전 경로 초기화
        if not self._selected:
            self._clear_previous_route()

        m, = self.ax.plot(*self.nodes[vt], "o", markersize=12,
                          markerfacecolor="yellow", markeredgecolor="k", zorder=5)
        txt = self.ax.text(*self.nodes[vt], f"\n▶ {['Start', 'End'][len(self._selected)]}",
                           color="darkorange", fontsize=8, weight="bold")
        self._route_artists.extend([m, txt])
        self._selected.append(vt)

        # 두 점 선택 완료
        if len(self._selected) == 2:
            s, e = self._selected
            self._selected = []

            dist, prev = self.p.dijkstra(s, e)
            if math.isinf(dist.get(e, math.inf)):
                print("❌  두 Vertiport 사이에 경로가 없습니다.")
                return

            raw = self.p.reconstruct(prev, s, e)
            full = rebuild_route(self.p, raw)
            total = self._draw_route(full)
            print(f"✅  {s} → {e}  총거리 {total:.3f} km")

    # .....................................................................
    def _draw_route(self, path: List[str | Tuple[float, float]]) -> float:
        coords = [self.nodes[p] if isinstance(p, str) else p for p in path]
        xs, ys = zip(*coords)
        line, = self.ax.plot(xs, ys, "-o", color="magenta",
                             linewidth=2.5, markersize=3, zorder=4)
        self._route_artists.append(line)

        dist_km = sum(math.hypot(xs[i + 1] - xs[i], ys[i + 1] - ys[i])
                      for i in range(len(xs) - 1))

        self.ax.set_title(f"Route: {path[0]} → {path[-1]}  |  {dist_km:.3f} km")
        self.fig.canvas.draw_idle()
        return dist_km

    # .....................................................................
    def show(self):
        plt.tight_layout()
        plt.show()


# --------------------------------------------------------------------------- #
# (lon, lat) 축 전용 시각화 – PathVisualizerGeo
# --------------------------------------------------------------------------- #
class PathVisualizerGeo(PathVisualizer):
    """지리 좌표(경도·위도) 전용 시각화 + 클릭 탐색."""

    # ────────────────────────────────────────────────
    # 1) 내부 헬퍼: km 반경 → 위/경도 도(°) 근사
    #    (서울 근방 저위도 가정, 수 m 오차)
    @staticmethod
    def _km_to_deg(d_km: float, lat_deg: float) -> Tuple[float, float]:
        """
        • d_km      : 선/원 반경(km)
        • lat_deg   : 기준 위도(deg)
        반환 : (Δlon_deg, Δlat_deg)
        """
        d_lon, d_lat = _km_to_dlon_dlat(d_km, d_km, lat_deg)
        return d_lon, d_lat
    # ────────────────────────────────────────────────
    def _draw_base_map(self) -> None:
        ax = self.ax
        ax.clear()

        # ── Vertiport + 원/부채꼴
        for v in self.iport_list:
            lon, lat = self.p.nodes_geo[v["name"]]
            ax.plot(lon, lat, "ko", alpha=.3)
            ax.text(lon, lat, f" {v['name']}", fontsize=7,
                    va="bottom", color="blue", alpha=.8)

            # INR / OTR / MTR 원
            for r_km, col, ls, alp in (
                (v["INR"], "green", "--", .5),
                (v.get("OTR", 0), "red",  "--", .5),
                (v["MTR"], "purple", "-.", .2),
            ):
                if r_km <= 0: continue
                dlon, dlat = self._km_to_deg(r_km, lat)
                ell = Ellipse((lon, lat),
                              width = 2*abs(dlon),   # 경도 방향 지름
                              height= 2*abs(dlat),   # 위도 방향 지름
                              edgecolor=col, facecolor="none",
                              linestyle=ls, alpha=alp,
                              transform=ax.transData)
                ax.add_patch(ell)

            # 방향 부채꼴 (±10°)
            def _wedge(deg, color):
                rlat = self._km_to_deg(v["MTR"], lat)[1]
                ang  = 90 - deg
                w = Wedge((lon, lat), rlat,
                          ang-10, ang+10,
                          facecolor=color, alpha=.2,
                          transform=ax.transData)
                ax.add_patch(w)

            _wedge(v["INR_Deg"], "green")
            _wedge(v["OTR_Deg"], "red")

        # ── Waypoint
        for w in self.waypoint_list:
            lon, lat = self.p.nodes_geo[w["name"]]
            ax.plot(lon, lat, "ko", alpha=.6)
            ax.text(lon, lat, f" {w['name']}", fontsize=7,
                    va="bottom", color="green", alpha=.6)

        # ── 링크 (VP-blue, WP-red)
        for u, nbrs in self.p.vp_graph.items():
            lon1, lat1 = self.p.nodes_geo[u]
            for v, _ in nbrs:
                lon2, lat2 = self.p.nodes_geo[v]
                ax.plot([lon1, lon2], [lat1, lat2], "b-", alpha=.2)
        for u, nbrs in self.p.wp_graph.items():
            lon1, lat1 = self.p.nodes_geo[u]
            for v, _ in nbrs:
                lon2, lat2 = self.p.nodes_geo[v]
                ax.plot([lon1, lon2], [lat1, lat2], "r-", alpha=1.)

        # ── IO 노드
        for io in self.io_nodes:
            lon, lat = self.p.nodes_geo[io["name"]]
            ax.plot(lon, lat, "x", color="blue", markersize=8, alpha=.6)

        ax.set_xlabel("longitude (°)")
        ax.set_ylabel("latitude (°)")
        ax.set_title("Vertiport Network (lon/lat)")
        ax.grid(True)

    # ────────────────────────────────────────────────
    # 2) 최근접 Vertiport (위·경도 도 단위)
    def _nearest_vertiport(self, lon: float, lat: float,
                           th_deg: float = 0.02) -> str | None:  # ≈2 km
        cand, dmin = None, float("inf")
        for v in self.iport_list:
            lon_v, lat_v = self.p.nodes_geo[v["name"]]
            d = math.hypot(lon_v - lon, lat_v - lat)
            if d < dmin:
                cand, dmin = v["name"], d
        return cand if dmin <= th_deg else None

    # ────────────────────────────────────────────────
    # 3) 클릭 이벤트 (노드 선택)
    def _on_click(self, event):
        if event.inaxes is not self.ax or event.xdata is None:
            return

        vt = self._nearest_vertiport(event.xdata, event.ydata)
        if not vt:
            print("⚠️  Vertiport 근처(지도 상) 를 클릭하세요.")
            return

        # 첫 클릭이면 이전 경로/마커 제거
        if not self._selected:
            self._clear_previous_route()

        lon, lat = self.p.nodes_geo[vt]
        m, = self.ax.plot(lon, lat, "o", markersize=12,
                          markerfacecolor="yellow", markeredgecolor="k", zorder=5)
        txt = self.ax.text(lon, lat,
                           f"\n▶ {['Start', 'End'][len(self._selected)]}",
                           color="darkorange", fontsize=8, weight="bold")
        self._route_artists.extend([m, txt])
        self._selected.append(vt)

        # 두 점 선택 완료 → 경로 계산·표시
        if len(self._selected) == 2:
            s, e = self._selected
            self._selected.clear()

            dist, prev = self.p.dijkstra(s, e)
            if math.isinf(dist.get(e, math.inf)):
                print("❌  두 Vertiport 사이에 경로가 없습니다.")
                return

            raw  = self.p.reconstruct(prev, s, e)
            full = rebuild_route(self.p, raw)
            total = self._draw_route(full)           # ← 거리 받기
            wps = [n for n in full if isinstance(n, str) and n in self.p.waypoint_names]
            print("🛰️  사용 Waypoints:", " → ".join(wps))

            print(f"✅  {s} → {e}  총거리 {total:.3f} km")


    # 4) 경로 그리기 (km 튜플 → lon/lat 보정)
    def _draw_route(self, path):
        """
        • path : Vertiport/Waypoint/보간 튜플 목록
        - 위·경도 축에 경로를 그리고
        - km 기준 총거리를 계산해 제목에 표시
        - 반환값 : dist_km (float)
        """
        # ① 위·경도 저장
        lons, lats = [], []         

        # ② km-평면 좌표 저장(거리 계산용)
        xs_km, ys_km = [], []     

        # 기준점 초기화
        ref_lon = ref_lat = prev_x = prev_y = None

        for p in path:
            if isinstance(p, str):                     # 정식 노드
                ref_lon, ref_lat = self.p.nodes_geo[p]
                prev_x,  prev_y  = self.p.nodes[p]
                lon, lat = ref_lon, ref_lat
            else:                                     # 보간 튜플 (km)
                x_km, y_km = p
                d_lon, d_lat = _km_to_dlon_dlat(
                    x_km - prev_x, y_km - prev_y, ref_lat
                )
                lon = ref_lon + d_lon
                lat = ref_lat + d_lat
                ref_lon, ref_lat = lon, lat
                prev_x, prev_y   = x_km, y_km

            lons.append(lon); lats.append(lat)
            xs_km.append(prev_x); ys_km.append(prev_y)

        # 경로 선+점 그리기
        ln, = self.ax.plot(lons, lats, "-o", color="magenta",
                           linewidth=2.5, markersize=3, zorder=4)
        self._route_artists.append(ln)

        # 총거리 계산
        dist_km = sum(
            math.hypot(xs_km[i+1] - xs_km[i],
                       ys_km[i+1] - ys_km[i])
            for i in range(len(xs_km) - 1)
        )

        self.ax.set_title(
            f"Route : {path[0]} → {path[-1]}  |  {dist_km:.3f} km (lon/lat)"
        )
        self.fig.canvas.draw_idle()
        return dist_km

# --------------------------------------------------------------------------- #
#  간편 테스트용 main
#    • 기본: 네트워크 지리 맵 창 열기(클릭으로 경로 탐색)
#    • 옵션: --start / --end 지정 시 바로 최단경로 계산·표시
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import argparse, sys, math, os

    # ── 인자 파서 ----------------------------------------------------------
    ap = argparse.ArgumentParser(
        description="PathPlanner quick-test driver"
    )
    ap.add_argument("--csvdir", default="Monitoring/Sources",
                    help="vertiport.csv / waypoint.csv 디렉터리 (기본: Monitoring/Sources)")
    ap.add_argument("--start", "-s", help="출발 Vertiport 이름")
    ap.add_argument("--end",   "-e", help="도착 Vertiport 이름")
    ap.add_argument("--plain", action="store_true",
                    help="위경도 대신 x-y(km) 평면 맵 사용")
    args = ap.parse_args()

    vp_csv = os.path.join(args.csvdir, "vertiport.csv")
    wp_csv = os.path.join(args.csvdir, "waypoint.csv")
    if not (os.path.isfile(vp_csv) and os.path.isfile(wp_csv)):
        sys.exit(f"❌  CSV 파일을 찾을 수 없습니다: {vp_csv}, {wp_csv}")

    # ── Planner & 시각화 객체 ---------------------------------------------
    planner = PathPlanner(vp_csv, wp_csv)
    VizCls  = PathVisualizer if args.plain else PathVisualizerGeo
    viz     = VizCls(planner)

    # ── start/end 지정 시 즉시 경로 계산·삽입 ------------------------------
    if args.start and args.end:
        dist, prev = planner.dijkstra(args.start, args.end)
        if math.isinf(dist.get(args.end, math.inf)):
            sys.exit(f"❌  '{args.start}' → '{args.end}' 경로가 없습니다.")

        raw  = planner.reconstruct(prev, args.start, args.end)
        full = rebuild_route(planner, raw)
        viz._draw_route(full)        # 시각화 축에 바로 그리기
        print(f"✅  {args.start} → {args.end}  |  거리 {dist[args.end]:.3f} km")

    # ── GUI 표시 -----------------------------------------------------------
    viz.show()
