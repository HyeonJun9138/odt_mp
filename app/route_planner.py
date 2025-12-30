from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import csv
import heapq
import math


@dataclass(frozen=True)
class Port:
    name: str
    lat: float
    lon: float
    inr_km: float
    otr_km: float
    inr_deg: Optional[float]
    otr_deg: Optional[float]
    turn_dir: str
    links: Tuple[str, ...]


@dataclass(frozen=True)
class Waypoint:
    name: str
    lat: float
    lon: float
    alt_ft: Optional[float]
    links: Tuple[str, ...]


@dataclass(frozen=True)
class RouteResult:
    path: List[str]
    distance_km: float
    points: List[Tuple[float, float]]  # (lon, lat)


class LocalProjection:
    def __init__(self, lon0: float, lat0: float) -> None:
        self.lon0 = lon0
        self.lat0 = lat0
        self._km_per_deg_lat = 111.32
        self._km_per_deg_lon = 111.32 * math.cos(math.radians(lat0))

    def to_xy_km(self, lon: float, lat: float) -> Tuple[float, float]:
        dx = (lon - self.lon0) * self._km_per_deg_lon
        dy = (lat - self.lat0) * self._km_per_deg_lat
        return dx, dy

    def to_lonlat(self, x_km: float, y_km: float) -> Tuple[float, float]:
        lon = self.lon0 + (x_km / self._km_per_deg_lon)
        lat = self.lat0 + (y_km / self._km_per_deg_lat)
        return lon, lat


class RoutePlanner:
    def __init__(self, ports: Dict[str, Port], waypoints: Dict[str, Waypoint]) -> None:
        if not ports or not waypoints:
            raise ValueError("RoutePlanner requires ports and waypoints.")
        self.ports = ports
        self.waypoints = waypoints

        all_lons = [p.lon for p in ports.values()] + [w.lon for w in waypoints.values()]
        all_lats = [p.lat for p in ports.values()] + [w.lat for w in waypoints.values()]
        lon0 = sum(all_lons) / len(all_lons)
        lat0 = sum(all_lats) / len(all_lats)
        self.projection = LocalProjection(lon0, lat0)

        self.node_xy: Dict[str, Tuple[float, float]] = {}
        for port in ports.values():
            self.node_xy[port.name] = self.projection.to_xy_km(port.lon, port.lat)
        for wp in waypoints.values():
            self.node_xy[wp.name] = self.projection.to_xy_km(wp.lon, wp.lat)

        self.wp_graph: Dict[str, List[Tuple[str, float]]] = {n: [] for n in waypoints}
        self._build_waypoint_graph()
        self.port_links = {p.name: self._filter_links(p.links) for p in ports.values()}
        self.waypoint_to_ports = self._build_waypoint_to_ports()
        self.graph: Dict[str, List[Tuple[str, float]]] = {n: [] for n in self.node_xy}
        self._build_full_graph()

    @classmethod
    def from_csv(
        cls,
        vertiport_csv: Path | str,
        waypoint_csv: Path | str,
        *,
        encoding: Optional[str] = None,
    ) -> "RoutePlanner":
        ports = _load_ports(Path(vertiport_csv), encoding=encoding)
        waypoints = _load_waypoints(Path(waypoint_csv), encoding=encoding)
        return cls(ports, waypoints)

    def list_ports(self) -> List[str]:
        return sorted(self.ports.keys())

    def list_waypoints(self) -> List[str]:
        return sorted(self.waypoints.keys())

    def find_route(
        self,
        start_port: str,
        end_port: str,
        *,
        include_turn_arcs: bool = False,
        arc_step_deg: int = 5,
    ) -> RouteResult:
        path, distance_km = self._shortest_path(start_port, end_port)
        points = self._build_geometry(
            path,
            include_turn_arcs=include_turn_arcs,
            arc_step_deg=arc_step_deg,
        )
        return RouteResult(path=path, distance_km=distance_km, points=points)

    def find_route_via(
        self,
        start_port: str,
        end_port: str,
        via_nodes: Iterable[str],
        *,
        include_turn_arcs: bool = False,
        arc_step_deg: int = 5,
    ) -> RouteResult:
        sequence = [start_port] + list(via_nodes) + [end_port]
        full_path: List[str] = []
        total_dist = 0.0
        for a, b in zip(sequence, sequence[1:]):
            seg_path, seg_dist = self._shortest_path(a, b)
            if full_path:
                full_path.extend(seg_path[1:])
            else:
                full_path.extend(seg_path)
            total_dist += seg_dist
        points = self._build_geometry(
            full_path,
            include_turn_arcs=include_turn_arcs,
            arc_step_deg=arc_step_deg,
        )
        return RouteResult(path=full_path, distance_km=total_dist, points=points)

    def _build_waypoint_graph(self) -> None:
        for wp in self.waypoints.values():
            for link in wp.links:
                if link in self.waypoints:
                    self._add_edge(wp.name, link)

    def _add_edge(self, a: str, b: str) -> None:
        if a == b:
            return
        dist = self._distance_km(a, b)
        self.wp_graph[a].append((b, dist))
        self.wp_graph[b].append((a, dist))

    def _filter_links(self, links: Iterable[str]) -> List[str]:
        return [name for name in links if name in self.waypoints]

    def _build_waypoint_to_ports(self) -> Dict[str, List[str]]:
        mapping: Dict[str, List[str]] = {n: [] for n in self.waypoints}
        for port in self.ports.values():
            for link in self.port_links.get(port.name, []):
                mapping[link].append(port.name)
        return mapping

    def _build_full_graph(self) -> None:
        for wp, neighbors in self.wp_graph.items():
            for nb, _dist in neighbors:
                self._add_graph_edge(wp, nb)
        for port, links in self.port_links.items():
            for wp in links:
                self._add_graph_edge(port, wp)

    def _add_graph_edge(self, a: str, b: str) -> None:
        if a == b:
            return
        dist = self._distance_km(a, b)
        if not any(n == b for n, _ in self.graph[a]):
            self.graph[a].append((b, dist))
        if not any(n == a for n, _ in self.graph[b]):
            self.graph[b].append((a, dist))

    def _distance_km(self, a: str, b: str) -> float:
        ax, ay = self.node_xy[a]
        bx, by = self.node_xy[b]
        return math.hypot(bx - ax, by - ay)

    def _shortest_path(self, start_port: str, end_port: str) -> Tuple[List[str], float]:
        if start_port not in self.graph:
            raise ValueError(f"Unknown start node: {start_port}")
        if end_port not in self.graph:
            raise ValueError(f"Unknown end node: {end_port}")
        if start_port == end_port:
            return [start_port], 0.0

        dist: Dict[str, float] = {start_port: 0.0}
        prev: Dict[str, Optional[str]] = {start_port: None}
        queue: List[Tuple[float, str]] = [(0.0, start_port)]

        while queue:
            cost, node = heapq.heappop(queue)
            if node == end_port:
                break
            if cost != dist.get(node, math.inf):
                continue
            for nxt, weight in self.graph.get(node, []):
                ncost = cost + weight
                if ncost < dist.get(nxt, math.inf):
                    dist[nxt] = ncost
                    prev[nxt] = node
                    heapq.heappush(queue, (ncost, nxt))

        if end_port not in dist:
            raise RuntimeError(f"No route found between {start_port} and {end_port}.")

        path: List[str] = []
        cur: Optional[str] = end_port
        while cur is not None:
            path.append(cur)
            cur = prev.get(cur)
        path.reverse()
        return path, dist[end_port]

    def _build_geometry(
        self,
        path: List[str],
        *,
        include_turn_arcs: bool,
        arc_step_deg: int,
    ) -> List[Tuple[float, float]]:
        if not path:
            return []
        if not include_turn_arcs or len(path) == 1:
            return [self._node_lonlat(name) for name in path]

        start_name = path[0]
        end_name = path[-1]
        start_xy = self.node_xy[start_name]
        end_xy = self.node_xy[end_name]

        points_xy: List[Tuple[float, float]] = [start_xy]

        next_name = path[1] if len(path) > 1 else None
        if next_name:
            points_xy.extend(self._departure_arc_points(start_name, next_name, arc_step_deg))

        for name in path[1:-1]:
            points_xy.append(self.node_xy[name])

        prev_name = path[-2] if len(path) > 1 else None
        if prev_name:
            points_xy.extend(self._arrival_arc_points(prev_name, end_name, arc_step_deg))

        points_xy.append(end_xy)
        return [self.projection.to_lonlat(x, y) for x, y in points_xy]

    def _departure_arc_points(
        self,
        start_port: str,
        next_node: str,
        step_deg: int,
    ) -> List[Tuple[float, float]]:
        port = self.ports[start_port]
        radius = port.otr_km if port.otr_km > 0 else port.inr_km
        if radius <= 0:
            return []
        start_xy = self.node_xy[start_port]
        next_xy = self.node_xy[next_node]
        target_bearing = _bearing_deg(start_xy, next_xy)
        outbound_bearing = port.otr_deg if port.otr_deg is not None else target_bearing
        return _arc_points_xy(
            start_xy,
            radius,
            outbound_bearing,
            target_bearing,
            port.turn_dir,
            step_deg,
        )

    def _arrival_arc_points(
        self,
        prev_node: str,
        end_port: str,
        step_deg: int,
    ) -> List[Tuple[float, float]]:
        port = self.ports[end_port]
        radius = port.inr_km if port.inr_km > 0 else port.otr_km
        if radius <= 0:
            return []
        end_xy = self.node_xy[end_port]
        prev_xy = self.node_xy[prev_node]
        entry_bearing = _bearing_deg(end_xy, prev_xy)
        inbound_bearing = port.inr_deg if port.inr_deg is not None else _bearing_deg(prev_xy, end_xy)
        return _arc_points_xy(
            end_xy,
            radius,
            entry_bearing,
            inbound_bearing,
            port.turn_dir,
            step_deg,
        )

    def _node_lonlat(self, name: str) -> Tuple[float, float]:
        if name in self.ports:
            port = self.ports[name]
            return port.lon, port.lat
        wp = self.waypoints[name]
        return wp.lon, wp.lat


def _arc_points_xy(
    center_xy: Tuple[float, float],
    radius_km: float,
    start_bearing: float,
    end_bearing: float,
    turn_dir: str,
    step_deg: int,
) -> List[Tuple[float, float]]:
    if radius_km <= 0:
        return []
    start_angle = math.radians(90.0 - start_bearing)
    end_angle = math.radians(90.0 - end_bearing)

    direction = (turn_dir or "R").upper()
    step_rad = math.radians(abs(step_deg))
    angles: List[float] = []

    if direction == "R":
        if end_angle > start_angle:
            end_angle -= 2 * math.pi
        angles.append(start_angle)
        angle = start_angle - step_rad
        while angle > end_angle:
            angles.append(angle)
            angle -= step_rad
        angles.append(end_angle)
    else:
        if end_angle < start_angle:
            end_angle += 2 * math.pi
        angles.append(start_angle)
        angle = start_angle + step_rad
        while angle < end_angle:
            angles.append(angle)
            angle += step_rad
        angles.append(end_angle)

    cx, cy = center_xy
    points = [(cx + radius_km * math.cos(a), cy + radius_km * math.sin(a)) for a in angles]
    return points


def _bearing_deg(a_xy: Tuple[float, float], b_xy: Tuple[float, float]) -> float:
    ax, ay = a_xy
    bx, by = b_xy
    dx = bx - ax
    dy = by - ay
    if dx == 0 and dy == 0:
        return 0.0
    return (math.degrees(math.atan2(dx, dy)) + 360.0) % 360.0


def _load_ports(path: Path, encoding: Optional[str] = None) -> Dict[str, Port]:
    rows, fieldnames = _read_csv(path, encoding=encoding)
    name_col = _find_column(fieldnames, lambda k: "vertiport" in k) or fieldnames[0]
    link_col = _find_column(fieldnames, lambda k: "link" in k)

    lat_col = _find_column(fieldnames, lambda k: "lat" in k)
    lon_col = _find_column(fieldnames, lambda k: "lon" in k)
    if not lat_col or not lon_col:
        lat_col, lon_col = _infer_lat_lon_columns(fieldnames, rows)

    inr_deg_col = _find_column(fieldnames, lambda k: "inr" in k and "deg" in k)
    otr_deg_col = _find_column(fieldnames, lambda k: "otr" in k and "deg" in k)
    inr_col = _find_column(fieldnames, lambda k: "inr" in k and "deg" not in k)
    otr_col = _find_column(fieldnames, lambda k: "otr" in k and "deg" not in k)
    turn_col = _find_column(fieldnames, lambda k: "turn" in k or "circle" in k)

    ports: Dict[str, Port] = {}
    for row in rows:
        name = _clean_text(row.get(name_col))
        if not name:
            continue
        lat = _parse_float(row.get(lat_col))
        lon = _parse_float(row.get(lon_col))
        if lat is None or lon is None:
            continue
        inr_km = _parse_float(row.get(inr_col)) or 0.0
        otr_km = _parse_float(row.get(otr_col)) or 0.0
        inr_deg = _parse_float(row.get(inr_deg_col)) if inr_deg_col else None
        otr_deg = _parse_float(row.get(otr_deg_col)) if otr_deg_col else None
        turn_dir = _parse_turn_dir(row.get(turn_col))
        links = tuple(_split_links(row.get(link_col)))
        ports[name] = Port(
            name=name,
            lat=lat,
            lon=lon,
            inr_km=inr_km,
            otr_km=otr_km,
            inr_deg=inr_deg,
            otr_deg=otr_deg,
            turn_dir=turn_dir,
            links=links,
        )
    if not ports:
        raise RuntimeError(f"No vertiports parsed from {path}.")
    return ports


def _load_waypoints(path: Path, encoding: Optional[str] = None) -> Dict[str, Waypoint]:
    rows, fieldnames = _read_csv(path, encoding=encoding)
    name_col = _find_column(fieldnames, lambda k: "waypoint" in k) or fieldnames[0]
    link_col = _find_column(fieldnames, lambda k: "link" in k)

    lat_col = _find_column(fieldnames, lambda k: "lat" in k)
    lon_col = _find_column(fieldnames, lambda k: "lon" in k)
    if not lat_col or not lon_col:
        lat_col, lon_col = _infer_lat_lon_columns(fieldnames, rows)

    alt_col = _find_column(fieldnames, lambda k: "alt" in k or k.endswith("ft"))

    waypoints: Dict[str, Waypoint] = {}
    for row in rows:
        name = _clean_text(row.get(name_col))
        if not name:
            continue
        lat = _parse_float(row.get(lat_col))
        lon = _parse_float(row.get(lon_col))
        if lat is None or lon is None:
            continue
        alt_ft = _parse_float(row.get(alt_col)) if alt_col else None
        links = tuple(_split_links(row.get(link_col)))
        waypoints[name] = Waypoint(
            name=name,
            lat=lat,
            lon=lon,
            alt_ft=alt_ft,
            links=links,
        )
    if not waypoints:
        raise RuntimeError(f"No waypoints parsed from {path}.")
    return waypoints


def _read_csv(path: Path, encoding: Optional[str] = None) -> Tuple[List[Dict[str, str]], List[str]]:
    encodings = [encoding] if encoding else ["utf-8-sig", "utf-8", "cp949", "euc-kr"]
    last_error: Optional[Exception] = None
    for enc in encodings:
        try:
            with path.open("r", encoding=enc, newline="") as handle:
                reader = csv.DictReader(handle)
                rows = list(reader)
            if not reader.fieldnames:
                raise RuntimeError("Missing CSV headers.")
            return rows, list(reader.fieldnames)
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Failed to read CSV: {path}") from last_error


def _find_column(fieldnames: Iterable[str], predicate) -> Optional[str]:
    for name in fieldnames:
        key = _normalize_key(name)
        if predicate(key):
            return name
    return None


def _normalize_key(name: str) -> str:
    return "".join(ch.lower() for ch in name if ch.isalnum())


def _infer_lat_lon_columns(
    fieldnames: List[str],
    rows: List[Dict[str, str]],
) -> Tuple[str, str]:
    stats = _column_stats(fieldnames, rows)
    lat_candidates = []
    lon_candidates = []
    for col, (minv, maxv, meanv) in stats.items():
        if minv >= -90 and maxv <= 90 and 5 <= abs(meanv) <= 60:
            lat_candidates.append((abs(meanv), col))
        if minv >= -180 and maxv <= 180 and abs(meanv) >= 60:
            lon_candidates.append((abs(meanv), col))
    lat_col = max(lat_candidates)[1] if lat_candidates else None
    lon_col = max(lon_candidates)[1] if lon_candidates else None
    if not lat_col or not lon_col:
        raise RuntimeError("Failed to infer lat/lon columns.")
    return lat_col, lon_col


def _column_stats(
    fieldnames: Iterable[str],
    rows: List[Dict[str, str]],
    sample_size: int = 50,
) -> Dict[str, Tuple[float, float, float]]:
    stats: Dict[str, Tuple[float, float, float]] = {}
    for col in fieldnames:
        values: List[float] = []
        for row in rows[:sample_size]:
            value = _parse_float(row.get(col))
            if value is not None:
                values.append(value)
        if len(values) < 3:
            continue
        stats[col] = (min(values), max(values), sum(values) / len(values))
    return stats


def _split_links(value: Optional[str]) -> List[str]:
    if not value:
        return []
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _clean_text(value: Optional[str]) -> str:
    return str(value).strip() if value is not None else ""


def _parse_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def _parse_turn_dir(value: Optional[str]) -> str:
    text = _clean_text(value).lower()
    if text.startswith("l"):
        return "L"
    if text.startswith("r"):
        return "R"
    return "R"


def _configure_matplotlib() -> None:
    import matplotlib as mpl
    from matplotlib import font_manager

    mpl.rcParams["axes.unicode_minus"] = False
    preferred = ["Malgun Gothic", "NanumGothic", "AppleGothic", "DejaVu Sans"]
    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in preferred:
        if name in available:
            mpl.rcParams["font.family"] = name
            break


def _demo_main() -> int:
    import argparse

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print("Matplotlib is required for the demo. Install it with: pip install matplotlib")
        print(f"Import error: {exc}")
        return 1

    _configure_matplotlib()

    parser = argparse.ArgumentParser(description="Route planner demo.")
    parser.add_argument(
        "--vertiport",
        default=None,
        help="Path to vertiport CSV (default: data/vertiport_default.csv)",
    )
    parser.add_argument(
        "--waypoint",
        default=None,
        help="Path to waypoint CSV (default: data/waypoint_default.csv)",
    )
    parser.add_argument(
        "--arcs",
        action="store_true",
        help="Include INR/OTR turn arcs in the route geometry.",
    )
    parser.add_argument(
        "--threshold-km",
        type=float,
        default=0.8,
        help="Click selection threshold in km.",
    )
    parser.add_argument(
        "--link-threshold-km",
        type=float,
        default=0.3,
        help="Click selection threshold for links in km.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    vp_path = Path(args.vertiport) if args.vertiport else root / "data" / "vertiport_default.csv"
    wp_path = Path(args.waypoint) if args.waypoint else root / "data" / "waypoint_default.csv"

    planner = RoutePlanner.from_csv(vp_path, wp_path)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)
    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")
    ax.set_title("Route planner demo: click start port then end port")

    ports_xy = {name: planner.node_xy[name] for name in planner.ports}
    wps_xy = {name: planner.node_xy[name] for name in planner.waypoints}

    pxs, pys = zip(*ports_xy.values())
    wxs, wys = zip(*wps_xy.values())

    ax.scatter(wxs, wys, s=16, c="tab:gray", alpha=0.5, label="waypoint")
    ax.scatter(pxs, pys, s=60, c="tab:blue", label="vertiport")

    wp_edges = set()
    for a, neighbors in planner.wp_graph.items():
        for b, _dist in neighbors:
            key = tuple(sorted((a, b)))
            if key in wp_edges:
                continue
            wp_edges.add(key)
            x1, y1 = wps_xy[a]
            x2, y2 = wps_xy[b]
            ax.plot([x1, x2], [y1, y2], color="tab:orange", linewidth=0.6, alpha=0.5, zorder=1)

    port_edges = set()
    for port, links in planner.port_links.items():
        for wp in links:
            if wp not in wps_xy:
                continue
            key = (port, wp)
            if key in port_edges:
                continue
            port_edges.add(key)
            x1, y1 = ports_xy[port]
            x2, y2 = wps_xy[wp]
            ax.plot([x1, x2], [y1, y2], color="tab:blue", linewidth=0.8, alpha=0.4, zorder=1)

    for name, (x, y) in ports_xy.items():
        ax.text(x, y, f" {name}", fontsize=8, va="bottom", color="tab:blue")

    ax.legend(loc="upper right")

    selection: List[str] = []
    selection_artists: List[object] = []
    route_artists: List[object] = []
    active_edge_artists: List[object] = []
    manual_preview_artists: List[object] = []
    via_artists: Dict[str, List[object]] = {}
    active_edges: List[Tuple[str, str]] = []
    mode: Optional[str] = None
    manual_path: List[str] = []
    manual_prev: Optional[str] = None
    manual_current: Optional[str] = None
    via_nodes: List[str] = []

    def clear_artists(artists: List[object]) -> None:
        for artist in artists:
            artist.remove()
        artists.clear()

    def clear_route() -> None:
        clear_artists(route_artists)

    def clear_selection() -> None:
        clear_artists(selection_artists)

    def clear_active_edges() -> None:
        clear_artists(active_edge_artists)
        active_edges.clear()

    def clear_manual_preview() -> None:
        clear_artists(manual_preview_artists)

    def clear_via_markers() -> None:
        for arts in via_artists.values():
            for art in arts:
                art.remove()
        via_artists.clear()

    def reset_state() -> None:
        nonlocal mode, manual_path, manual_prev, manual_current, via_nodes
        clear_route()
        clear_selection()
        clear_active_edges()
        clear_manual_preview()
        clear_via_markers()
        selection.clear()
        via_nodes = []
        manual_path = []
        manual_prev = None
        manual_current = None
        mode = None
        ax.set_title("Route planner demo: click start port then end port")

    def mark_port(name: str, color: str, label: str) -> None:
        x, y = ports_xy[name]
        marker, = ax.plot(x, y, "o", markersize=12, color=color, markeredgecolor="black", zorder=5)
        text = ax.text(x, y, label, color="white", ha="center", va="center", fontsize=8, zorder=6)
        selection_artists.extend([marker, text])

    def nearest_port(x: float, y: float, threshold: float) -> Optional[str]:
        best = None
        best_dist = threshold
        for name, (px, py) in ports_xy.items():
            d = math.hypot(px - x, py - y)
            if d <= best_dist:
                best = name
                best_dist = d
        return best

    def nearest_waypoint(x: float, y: float, threshold: float) -> Optional[str]:
        best = None
        best_dist = threshold
        for name, (wx, wy) in wps_xy.items():
            d = math.hypot(wx - x, wy - y)
            if d <= best_dist:
                best = name
                best_dist = d
        return best

    def point_segment_distance(
        px: float, py: float, axp: float, ayp: float, bxp: float, byp: float
    ) -> float:
        dx = bxp - axp
        dy = byp - ayp
        if dx == 0 and dy == 0:
            return math.hypot(px - axp, py - ayp)
        t = ((px - axp) * dx + (py - ayp) * dy) / (dx * dx + dy * dy)
        t = max(0.0, min(1.0, t))
        proj_x = axp + t * dx
        proj_y = ayp + t * dy
        return math.hypot(px - proj_x, py - proj_y)

    def pick_active_edge(x: float, y: float) -> Optional[Tuple[str, str]]:
        best = None
        best_dist = args.link_threshold_km
        for a, b in active_edges:
            axp, ayp = planner.node_xy[a]
            bxp, byp = planner.node_xy[b]
            d = point_segment_distance(x, y, axp, ayp, bxp, byp)
            if d <= best_dist:
                best = (a, b)
                best_dist = d
        return best

    def set_active_edges(current: str, prev: Optional[str]) -> None:
        clear_active_edges()
        edges: List[Tuple[str, str]] = []
        for nb, _dist in planner.graph.get(current, []):
            if prev and nb == prev:
                continue
            edges.append((current, nb))
        for a, b in edges:
            axp, ayp = planner.node_xy[a]
            bxp, byp = planner.node_xy[b]
            line, = ax.plot([axp, bxp], [ayp, byp], color="lime", linewidth=2, alpha=0.7, zorder=2)
            active_edge_artists.append(line)
        active_edges.extend(edges)

    def update_manual_preview() -> None:
        clear_manual_preview()
        if len(manual_path) < 2:
            return
        coords = [planner.node_xy[n] for n in manual_path]
        xs, ys = zip(*coords)
        line, = ax.plot(xs, ys, "--", color="lime", linewidth=1.6, zorder=3)
        manual_preview_artists.append(line)

    def add_via_marker(name: str) -> None:
        if name in via_artists:
            return
        x, y = wps_xy[name]
        marker, = ax.plot(x, y, "*", color="purple", markersize=11, zorder=5)
        label = ax.text(x, y, "V", color="white", ha="center", va="center", fontsize=7, zorder=6)
        via_artists[name] = [marker, label]

    def draw_route_result(result: RouteResult) -> None:
        clear_route()
        pts_xy = [planner.projection.to_xy_km(lon, lat) for lon, lat in result.points]
        xs, ys = zip(*pts_xy)
        line, = ax.plot(xs, ys, "-", color="magenta", linewidth=2.2, zorder=4)
        node_xy = [planner.node_xy[name] for name in result.path if name in planner.node_xy]
        node_xs, node_ys = zip(*node_xy)
        nodes, = ax.plot(node_xs, node_ys, "o", color="magenta", markersize=4, zorder=4)
        route_artists.extend([line, nodes])
        ax.set_title(f"Route: {result.path[0]} -> {result.path[-1]} | {result.distance_km:.2f} km")
        print(f"Route {result.path[0]} -> {result.path[-1]}: {result.path} ({result.distance_km:.2f} km)")

    def handle_click(event) -> None:
        nonlocal mode, manual_prev, manual_current
        if event.inaxes is not ax or event.xdata is None or event.ydata is None:
            return
        port_pick = nearest_port(event.xdata, event.ydata, args.threshold_km)
        if len(selection) < 2:
            if not port_pick:
                return
            if not selection:
                selection.append(port_pick)
                mark_port(port_pick, "tab:green", "S")
                ax.set_title(f"Start selected: {port_pick}. Click end port.")
                fig.canvas.draw_idle()
                return
            if port_pick == selection[0]:
                return
            selection.append(port_pick)
            mark_port(port_pick, "tab:red", "E")
            mode = None
            manual_path.clear()
            via_nodes.clear()
            clear_route()
            clear_manual_preview()
            clear_via_markers()
            set_active_edges(selection[0], None)
            ax.set_title(
                f"S={selection[0]} E={selection[1]} | click link for manual mode or waypoint for via mode. Press Enter."
            )
            fig.canvas.draw_idle()
            return

        if port_pick:
            reset_state()
            selection.append(port_pick)
            mark_port(port_pick, "tab:green", "S")
            ax.set_title(f"Start selected: {port_pick}. Click end port.")
            fig.canvas.draw_idle()
            return

        if mode is None:
            wp_pick = nearest_waypoint(event.xdata, event.ydata, args.threshold_km)
            edge_pick = pick_active_edge(event.xdata, event.ydata)
            if wp_pick:
                mode = "via"
                via_nodes.append(wp_pick)
                clear_active_edges()
                add_via_marker(wp_pick)
                ax.set_title("Via mode: select waypoints, then press Enter.")
                fig.canvas.draw_idle()
                return
            if edge_pick:
                mode = "manual"
                manual_path[:] = [edge_pick[0], edge_pick[1]]
                manual_prev = edge_pick[0]
                manual_current = edge_pick[1]
                update_manual_preview()
                if manual_current == selection[1]:
                    clear_active_edges()
                    ax.set_title("Manual mode: destination reached. Press Enter to finalize.")
                else:
                    set_active_edges(manual_current, manual_prev)
                    ax.set_title(f"Manual mode: choose next link from {manual_current}. Press Enter when done.")
                fig.canvas.draw_idle()
                return
            return

        if mode == "manual":
            edge_pick = pick_active_edge(event.xdata, event.ydata)
            if not edge_pick:
                return
            next_node = edge_pick[1]
            if manual_current is None:
                return
            manual_path.append(next_node)
            manual_prev = manual_current
            manual_current = next_node
            update_manual_preview()
            if manual_current == selection[1]:
                clear_active_edges()
                ax.set_title("Manual mode: destination reached. Press Enter to finalize.")
            else:
                set_active_edges(manual_current, manual_prev)
                ax.set_title(f"Manual mode: choose next link from {manual_current}.")
            fig.canvas.draw_idle()
            return

        if mode == "via":
            wp_pick = nearest_waypoint(event.xdata, event.ydata, args.threshold_km)
            if not wp_pick:
                return
            if wp_pick in via_nodes:
                return
            via_nodes.append(wp_pick)
            add_via_marker(wp_pick)
            ax.set_title("Via mode: select waypoints, then press Enter.")
            fig.canvas.draw_idle()
            return
        fig.canvas.draw_idle()

    def handle_key(event) -> None:
        if event.key not in ("enter", "return"):
            return
        if len(selection) < 2:
            return
        try:
            if mode is None:
                result = planner.find_route(selection[0], selection[1], include_turn_arcs=args.arcs)
                draw_route_result(result)
            elif mode == "manual":
                if not manual_path:
                    result = planner.find_route(selection[0], selection[1], include_turn_arcs=args.arcs)
                    draw_route_result(result)
                else:
                    dist = sum(
                        planner._distance_km(a, b) for a, b in zip(manual_path, manual_path[1:])
                    )
                    combined_path = list(manual_path)
                    if manual_path[-1] != selection[1]:
                        tail_path, tail_dist = planner._shortest_path(manual_path[-1], selection[1])
                        combined_path.extend(tail_path[1:])
                        dist += tail_dist
                    points = planner._build_geometry(
                        combined_path,
                        include_turn_arcs=args.arcs,
                        arc_step_deg=5,
                    )
                    draw_route_result(RouteResult(path=combined_path, distance_km=dist, points=points))
            elif mode == "via":
                result = planner.find_route_via(
                    selection[0], selection[1], via_nodes, include_turn_arcs=args.arcs
                )
                draw_route_result(result)
            clear_active_edges()
            clear_manual_preview()
            fig.canvas.draw_idle()
        except Exception as exc:
            ax.set_title(f"Route error: {exc}")
            print(f"Route error: {exc}")
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect("button_press_event", handle_click)
    fig.canvas.mpl_connect("key_press_event", handle_key)
    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(_demo_main())
