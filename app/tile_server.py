from __future__ import annotations

import json
import mimetypes
import os
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from socketserver import ThreadingMixIn
import threading
from typing import Optional
from urllib.parse import urlparse

from app.config import DEFAULT_CENTER_LAT, DEFAULT_CENTER_LON, DEFAULT_START_ZOOM, USE_BOUNDS
from app.mbtiles import MBTiles


class TileHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self, server_address: tuple[str, int], mbtiles: MBTiles, web_dir: Path, resources_dir: Path
    ):
        super().__init__(server_address, TileRequestHandler)
        self.mbtiles = mbtiles
        self.web_dir = Path(web_dir)
        self.resources_dir = Path(resources_dir)

    def tile_url(self) -> str:
        host, port = self.server_address
        ext = _normalize_format(self.mbtiles.info.tile_format)
        return f"http://{host}:{port}/tiles/{{z}}/{{x}}/{{y}}.{ext}"


class TileServer:
    def __init__(
        self, mbtiles: MBTiles, web_dir: Path, resources_dir: Path, host: str, port: int
    ) -> None:
        self._server = TileHTTPServer((host, port), mbtiles, web_dir, resources_dir)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        host, port = self._server.server_address
        return f"http://{host}:{port}/"

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        if not self._thread.is_alive():
            return
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=2)


class TileRequestHandler(BaseHTTPRequestHandler):
    server: TileHTTPServer

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            self._serve_index()
            return
        if path.startswith("/tiles/"):
            self._serve_tile(path)
            return
        if path.startswith("/resources/"):
            self._serve_resource(path)
            return
        self._serve_static(path)

    def log_message(self, fmt: str, *args: object) -> None:
        return

    def _serve_index(self) -> None:
        template_path = self.server.web_dir / "index.html"
        if not template_path.exists():
            self._send_text(HTTPStatus.NOT_FOUND, "Missing index.html")
            return
        html = template_path.read_text(encoding="utf-8")
        info = self.server.mbtiles.info
        center_lat, center_lon, start_zoom = info.start_view()
        if DEFAULT_CENTER_LAT is not None and DEFAULT_CENTER_LON is not None:
            center_lat = float(DEFAULT_CENTER_LAT)
            center_lon = float(DEFAULT_CENTER_LON)
        if DEFAULT_START_ZOOM is not None:
            start_zoom = float(DEFAULT_START_ZOOM)
        bounds_json = _format_bounds(info.bounds) if USE_BOUNDS else "null"
        html = (
            html.replace("__TILE_URL__", self.server.tile_url())
            .replace("__MIN_ZOOM__", str(info.min_zoom))
            .replace("__MAX_ZOOM__", str(info.max_zoom))
            .replace("__CENTER_LAT__", f"{center_lat:.6f}")
            .replace("__CENTER_LON__", f"{center_lon:.6f}")
            .replace("__START_ZOOM__", str(start_zoom))
            .replace("__BOUNDS_JSON__", bounds_json)
        )
        self._send_bytes(HTTPStatus.OK, html.encode("utf-8"), "text/html; charset=utf-8")

    def _serve_tile(self, path: str) -> None:
        parts = path.split("/")
        if len(parts) < 5:
            self._send_text(HTTPStatus.NOT_FOUND, "Invalid tile path")
            return
        try:
            z = int(parts[2])
            x = int(parts[3])
            y_str, _ext = os.path.splitext(parts[4])
            y = int(y_str)
        except ValueError:
            self._send_text(HTTPStatus.BAD_REQUEST, "Invalid tile coordinates")
            return

        data = self.server.mbtiles.get_tile(z, x, y)
        if data is None:
            self._send_text(HTTPStatus.NOT_FOUND, "Tile not found")
            return

        content_type = _content_type(self.server.mbtiles.info.tile_format)
        encoding = None
        if self.server.mbtiles.info.tile_format == "pbf" and data[:2] == b"\x1f\x8b":
            encoding = "gzip"
        self._send_bytes(HTTPStatus.OK, data, content_type, encoding=encoding)

    def _serve_static(self, path: str) -> None:
        file_path = _safe_join(self.server.web_dir, path)
        if file_path is None or not file_path.is_file():
            self._send_text(HTTPStatus.NOT_FOUND, "Not found")
            return
        mime, _ = mimetypes.guess_type(file_path)
        content_type = mime or "application/octet-stream"
        self._send_bytes(HTTPStatus.OK, file_path.read_bytes(), content_type)

    def _serve_resource(self, path: str) -> None:
        resource_path = path[len("/resources/") :]
        file_path = _safe_join(self.server.resources_dir, resource_path)
        if file_path is None or not file_path.is_file():
            self._send_text(HTTPStatus.NOT_FOUND, "Not found")
            return
        mime, _ = mimetypes.guess_type(file_path)
        content_type = mime or "application/octet-stream"
        self._send_bytes(HTTPStatus.OK, file_path.read_bytes(), content_type)

    def _send_text(self, status: HTTPStatus, message: str) -> None:
        self._send_bytes(status, message.encode("utf-8"), "text/plain; charset=utf-8")

    def _send_bytes(
        self, status: HTTPStatus, data: bytes, content_type: str, encoding: Optional[str] = None
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        if encoding:
            self.send_header("Content-Encoding", encoding)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def _safe_join(base_dir: Path, request_path: str) -> Optional[Path]:
    safe_path = os.path.normpath(request_path).lstrip("/\\")
    full = (base_dir / safe_path).resolve()
    if base_dir != full and base_dir not in full.parents:
        return None
    return full


def _normalize_format(tile_format: str) -> str:
    fmt = (tile_format or "png").lower()
    if fmt == "jpeg":
        return "jpg"
    return fmt


def _content_type(tile_format: str) -> str:
    fmt = _normalize_format(tile_format)
    if fmt in ("jpg", "jpeg"):
        return "image/jpeg"
    if fmt == "png":
        return "image/png"
    if fmt == "webp":
        return "image/webp"
    if fmt == "pbf":
        return "application/vnd.mapbox-vector-tile"
    return "application/octet-stream"


def _format_bounds(bounds: Optional[tuple[float, float, float, float]]) -> str:
    if bounds is None:
        return "null"
    min_lon, min_lat, max_lon, max_lat = bounds
    return json.dumps([[min_lon, min_lat], [max_lon, max_lat]])
