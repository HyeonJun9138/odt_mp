from __future__ import annotations

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
RESOURCES_DIR = ROOT_DIR / "resources"
MBTILES_PATH = RESOURCES_DIR / "korea.mbtiles"
DEM_DIR = RESOURCES_DIR / "dem"
DEM_TILE_SIZE = 256
DEM_MAX_ZOOM = 12

WEB_DIR = ROOT_DIR / "app" / "web"

SERVER_HOST = "127.0.0.1"
SERVER_PORT = 0

APP_TITLE = "Korea MBTiles Viewer"

DEFAULT_CENTER_LAT = 37.5665
DEFAULT_CENTER_LON = 126.978
DEFAULT_START_ZOOM = 11.5
USE_BOUNDS = False
