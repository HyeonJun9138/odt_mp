from __future__ import annotations

import sys

from app.config import APP_TITLE, MBTILES_PATH, RESOURCES_DIR, SERVER_HOST, SERVER_PORT, WEB_DIR
from app.mbtiles import MBTiles
from app.tile_server import TileServer
from app.gui.app import run_app


def main() -> int:
    if not MBTILES_PATH.exists():
        print(f"MBTiles not found: {MBTILES_PATH}")
        return 1

    mbtiles = MBTiles(MBTILES_PATH)
    server = TileServer(
        mbtiles=mbtiles,
        web_dir=WEB_DIR,
        resources_dir=RESOURCES_DIR,
        host=SERVER_HOST,
        port=SERVER_PORT,
    )
    server.start()
    try:
        return run_app(server.url, title=APP_TITLE)
    finally:
        server.stop()
        mbtiles.close()


if __name__ == "__main__":
    raise SystemExit(main())
