from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import threading
from typing import Optional

try:
    import numpy as np
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.io import MemoryFile
    from rasterio.merge import merge
    from rasterio.transform import from_bounds
    from rasterio.vrt import WarpedVRT
except Exception:  # pragma: no cover - optional dependency
    rasterio = None


WEB_MERCATOR_HALF = 20037508.342789244


def _tile_bounds_mercator(z: int, x: int, y: int) -> tuple[float, float, float, float]:
    n = 1 << z
    tile_span = (2 * WEB_MERCATOR_HALF) / n
    west = -WEB_MERCATOR_HALF + (x * tile_span)
    east = west + tile_span
    north = WEB_MERCATOR_HALF - (y * tile_span)
    south = north - tile_span
    return (west, south, east, north)


def _encode_terrarium(elevation: "np.ndarray") -> "np.ndarray":
    value = np.clip(elevation + 32768.0, 0, 65535.996)
    r = np.floor(value / 256.0)
    g = np.floor(value - (r * 256.0))
    b = np.floor((value - np.floor(value)) * 256.0)
    rgb = np.stack([r, g, b]).astype("uint8")
    return rgb


@dataclass
class DemTileProvider:
    dem_dir: Path
    tile_size: int = 256
    max_zoom: int = 12

    def __post_init__(self) -> None:
        self._lock = threading.Lock()
        self._datasets = []
        self._vrts = []
        if rasterio is None:
            return
        for tif in sorted(self.dem_dir.glob("*.tif")):
            try:
                dataset = rasterio.open(tif)
            except Exception:
                continue
            vrt = WarpedVRT(dataset, crs="EPSG:3857", resampling=Resampling.bilinear)
            self._datasets.append(dataset)
            self._vrts.append(vrt)

    @property
    def available(self) -> bool:
        return bool(self._vrts)

    def get_tile(self, z: int, x: int, y: int) -> Optional[bytes]:
        if not self.available or z > self.max_zoom:
            return None
        bounds = _tile_bounds_mercator(z, x, y)
        candidates = [vrt for vrt in self._vrts if _bounds_intersect(bounds, vrt.bounds)]
        if not candidates:
            return None
        transform = from_bounds(*bounds, self.tile_size, self.tile_size)
        res = (transform.a, -transform.e)
        with self._lock:
            data, _ = merge(
                candidates,
                bounds=bounds,
                res=res,
                out_shape=(1, self.tile_size, self.tile_size),
                resampling=Resampling.bilinear,
            )
        elevation = data[0].astype("float32")
        elevation = np.where(np.isfinite(elevation), elevation, 0.0)
        rgb = _encode_terrarium(elevation)
        with MemoryFile() as memfile:
            with memfile.open(
                driver="PNG",
                width=self.tile_size,
                height=self.tile_size,
                count=3,
                dtype="uint8",
            ) as dataset:
                dataset.write(rgb)
            return memfile.read()


def _bounds_intersect(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> bool:
    return not (a[2] <= b[0] or a[0] >= b[2] or a[3] <= b[1] or a[1] >= b[3])


def load_dem_provider(dem_dir: Path, tile_size: int = 256, max_zoom: int = 12) -> DemTileProvider:
    provider = DemTileProvider(dem_dir=dem_dir, tile_size=tile_size, max_zoom=max_zoom)
    return provider
