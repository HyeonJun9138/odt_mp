from __future__ import annotations

from pathlib import Path
from typing import Optional, Callable

from .dashboard import ViewportDashboard

__all__ = ["ViewportDashboard", "create_viewport_dashboard"]


def create_viewport_dashboard(
    settings_path: Optional[Path] = None,
    host: Optional[str] = None,
    port: Optional[int] = None,
    plugin_port: Optional[int] = None,
    plugin_id: Optional[int] = None,
    apply_callback: Optional[Callable[[dict], None]] = None,
    close_callback: Optional[Callable[[Optional[int], bool], None]] = None,
    initial_config: Optional[dict] = None,
) -> ViewportDashboard:
    """Instantiate a viewport dashboard window with optional connection defaults."""
    return ViewportDashboard(
        settings_path=settings_path,
        airsim_host=host,
        airsim_port=port,
        plugin_port=plugin_port,
        plugin_id=plugin_id,
        apply_callback=apply_callback,
        close_callback=close_callback,
        initial_config=initial_config,
    )
