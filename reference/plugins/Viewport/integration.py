from __future__ import annotations

try:
    import airsim  # type: ignore
    HAS_AIRSIM = True
except Exception:  # pragma: no cover - optional dependency
    airsim = None
    HAS_AIRSIM = False

__all__ = ["airsim", "HAS_AIRSIM"]
