from __future__ import annotations

from pathlib import Path
import json

DEFAULT_SETTINGS_PATH = Path.home() / "Documents" / "AirSim" / "settings.json"

CANONICAL_CAMERAS = [
    {"name": "front", "label": "Front", "default": None, "request_key": "front"},
    {"name": "back", "label": "Back", "default": None, "request_key": "back"},
    {"name": "upper", "label": "Upper", "default": None, "request_key": "upper"},
    {"name": "lower", "label": "Lower", "default": None, "request_key": "lower"},
]

CANONICAL_CAMERA_ORDER = [camera["name"] for camera in CANONICAL_CAMERAS]
CANONICAL_CAMERA_MAP = {camera["name"]: camera for camera in CANONICAL_CAMERAS}

CAMERA_KEY_ALIASES = {
    "backward": "back",
    "rear": "back",
    "rearward": "back",
    "upperm": "upper",
}

__all__ = [
    "DEFAULT_SETTINGS_PATH",
    "CANONICAL_CAMERAS",
    "CANONICAL_CAMERA_ORDER",
    "CANONICAL_CAMERA_MAP",
    "CAMERA_KEY_ALIASES",
    "normalize_camera_key",
    "format_camera_label",
    "load_camera_slots_from_settings",
]


def normalize_camera_key(name: str | None) -> str:
    """Return a normalized camera identifier used throughout the viewport stack."""
    if not name:
        return ""
    raw = str(name).strip()
    canonical = raw.replace("-", "_").replace(" ", "_").lower()
    return CAMERA_KEY_ALIASES.get(canonical, canonical)


def format_camera_label(name: str) -> str:
    value = name.replace("_", " ").replace("-", " ").strip()
    return value.title() if value else name


def load_camera_slots_from_settings(settings_path: Path | None = None) -> list[dict]:
    path = settings_path or DEFAULT_SETTINGS_PATH
    slots_by_name: dict[str, dict] = {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return []
    except Exception:
        return []
    vehicles = data.get("Vehicles", {}) if isinstance(data, dict) else {}
    for vehicle in vehicles.values():
        cameras = vehicle.get("Cameras", {}) if isinstance(vehicle, dict) else {}
        for cam_key, cam_data in cameras.items():
            canonical = normalize_camera_key(cam_key)
            if not canonical or canonical not in CANONICAL_CAMERA_MAP:
                continue
            if canonical in slots_by_name:
                continue
            label = None
            default_pose = None
            request_key = None
            if isinstance(cam_data, dict):
                label = cam_data.get("Label") or CANONICAL_CAMERA_MAP[canonical]["label"]
                try:
                    position = (
                        float(cam_data.get("X", 0.0)),
                        float(cam_data.get("Y", 0.0)),
                        float(cam_data.get("Z", 0.0)),
                    )
                    rotation_deg = (
                        float(cam_data.get("Pitch", 0.0)),
                        float(cam_data.get("Roll", 0.0)),
                        float(cam_data.get("Yaw", 0.0)),
                    )
                    default_pose = {
                        "position": position,
                        "rotation_deg": rotation_deg,
                    }
                except Exception:
                    default_pose = None
                request_key = cam_data.get("RequestKey") or cam_key
            else:
                label = CANONICAL_CAMERA_MAP[canonical]["label"]
                request_key = cam_key
            slots_by_name[canonical] = {
                "name": canonical,
                "label": label,
                "default": default_pose,
                "request_key": str(request_key).strip() if request_key else canonical,
            }
    return [slots_by_name[name] for name in CANONICAL_CAMERA_ORDER if name in slots_by_name]
