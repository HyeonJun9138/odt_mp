from __future__ import annotations

from PySide6.QtGui import QColor
from PySide6.QtWidgets import QApplication, QWidget, QGraphicsDropShadowEffect


class UiScale:
    """Utility helpers for density-independent sizing."""

    scale: float = 1.0

    @classmethod
    def init_from_screen(cls, app: QApplication) -> None:
        screen = app.primaryScreen()
        dpi = 96.0 if screen is None else screen.logicalDotsPerInch()
        cls.scale = max(1.0, min(dpi / 96.0, 2.0))

    @classmethod
    def dp(cls, value: int) -> int:
        return int(round(value * cls.scale))


def add_drop_shadow(widget: QWidget, radius: int = 14, alpha: int = 60) -> None:
    """Attach a subtle drop-shadow effect to the provided widget."""
    effect = QGraphicsDropShadowEffect(widget)
    effect.setBlurRadius(UiScale.dp(radius))
    effect.setXOffset(0)
    effect.setYOffset(0)
    effect.setColor(QColor(15, 23, 42, alpha))
    widget.setGraphicsEffect(effect)
