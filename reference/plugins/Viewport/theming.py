from __future__ import annotations

from typing import Protocol

from PySide6.QtWidgets import QApplication, QWidget

from .scaling import UiScale


class _SupportsStyleSheet(Protocol):
    def setStyleSheet(self, style: str) -> None: ...


def apply_common_qss(target: QApplication | QWidget) -> None:
    """Apply the shared style definitions to the given Qt target."""
    accent = "#38bdf8"
    accent_hover = "#0ea5e9"
    text_main = "#e2e8f0"
    card_bg = "#0f172a"
    card_alt = "#1e293b"
    left_pane_bg = "#0f172a"
    right_pane_bg = "#0b1120"
    strip_bg = "#111c32"
    surface_bg = "#0b1120"
    control_bg = "#0f1d33"

    qss = f"""
    QWidget {{
        color: {text_main};
    }}
    QLabel {{
        color: {text_main};
    }}
    QWidget#RootArea {{
        background-color: #050914;
    }}
    QLabel#TitleLabel {{
        font-size: {UiScale.dp(32)}px;
        font-weight: 700;
    }}
    QWidget#PluginCard {{
        background-color: {card_bg};
        border-radius: {UiScale.dp(14)}px;
        border: 1px solid rgba(148, 163, 184, 0.18);
    }}
    QFrame.SideCard {{
        background-color: {card_alt};
    }}
    QFrame#LeftPane {{
        background-color: {left_pane_bg};
        border-radius: {UiScale.dp(20)}px;
        border: 1px solid rgba(148, 163, 184, 0.12);
    }}
    QFrame#RightPane {{
        background-color: {right_pane_bg};
        border-radius: {UiScale.dp(20)}px;
        border: 1px solid rgba(148, 163, 184, 0.12);
    }}
    QFrame#BottomStrip {{
        background-color: {strip_bg};
        border-radius: {UiScale.dp(16)}px;
        border: 1px solid rgba(148, 163, 184, 0.12);
    }}
    QFrame.VideoInner {{
        background-color: {surface_bg};
        border: 2px solid rgba(148, 163, 184, 0.32);
        border-radius: {UiScale.dp(12)}px;
    }}
    QFrame.VideoInner[selected="true"] {{
        border-color: {accent};
        box-shadow: 0 0 {UiScale.dp(14)}px rgba(56, 189, 248, 0.45);
    }}
    QWidget#ControlStrip {{
        background-color: {control_bg};
        border-radius: {UiScale.dp(12)}px;
        padding: {UiScale.dp(14)}px;
    }}
    QSplitter#RootSplit::handle,
    QSplitter#LeftSplit::handle,
    QSplitter#RightSplit::handle {{
        background-color: rgba(148, 163, 184, 0.18);
        border-radius: {UiScale.dp(2)}px;
    }}
    QComboBox,
    QSpinBox,
    QLineEdit {{
        background-color: {surface_bg};
        padding: {UiScale.dp(6)}px {UiScale.dp(10)}px;
        border-radius: {UiScale.dp(8)}px;
        border: 1px solid rgba(148, 163, 184, 0.28);
        color: {text_main};
    }}
    QComboBox:hover,
    QSpinBox:hover,
    QLineEdit:hover {{
        border-color: {accent};
    }}
    QComboBox::drop-down {{
        border: none;
        width: {UiScale.dp(24)}px;
    }}
    QPlainTextEdit#LogEdit {{
        background-color: {surface_bg};
        border-radius: {UiScale.dp(12)}px;
        border: 1px solid rgba(148, 163, 184, 0.22);
        color: {text_main};
    }}
    QPushButton {{
        border-radius: {UiScale.dp(10)}px;
        font-weight: 600;
        padding: {UiScale.dp(8)}px {UiScale.dp(18)}px;
    }}
    QPushButton#AccentButton {{
        background-color: {accent};
        color: #051525;
    }}
    QPushButton#AccentButton:hover {{
        background-color: {accent_hover};
    }}
    QPushButton#BackButton {{
        background-color: transparent;
        border: 1px solid rgba(148, 163, 184, 0.42);
        color: {text_main};
        padding: {UiScale.dp(6)}px {UiScale.dp(14)}px;
        font-size: {UiScale.dp(11)}px;
    }}
    QPushButton#BackButton:hover {{
        border-color: {accent};
        color: {accent};
    }}
    QToolButton {{
        background-color: {surface_bg};
        border: 1px solid rgba(148, 163, 184, 0.28);
        border-radius: {UiScale.dp(8)}px;
        color: {text_main};
        font-size: {UiScale.dp(9)}px;
        padding: {UiScale.dp(4)}px {UiScale.dp(10)}px;
    }}
    QToolButton:hover {{
        border-color: {accent};
        color: {accent};
    }}
    """
    setter: _SupportsStyleSheet = target  # type: ignore[assignment]
    setter.setStyleSheet(qss)
