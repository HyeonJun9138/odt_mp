from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..scaling import UiScale, add_drop_shadow
from .aspect_box import AspectBox


class CameraControlPanel(QFrame):
    """Right-hand panel hosting the live preview and camera controls."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("SideCard")
        add_drop_shadow(self, radius=18, alpha=70)

        root = QHBoxLayout(self)
        root.setContentsMargins(UiScale.dp(20), UiScale.dp(20), UiScale.dp(20), UiScale.dp(20))
        root.setSpacing(UiScale.dp(18))

        content_col = QVBoxLayout()
        content_col.setContentsMargins(0, 0, 0, 0)
        content_col.setSpacing(UiScale.dp(12))

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(UiScale.dp(10))
        title = QLabel("Live Camera Feed", self)
        title.setObjectName("SectionLabel")
        self.camera_badge = QLabel("FRONT", self)
        self.camera_badge.setObjectName("Badge")
        header.addWidget(title)
        header.addStretch(1)
        header.addWidget(self.camera_badge)
        content_col.addLayout(header)

        self.preview = AspectBox(self, 16, 9, min_w=220, max_w=None)
        self.preview.setMinimumHeight(UiScale.dp(200))
        self.preview.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        content_col.addWidget(self.preview, 1)

        self.meta = QLabel("Selected camera preview", self)
        self.meta.setObjectName("TileCaption")
        self.meta.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.meta.setWordWrap(True)
        self.meta.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        content_col.addWidget(self.meta)

        root.addLayout(content_col, 3)

        controls_host = QWidget(self)
        controls_host.setObjectName("ControlStrip")
        controls_host.setMinimumWidth(UiScale.dp(200))
        controls_host.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        controls_layout = QVBoxLayout(controls_host)
        controls_layout.setContentsMargins(UiScale.dp(12), UiScale.dp(12), UiScale.dp(12), UiScale.dp(12))
        controls_layout.setSpacing(UiScale.dp(14))

        def make_tool_button(label: str) -> QToolButton:
            btn = QToolButton(self)
            btn.setText(label)
            btn.setMinimumSize(UiScale.dp(30), UiScale.dp(22))
            btn.setMaximumWidth(UiScale.dp(80))
            btn.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
            return btn

        def make_command_button(label: str) -> QPushButton:
            btn = QPushButton(label, self)
            btn.setObjectName("BackButton")
            btn.setMinimumHeight(UiScale.dp(26))
            btn.setMaximumWidth(UiScale.dp(140))
            btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
            return btn

        def build_section(title_text: str, buttons: list[tuple[int, int, QWidget]]) -> QWidget:
            box = QWidget(self)
            box.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            layout = QVBoxLayout(box)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(UiScale.dp(6))
            lbl = QLabel(title_text, self)
            lbl.setObjectName("SmallHead")
            lbl.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter)
            layout.addWidget(lbl, 0, Qt.AlignmentFlag.AlignTop)
            grid_host = QWidget(self)
            grid = QGridLayout(grid_host)
            grid.setContentsMargins(0, 0, 0, 0)
            grid.setHorizontalSpacing(UiScale.dp(6))
            grid.setVerticalSpacing(UiScale.dp(6))
            for row, col, widget in buttons:
                grid.addWidget(widget, row, col, alignment=Qt.AlignmentFlag.AlignCenter)
            max_row = max((row for row, _, _ in buttons), default=0)
            for idx in range(max_row + 1):
                grid.setRowStretch(idx, 1)
            for col_idx in range(3):
                grid.setColumnStretch(col_idx, 1)
            layout.addWidget(grid_host, 1)
            return box

        self.btnMoveUp = make_tool_button("Up")
        self.btnMoveDown = make_tool_button("Down")
        self.btnMoveLeft = make_tool_button("Left")
        self.btnMoveRight = make_tool_button("Right")

        move_grid_buttons = [
            (0, 1, self.btnMoveUp),
            (1, 0, self.btnMoveLeft),
            (1, 1, self.btnMoveDown),
            (1, 2, self.btnMoveRight),
        ]
        move_section = build_section("Move", move_grid_buttons)

        self.btnPitchUp = make_tool_button("↻↑")
        self.btnPitchDown = make_tool_button("↺↓")
        self.btnYawLeft = make_tool_button("↺←")
        self.btnYawRight = make_tool_button("↻→")

        rotate_grid_buttons = [
            (0, 1, self.btnPitchUp),
            (1, 0, self.btnYawLeft),
            (1, 1, self.btnPitchDown),
            (1, 2, self.btnYawRight),
        ]
        rotate_section = build_section("Rotate", rotate_grid_buttons)

        controls_layout.addStretch(1)
        controls_layout.addWidget(move_section, 2)
        controls_layout.addStretch(1)
        controls_layout.addWidget(rotate_section, 2)
        controls_layout.addStretch(1)

        self.btnResetCamera = make_command_button("Reset Camera")
        controls_layout.addWidget(self.btnResetCamera, 0, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignBottom)

        root.addWidget(controls_host, 1)

    def set_selected_camera(self, name: str) -> None:
        label = (name or "N/A").upper()
        self.camera_badge.setText(label)
        self.meta.setText(f"{label} feed is active")
