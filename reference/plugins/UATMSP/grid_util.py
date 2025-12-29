# -*- coding: utf-8 -*-
from typing import Tuple, List
from PyQt5.QtWidgets import QTableWidget, QWidget

ROWS = 20
COLS = 30

def n_to_rc(n: int) -> Tuple[int, int]:
    """1-base 번호를 0-base (row, col)로 변환"""
    n0 = n - 1
    return n0 // COLS, n0 % COLS

def rect_from_ranges(ranges: List[Tuple[int, int]]) -> Tuple[int, int, int, int]:
    """
    같은 열대역을 가진 여러 줄 구간(예: 32~49, 62~79, 92~109)을
    하나의 직사각형 스팬으로 환산.
    return: (row, col, row_span, col_span) - 모두 0-base 기준
    """
    # 첫 구간의 시작 → 좌상
    rs, re = ranges[0]
    r0, c0 = n_to_rc(rs)
    # 마지막 구간의 끝 → 우하
    rs2, re2 = ranges[-1]
    r1, c1 = n_to_rc(re2)

    row_span = r1 - r0 + 1
    col_span = c1 - c0 + 1
    return r0, c0, row_span, col_span

def put_spanned_widget(table: QTableWidget, widget: QWidget,
                       ranges: List[Tuple[int, int]]) -> None:
    """구간 리스트를 하나의 span으로 묶어 위젯 배치"""
    r0, c0, rs, cs = rect_from_ranges(ranges)
    table.setSpan(r0, c0, rs, cs)
    table.setCellWidget(r0, c0, widget)
