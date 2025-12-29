#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFrame, QPushButton, QListWidget, QFileDialog,
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView, QAbstractScrollArea,
    QMessageBox
)
import csv
try:
    from Analyzer.Functions.core import get_db_dir
except ModuleNotFoundError:
    # main.py를 직접 실행할 때 대비
    import sys, os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from Functions.core import get_db_dir

class UploadPage(QWidget):
    """중앙 영역에서 전환되어 쓰이는 '분석 데이터 등록' 내부 페이지"""
    closed = pyqtSignal()            # 닫기(메인 페이지로 복귀)
    registered = pyqtSignal(list)    # 등록 완료(files)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()
        self._connect_signals()

    # UI
    def _build_ui(self):
        outer = QVBoxLayout(self); outer.setContentsMargins(0, 0, 0, 0); outer.setSpacing(0)

        card = QFrame(); card.setObjectName("RightCard")
        lay = QVBoxLayout(card); lay.setContentsMargins(20, 20, 20, 20); lay.setSpacing(12)

        # 헤더
        hdr = QHBoxLayout()
        hdr_title = QPushButton("분석 데이터 관리")
        hdr_title.setEnabled(False)
        hdr_title.setStyleSheet("QPushButton { color:#ffffff; background:transparent; border:none; font-size:28px; font-weight:800; }")
        hdr.addWidget(hdr_title); hdr.addStretch(1)

        self.btn_upload_close = QPushButton("닫기")
        self.btn_upload_close.setCursor(Qt.PointingHandCursor)
        self.btn_upload_close.setStyleSheet("QPushButton { background:#5a6274; color:#fff; border:none; border-radius:8px; padding:10px 14px; }")
        hdr.addWidget(self.btn_upload_close)
        lay.addLayout(hdr)

        # 컨트롤 버튼들
        ctrl = QHBoxLayout()
        self.btn_add_files    = QPushButton("파일 추가")
        self.btn_remove_file  = QPushButton("선택 삭제")
        self.btn_clear_files  = QPushButton("전부 지우기")
        for b in (self.btn_add_files, self.btn_remove_file, self.btn_clear_files):
            b.setCursor(Qt.PointingHandCursor)
            b.setStyleSheet("QPushButton { background:#586073; color:#fff; border:none; border-radius:8px; padding:10px 14px; }")
            ctrl.addWidget(b)
        ctrl.addStretch(1)
        lay.addLayout(ctrl)

        # 파일 목록
        self.lst_files = QListWidget()
        self.lst_files.setSelectionMode(QListWidget.ExtendedSelection)
        self.lst_files.setMinimumHeight(140)
        lay.addWidget(self.lst_files, 0)

        # 표 미리보기 (좌우 스크롤)
        self.tbl_preview = QTableWidget()
        self.tbl_preview.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tbl_preview.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tbl_preview.setAlternatingRowColors(False)
        self.tbl_preview.setWordWrap(False)
        self.tbl_preview.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.tbl_preview.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.tbl_preview.setHorizontalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.tbl_preview.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.tbl_preview.setSizeAdjustPolicy(QAbstractScrollArea.AdjustIgnored)
        self.tbl_preview.horizontalHeader().setStretchLastSection(False)
        self.tbl_preview.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.tbl_preview.verticalHeader().setVisible(False)
        lay.addWidget(self.tbl_preview, 1)

        # 푸터(등록)
        ftr = QHBoxLayout(); ftr.addStretch(1)
        self.btn_register = QPushButton("등록")
        self.btn_register.setCursor(Qt.PointingHandCursor)
        self.btn_register.setStyleSheet("QPushButton { background:#2e8b3d; color:#fff; border:none; border-radius:10px; padding:12px 22px; font-weight:700; }")
        ftr.addWidget(self.btn_register)
        lay.addLayout(ftr)

        outer.addWidget(card)
        
    def _preview_first_file(self):
        """리스트에 항목이 있으면 첫 항목을 선택하고 미리보기 갱신."""
        if self.lst_files.count() == 0:
            return
        if self.lst_files.currentRow() < 0:
            self.lst_files.setCurrentRow(0)
        self._on_preview_selected()
        
    # 시그널
    def _connect_signals(self):
        self.btn_upload_close.clicked.connect(self._on_close_upload)
        self.btn_add_files.clicked.connect(self._on_add_files)
        self.btn_remove_file.clicked.connect(self._on_remove_selected)
        self.btn_clear_files.clicked.connect(self._on_clear_files)
        self.btn_register.clicked.connect(self._on_register_files)

        # 선택 변경 시 자동 미리보기
        self.lst_files.itemSelectionChanged.connect(self._on_preview_selected)

    # 동작 슬롯들
    def _on_close_upload(self):
        self.closed.emit()

    def _on_add_files(self):
        start_dir = str(get_db_dir())
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "분석 데이터 파일 선택",
            start_dir,
            "엑셀/CSV 파일 (*.xlsx *.xls *.csv *.tsv *.txt);;모든 파일 (*.*)"
        )
        if not files:
            return
        existing = {self.lst_files.item(i).text() for i in range(self.lst_files.count())}
        added_any = False
        for f in files:
            if f not in existing:
                self.lst_files.addItem(f)
                added_any = True
        if added_any:
            self._preview_first_file()

    def _on_remove_selected(self):
        for it in self.lst_files.selectedItems():
            self.lst_files.takeItem(self.lst_files.row(it))

    def _on_clear_files(self):
        self.lst_files.clear()
        self.tbl_preview.clear()
        self.tbl_preview.setRowCount(0)
        self.tbl_preview.setColumnCount(0)

    def _on_preview_selected(self):
        if self.lst_files.count() == 0:
            self.tbl_preview.clear()
            self.tbl_preview.setRowCount(0); self.tbl_preview.setColumnCount(0)
            return
        sel = self.lst_files.selectedItems()
        path = sel[0].text() if sel else self.lst_files.item(0).text()
        self._load_table_preview(path)

    def _on_register_files(self):
        files = [self.lst_files.item(i).text() for i in range(self.lst_files.count())]
        if not files:
            QMessageBox.warning(self, "등록", "먼저 파일을 추가하세요.")
            return
        # TODO: 실제 등록 로직 연결 (DB/디렉토리 복사 등)
        QMessageBox.information(self, "등록", f"{len(files)}개 파일 등록 완료.")
        self.registered.emit(files)
        self.closed.emit()

    # CSV/엑셀 표 미리보기
    def _load_table_preview(self, path: str, max_rows: int = 200, max_cols: int = 256):
        """엑셀(.xlsx/.xls) 또는 CSV/TSV/TXT를 표로 미리보기
           - 세미콜론은 구분자로 취급하지 않음(내부 좌표 보호)
           - 헤더 자동 감지(LocalID/Seg*/ID/Type/Pax/From/STD/To/STA)
        """
        self.tbl_preview.clear()
        self.tbl_preview.setRowCount(0)
        self.tbl_preview.setColumnCount(0)

        ext = Path(path).suffix.lower()
        rows, headers = [], None

        def _pad_to_ncols(data, ncols):
            for r in data:
                if len(r) < ncols:
                    r += [""] * (ncols - len(r))
                elif len(r) > ncols:
                    del r[ncols:]

        try:
            if ext in (".xlsx", ".xls"):
                # 엑셀: 1행을 헤더로 간주
                try:
                    import openpyxl
                    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
                    ws = wb.active
                    tmp = []
                    for i, r in enumerate(ws.iter_rows(values_only=True)):
                        tmp.append([("" if v is None else str(v)) for v in r])
                        if i >= max_rows:
                            break
                    if tmp:
                        ncols = min(max(len(r) for r in tmp), max_cols)
                        _pad_to_ncols(tmp, ncols)
                        headers = tmp[0][:ncols]
                        rows = tmp[1:]
                    else:
                        rows = [["빈 워크시트입니다."]]
                except Exception:
                    # pandas로 폴백
                    try:
                        import pandas as pd  # type: ignore
                        df = pd.read_excel(path, nrows=max_rows).fillna("")
                        headers = [str(c) for c in df.columns][:max_cols]
                        rows = [list(map(str, r))[:max_cols] for r in df.values.tolist()]
                    except Exception as e:
                        rows = [[f"엑셀 미리보기 실패: {e}"]]
            else:
                # 텍스트 계열: ',' / '\t' / '|' 후보 중 최다 기호 사용 (세미콜론 제외)
                for enc in ("utf-8-sig", "utf-8", "cp949", "euc-kr", "latin-1"):
                    try:
                        with open(path, "r", encoding=enc, errors="ignore") as f:
                            first_line = f.readline()
                            candidates = [",", "\t", "|"]
                            delim = max(candidates, key=lambda d: first_line.count(d)) if first_line else ","
                            f.seek(0)

                            reader = csv.reader(f, delimiter=delim)
                            tmp = []
                            for i, row in enumerate(reader):
                                tmp.append([("" if x is None else str(x)) for x in row])
                                if i >= max_rows:
                                    break
                            if not tmp:
                                continue

                            ncols = min(max(len(r) for r in tmp), max_cols)
                            _pad_to_ncols(tmp, ncols)

                            # 헤더 판단
                            first_lower = [c.strip().lower() for c in tmp[0]]
                            if (any(c in ("localid", "id", "type", "pax", "from", "std", "to", "sta") for c in first_lower)
                                or any(c.startswith("seg") for c in first_lower)):
                                headers = tmp[0][:ncols]
                                rows = tmp[1:]
                            else:
                                headers = None
                                rows = tmp
                            break
                    except Exception:
                        continue

            if not rows and not headers:
                rows = [["미리보기를 불러올 수 없습니다."]]

        except Exception as e:
            rows = [[f"미리보기 오류: {e}"]]

        # 최종 테이블 채우기
        ncols = min(max(len(r) for r in rows), max_cols) if rows else (len(headers) if headers else 0)
        if ncols <= 0:
            return

        self.tbl_preview.setColumnCount(ncols)
        if headers and len(headers) >= ncols:
            self.tbl_preview.setHorizontalHeaderLabels(headers[:ncols])
        else:
            self.tbl_preview.setHorizontalHeaderLabels([f"C{i+1}" for i in range(ncols)])

        self.tbl_preview.setRowCount(len(rows))
        for r_idx, r in enumerate(rows):
            for c_idx in range(ncols):
                val = r[c_idx] if c_idx < len(r) else ""
                self.tbl_preview.setItem(r_idx, c_idx, QTableWidgetItem(val))

        # 보기 옵션(가로 스크롤 유지)
        self.tbl_preview.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.tbl_preview.horizontalHeader().setStretchLastSection(False)
        self.tbl_preview.verticalHeader().setVisible(False)
        self.tbl_preview.setWordWrap(False)
