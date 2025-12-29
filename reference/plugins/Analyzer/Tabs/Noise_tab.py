from __future__ import annotations
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import json
import tempfile
import datetime as dt
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd
import geopandas as gpd
import folium

from PyQt5.QtGui     import QFont, QPixmap, QFontMetrics,QDesktopServices
from PyQt5.QtCore    import Qt, QUrl, pyqtSignal, QThread
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QFileDialog, QSlider, QFrame, QStyle, QMessageBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractSpinBox, QSpinBox, QInputDialog,
    QAbstractItemView, QStyleOptionSlider, QTextBrowser, QDialog
)

from PyQt5.QtWebEngineWidgets import QWebEngineView
# 기존 여러 파일 import → 압축
from Analyzer.Tabs.base_tab import BaseTab
from Analyzer.Functions.core import (
    MOD3, ROOT, RES,
    SLD_MAX_SEC, TIME_STEP_SEC, TIME_PAGE_SEC, FILTER_WINDOW_S,
    TABLE_MAX_ROWS, PADDING_FACTOR, NOISE_COL_PRIORITY, NOISE_VMIN, NOISE_VMAX,
    POP_ZOOM_BUMP, DEF_GRID_SHP, DEF_VP_CSV, DEF_WP_CSV, DEF_NOISE,
    compute_daily_level_from_noise, pick_noise_df, linear_cmap as _linear_cmap, pick_any as _pick_any,
    JumpSlider
)
from Analyzer.Functions.io import load_grid, load_overlays, load_noise
from Analyzer.Functions.mapkit import (
    build_js_api, run_js, update_noise, set_ranks, clear_top, highlight_gids, clear_highlight)
import requests
from Analyzer.Functions.config_models import get_openai_api_key, set_openai_api_key

class _AIAnalyzeThread(QThread):
    done = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, api_key: str, prompt: str = "안녕하세요", parent=None):
        super().__init__(parent)
        self.api_key = api_key
        self.prompt  = prompt

    def run(self):
        try:
            url = "https://api.openai.com/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": self.prompt}],
                "temperature": 0.2,
            }
            r = requests.post(url, headers=headers, json=payload, timeout=30)
            r.raise_for_status()
            data = r.json()
            text = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
            if not text:
                raise RuntimeError("빈 응답입니다.")
            self.done.emit(text.strip())
        except Exception as e:
            self.error.emit(str(e))

# ─────────────────────────────────────────────────────────────
# 본체
# ─────────────────────────────────────────────────────────────
class NoiseTab(BaseTab):
    def __init__(self, parent=None, noise_csv: str=None):
        super().__init__(parent, logo_path=RES / "kada_logo.png", version_text="v0.0.1",show_back=True)
        self.setWindowTitle("Analyzer – Impact Viewer")
        self.resize(1440, 800)

        # 공통 맵 상태
        self._map_center = [37.5665, 126.9780]
        self._grid_bounds: Optional[tuple]   = None

        # 상태
        self._mode = "pop"
        self._hour = 0
        self._tsec = 0
        self._html = None
        self._noise_col: Optional[str] = None
        self._wcol: Optional[str] = None  # Leq 가중치 컬럼 기록

        # 웹/JS 상태
        self._web_ready = False
        self._base_ready = False
        self._map_js_name = None
        self._grid_js_name = None
        self._noise_js_name = None
        self._last_noise_vals = {}
        self._last_noise_colors = {}

        # 경로
        self.csv_noise = noise_csv   # 문자열/Path 모두 허용
        self.shp_grid  = DEF_GRID_SHP
        self.vp_csv    = DEF_VP_CSV
        self.wp_csv    = DEF_WP_CSV

        # 캐시
        self._gdf: Optional[gpd.GeoDataFrame] = None
        self._noise: Optional[pd.DataFrame]   = None

        # 혼합보기(Top20) 상태
        self._top20_vals  = {}
        self._top20_cols  = {}
        self._top20_gids  = []
        self._top20_ranks = {}
        
        

        # 뒤로가기 버튼 동작: 창 닫기(메인에서 신호로 처리)
        self.backRequested.connect(self._on_back)


        # UI 구성/데이터 로드/지도 준비
        self._build_ui()
        self._load_all()
        self._ensure_base_map()
        
    def _on_back(self):
        self.close()

    def _make_logo_clickable(self, label):
        if not label:
            return
        label.setCursor(Qt.PointingHandCursor)
        def _open(_: object):
            QDesktopServices.openUrl(QUrl("https://kada.konkuk.ac.kr"))
        label.mousePressEvent = _open
    # ───────────────────────────── 계산 루틴 ─────────────────────────────
    def _compute_daily_level_from_noise(self, use_lden: bool = True):
        if self._noise is None:
            return None
        return compute_daily_level_from_noise(self._noise, use_lden=use_lden)

    def _pick_noise_df(self, agg: str = "auto"):
        df, col, wcol = pick_noise_df(self._noise, self._tsec, self._hour, agg=agg)
        self._noise_col = col
        self._wcol = wcol
        return df

    # ───────────────────────────── UI 구성 ─────────────────────────────
    def _build_ui(self):

        root = self.body

        # ── 상단: 슬라이더 + 시/분/초 스핀
        topline = QHBoxLayout(); topline.setSpacing(12)

        # 슬라이더
        slider_col = QVBoxLayout()
        self.sld = JumpSlider(Qt.Horizontal); self.sld.setObjectName("TimeSlider")
        self.sld.setRange(0, SLD_MAX_SEC); self.sld.setSingleStep(TIME_STEP_SEC)
        self.sld.setPageStep(TIME_PAGE_SEC); self.sld.setTickInterval(3600)
        self.sld.setTickPosition(QSlider.TicksBelow); self.sld.setTracking(False)
        self.sld.valueChanged.connect(self._on_time_changed)
        slider_col.addWidget(self.sld)
        marks = QHBoxLayout()
        lbl00 = QLabel("00"); lbl00.setFixedWidth(24)
        lbl24 = QLabel("24."); lbl24.setFixedWidth(24); lbl24.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        marks.addWidget(lbl00); marks.addStretch(1); marks.addWidget(lbl24)
        slider_col.addLayout(marks)
        topline.addLayout(slider_col, 3)

        # 시/분/초 스핀 (휠/키/버튼 모두 가능, suffix로 명확히 표기)
        self.spin_h = QSpinBox(objectName="HSpin"); self.spin_h.setRange(0, 23); self.spin_h.setSuffix(" h")
        self.spin_m = QSpinBox(objectName="MSpin"); self.spin_m.setRange(0, 59); self.spin_m.setSuffix(" m")
        self.spin_s = QSpinBox(objectName="SSpin"); self.spin_s.setRange(0, 59); self.spin_s.setSuffix(" s")

        # 스텝을 전역 TIME_STEP_SEC에 맞춰서 설정(분/초)
        step_s = max(1, TIME_STEP_SEC if TIME_STEP_SEC < 60 else 1)
        step_m = max(1, TIME_STEP_SEC // 60 if TIME_STEP_SEC >= 60 else 1)
        self.spin_h.setSingleStep(1)
        self.spin_m.setSingleStep(step_m)
        self.spin_s.setSingleStep(step_s)

        for sp in (self.spin_h, self.spin_m, self.spin_s):
            sp.setButtonSymbols(QAbstractSpinBox.UpDownArrows)
            sp.setFixedWidth(90)

        # 현재 _tsec 값을 스핀에 반영
        s = int(getattr(self, "_tsec", 0)) % SLD_MAX_SEC
        self.spin_h.setValue(s // 3600)
        self.spin_m.setValue((s % 3600) // 60)
        self.spin_s.setValue(s % 60)

        # 스핀 값 변경 시 슬라이더로 적용(시간 동기화)
        self.spin_h.valueChanged.connect(self._on_spins_changed)
        self.spin_m.valueChanged.connect(self._on_spins_changed)
        self.spin_s.valueChanged.connect(self._on_spins_changed)

        spbox = QHBoxLayout()
        spbox.addWidget(self.spin_h)
        spbox.addWidget(self.spin_m)
        spbox.addWidget(self.spin_s)
        topline.addLayout(spbox, 2)
        root.addLayout(topline)

        # ── 중앙: 좌 지도 / 우 로그+표(분할)
        center = QHBoxLayout(); center.setSpacing(12)

        # 왼쪽: 지도
        mapPanel = QFrame(objectName="Panel"); mapL = QVBoxLayout(mapPanel); mapL.setContentsMargins(6,6,6,6)
        self.web = QWebEngineView(); self.web.setMinimumHeight(520); mapL.addWidget(self.web)
        self.web.loadFinished.connect(self._on_web_loaded)
        center.addWidget(mapPanel, 3)

        # 오른쪽: 위 로그 / 아래 표
        logPanel = QFrame(objectName="Panel")
        logL = QVBoxLayout(logPanel)
        logL.setContentsMargins(6, 6, 6, 6)
        logL.setSpacing(12)   # ← 간격을 12px로 (기존보다 넉넉하게)
        
        # 위쪽 텍스트 로그
        self.log = QLabel("Log")
        self.log.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.log.setWordWrap(True)
        logL.addWidget(self.log, 1)  # ← 로그 먼저 추가

        # ▼ 로그와 표 사이 구분선 + 여백 (여기로 이동)
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setFrameShadow(QFrame.Plain)
        sep.setFixedHeight(1)
        sep.setStyleSheet("background:#3a4050;")
        logL.addWidget(sep)
        logL.addSpacing(6)  # 구분선 아래 여백

        # 아래쪽 표(제목 + 테이블)
        self.tbl_title = QLabel("")
        logL.addWidget(self.tbl_title)

        self.tbl = QTableWidget()
        self.tbl.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tbl.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tbl.setSelectionMode(QAbstractItemView.SingleSelection)
        self.tbl.setObjectName("DataTable")        # 스타일시트 적용용
        self.tbl.setAlternatingRowColors(True)     # 줄무늬
        self.tbl.setShowGrid(False)                # 격자선 숨김(깔끔)
        self.tbl.verticalHeader().setDefaultSectionSize(28)
        self.tbl.horizontalHeader().setFixedHeight(32)
        self.tbl.horizontalHeader().setDefaultAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.tbl.horizontalHeader().setHighlightSections(False)  # 선택시 헤더 반전 방지
        self.tbl.verticalHeader().setVisible(False)
        self.tbl.horizontalHeader().setStretchLastSection(True)
        self.tbl.setSortingEnabled(True)
        self.tbl.cellClicked.connect(self._on_table_row_clicked)  # ★ 표 클릭 → 지도 하이라이트
        logL.addWidget(self.tbl, 1)  # stretch=1

        center.addWidget(logPanel, 2)
        root.addLayout(center, 1)

        # ── 하단 버튼들
        btns = QHBoxLayout(); btns.setSpacing(12)
        self.btn_pop   = QPushButton("인구격자 보기",  objectName="ActionBtn")
        self.btn_noise = QPushButton("소음 영향 보기", objectName="ActionBtn")
        self.btn_mix   = QPushButton("혼합 보기",      objectName="ActionBtn")
        self.btn_pop.clicked.connect(lambda: self._switch("pop"))
        self.btn_noise.clicked.connect(lambda: self._switch("noise"))
        self.btn_mix.clicked.connect(lambda: self._switch("mix"))

        self.btn_top20 = QPushButton("상위 20만 표시", objectName="TopBtn")
        self.btn_top20.setCheckable(True)
        self.btn_top20.setVisible(False)
        self.btn_top20.clicked.connect(self._toggle_top20)

        for b in (self.btn_pop, self.btn_noise, self.btn_mix, self.btn_top20):
            b.setMinimumWidth(150)

        btns.addWidget(self.btn_pop)
        btns.addWidget(self.btn_noise)
        btns.addWidget(self.btn_mix)
        btns.addWidget(self.btn_top20)
        btns.addStretch(1)

        # ★ AI 분석 버튼
        self.btn_ai    = QPushButton("Analysis with AI", objectName="ActionBtn")
        self.btn_ai.clicked.connect(self._on_ai_clicked)   # <-- 변경 요점
        self.btn_saveH = QPushButton("Save report as",   objectName="ActionBtn"); self.btn_saveH.clicked.connect(self._save_html)
        self.btn_saveP = QPushButton("Save report as",   objectName="ActionBtn"); self.btn_saveP.clicked.connect(self._save_png)
        for b in (self.btn_ai, self.btn_saveH, self.btn_saveP): b.setMinimumWidth(150)

        btns.addWidget(self.btn_ai); btns.addWidget(self.btn_saveH); btns.addWidget(self.btn_saveP)
        root.addLayout(btns)

        # 내부 상태
        self._ai_result_text = ""
        
        # --- Footer: KADA notice (작은 글씨) ---
        footer_line = QFrame()
        footer_line.setFrameShape(QFrame.HLine)
        footer_line.setFrameShadow(QFrame.Sunken)
        root.addWidget(footer_line)

        footer = QLabel("This software is developed by KADA © 2025 All rights reserved")
        footer.setObjectName("appFooter")
        footer.setAlignment(Qt.AlignCenter)
        footer.setStyleSheet("font-size: 11px; color: #8a8f98; padding: 4px 0;")
        root.addWidget(footer)
        
        # 초기 슬라이더/스핀 동기화
        self.sld.setValue(int(getattr(self, "_tsec", 0)))

    # ───────────────────────────── 시간 동기화 ─────────────────────────────
    def _on_time_changed(self, v: int):
        """슬라이더 변경 → 내부 시간/스핀/렌더 갱신"""
        step = max(1, TIME_STEP_SEC)
        v = int(round(v / step) * step)
        self._tsec = max(0, min(SLD_MAX_SEC, v))
        self._hour = self._tsec // 3600

        # 스핀 업데이트 (시그널 잠시 차단)
        h = (self._tsec // 3600) % 24
        m = (self._tsec % 3600) // 60
        s = self._tsec % 60
        for sp, val in ((self.spin_h, h), (self.spin_m, m), (self.spin_s, s)):
            sp.blockSignals(True); sp.setValue(val); sp.blockSignals(False)

        # 렌더 및 표 갱신
        self._render_current()

    def _on_ai_clicked(self):
        # 1) API 키 확보(없으면 즉석 입력 받아 전역 저장)
        key = get_openai_api_key()
        if not key:
            k, ok = QInputDialog.getText(self, "API Key 필요", "OpenAI API Key를 입력하세요:")
            if not ok or not k:
                return
            set_openai_api_key(k)
            key = k

        # 2) 버튼 상태 → 진행 중
        self.btn_ai.setEnabled(False)
        self.btn_ai.setText("분석 중…")
        self.btn_ai.setStyleSheet(
            "QPushButton { background:#8e24aa; color:#fff; border:0; border-radius:10px; padding:10px 18px; }"
        )

        # 3) 비동기 호출 시작(테스트: '안녕하세요' 송신)
        self._ai_thr = _AIAnalyzeThread(key, "안녕하세요", parent=self)
        self._ai_thr.done.connect(self._on_ai_done)
        self._ai_thr.error.connect(self._on_ai_error)
        self._ai_thr.start()


    def _on_ai_done(self, text: str):
        self._ai_result_text = text or ""
        # 버튼을 '분석 보기'로 전환
        self.btn_ai.setEnabled(True)
        self.btn_ai.setText("분석 보기")
        self.btn_ai.setStyleSheet(
            "QPushButton { background:#2e7d32; color:#fff; border:0; border-radius:10px; padding:10px 18px; }"
        )
        try:
            self.btn_ai.clicked.disconnect()
        except Exception:
        # 이미 끊긴 상태면 무시
            pass
        self.btn_ai.clicked.connect(self._open_ai_result)


    def _on_ai_error(self, msg: str):
        QMessageBox.critical(self, "AI 분석 실패", f"요청 중 오류가 발생했습니다.\n\n{msg}")
        # 버튼 원복
        self.btn_ai.setEnabled(True)
        self.btn_ai.setText("Analysis with AI")
        self.btn_ai.setStyleSheet("")  # 테마(QSS)로 복귀


    def _open_ai_result(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("AI 분석 결과")
        dlg.resize(720, 520)
        lay = QVBoxLayout(dlg)
        view = QTextBrowser()
        view.setPlainText(self._ai_result_text or "(빈 응답)")
        lay.addWidget(view, 1)
        btns = QHBoxLayout()
        btn_copy = QPushButton("복사"); btn_close = QPushButton("닫기")
        btn_copy.clicked.connect(lambda: QApplication.clipboard().setText(view.toPlainText()))
        btn_close.clicked.connect(dlg.accept)
        btns.addStretch(1); btns.addWidget(btn_copy); btns.addWidget(btn_close)
        lay.addLayout(btns)
        dlg.exec_()

    def _on_spins_changed(self, *_):
        """시/분/초 스핀 변경 → 슬라이더로 적용(나머지 동기화는 _on_time_changed가 처리)"""
        h, m, s = self.spin_h.value(), self.spin_m.value(), self.spin_s.value()
        tsec = (h * 3600) + (m * 60) + s
        step = max(1, TIME_STEP_SEC)
        tsec = int(round(tsec / step) * step) % SLD_MAX_SEC
        if tsec != self._tsec:
            self.sld.setValue(tsec)

    # ───────────────────────────── 로그/테이블 유틸 ─────────────────────────
    def _log(self, msg: str):
        self.log.setText(msg)

    def _set_table_headers(self, headers):
        """테이블 헤더/기본 속성 세팅"""
        self.tbl.setSortingEnabled(False)
        self.tbl.clear()
        self.tbl.setColumnCount(len(headers))
        self.tbl.setHorizontalHeaderLabels(headers)
        self.tbl.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        if len(headers) >= 2:
            self.tbl.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.tbl.setSortingEnabled(True)

    def _fill_table(self, rows, headers, title):
        """rows: list of tuples (gid, value, [extra...]) -> 표 채우기"""
        self._set_table_headers(headers)
        n = min(len(rows), TABLE_MAX_ROWS)
        self.tbl.setRowCount(n)
        for i in range(n):
            r = rows[i]
            # 0번째 컬럼 = GID (UserRole로 정수 저장)
            it0 = QTableWidgetItem(str(int(r[0])))
            it0.setData(Qt.UserRole, int(r[0]))
            self.tbl.setItem(i, 0, it0)
            # 나머지 값들
            for j in range(1, len(r)):
                val = r[j]
                if isinstance(val, float):
                    txt = f"{val:.1f}"
                elif isinstance(val, int):
                    txt = f"{val:,d}"
                else:
                    txt = str(val)
                self.tbl.setItem(i, j, QTableWidgetItem(txt))
        self.tbl_title.setText(title)

    def _build_table_population(self):
        """인구 격자 표 (Pop>0만)"""
        if self._gdf is None:
            self._fill_table([], ["Grid ID","Population"], "표: 인구 격자 (데이터 없음)")
            return
        pop_col = next((c for c in self._gdf.columns if c.lower() in ("val","population","pop","cnt","count")), None)
        if not pop_col:
            self._fill_table([], ["Grid ID","Population"], "표: 인구 격자 (컬럼 없음)")
            return
        df = self._gdf[["gid", pop_col]].dropna().copy()
        df["gid"] = pd.to_numeric(df["gid"], errors="coerce").astype("Int64")
        df[pop_col] = pd.to_numeric(df[pop_col], errors="coerce").fillna(0).astype(int)
        df = df[df[pop_col] > 0]
        df = df.sort_values(pop_col, ascending=False)
        rows = [(int(g), int(v)) for g, v in zip(df["gid"], df[pop_col])]
        title = f"표: 인구 격자 (총 {len(rows):,})"
        self._fill_table(rows, ["Grid ID","Population"], title)

    def _build_table_noise_window(self):
        """현재 시각 윈도우의 소음 격자 표 (0 dB 제외)"""
        df = self._pick_noise_df(agg="auto")
        if df is None or df.empty:
            self._fill_table([], ["Grid ID","dB"], "표: 소음 격자 (선택 시각 데이터 없음)")
            return
        df = df.dropna(subset=["L"]).copy()
        df["L"] = pd.to_numeric(df["L"], errors="coerce")
        df = df[df["L"] > 0].sort_values("L", ascending=False)
        rows = [(int(g), float(l)) for g, l in zip(df["gid"], df["L"])]
        title = f"표: 소음 격자 (총 {len(rows):,})"
        self._fill_table(rows, ["Grid ID","dB"], title)

    def _build_table_lden(self):
        """혼합보기(=L_den)의 상위 격자 표"""
        df = self._compute_daily_level_from_noise(use_lden=True)
        if df is None or df.empty:
            self._fill_table([], ["Grid ID","L_den(dB)"], "표: L_den (데이터 없음)")
            return
        df = df.dropna(subset=["L"]).copy()
        df["L"] = pd.to_numeric(df["L"], errors="coerce")
        df = df.sort_values("L", ascending=False)
        rows = [(int(g), float(l)) for g, l in zip(df["gid"], df["L"])]
        title = f"표: L_den (총 {len(rows):,})"
        self._fill_table(rows, ["Grid ID","L_den(dB)"], title)

    def _on_table_row_clicked(self, row: int, col: int):
        """표 행 클릭 시 해당 GID 격자 하이라이트(+패닝)"""
        try:
            gid_item = self.tbl.item(row, 0)
            if not gid_item:
                return
            gid = gid_item.data(Qt.UserRole)
            if gid is None:
                gid = int(gid_item.text())
            self._js_highlight_gids([gid], pan=True)
        except Exception:
            pass

    # ───────────────────────────── 지도 초기화/JS API ─────────────────────────
    def _base_map(self):
        return folium.Map(self._map_center, tiles="CartoDB positron", control_scale=True)

    def _push_map(self, fmap: folium.Map):
        if self._html and os.path.exists(self._html):
            try: os.remove(self._html)
            except Exception: pass
        self._html = tempfile.NamedTemporaryFile(suffix=".html", delete=False).name
        fmap.save(self._html)
        self.web.load(QUrl.fromLocalFile(self._html))

    def _fit_to_bounds(self, fmap, bounds, pad=PADDING_FACTOR, zoom_bump=0):
        if bounds is None: return
        try:
            minx, miny, maxx, maxy = [float(v) for v in list(bounds)]
        except Exception:
            return
        dx, dy = maxx - minx, maxy - miny
        if dx <= 0 and dy <= 0:
            dx = dy = 0.002
        pad_x = max(dx * pad, 0.001)
        pad_y = max(dy * pad, 0.001)
        south, west  = miny - pad_y, minx - pad_x
        north, east  = maxy + pad_y, maxx + pad_x
        fmap.fit_bounds([[south, west], [north, east]])

        # 지연 fit/zoom (렌더 후 적용)
        try:
            from folium import MacroElement
            from jinja2 import Template
            class _FitLater(MacroElement):
                _template = Template(u"""
                {% macro script(this, kwargs) %}
                var m = {{this._parent.get_name()}};
                var b = L.latLngBounds([[{{this.south}},{{this.west}}],[{{this.north}},{{this.east}}]]);
                setTimeout(function(){
                    m.fitBounds(b);
                    {% if this.bump != 0 %}
                    try { m.setZoom(m.getZoom() + {{this.bump}}); } catch(e) {}
                    {% endif %}
                }, 120);
                {% endmacro %}
                """)
                def __init__(self, s, w, n, e, bump):
                    super().__init__()
                    self.south = s; self.west = w; self.north = n; self.east = e; self.bump = int(bump)
            fmap.add_child(_FitLater(south, west, north, east, zoom_bump))
        except Exception:
            pass

        self._map_center = [(south+north)/2.0, (west+east)/2.0]

    def _add_vp_wp_and_rings(self, fmap: folium.Map):
        """Vertiport/Waypoint + INR/MTR 원 추가"""
        try:
            vp_df = pd.read_csv(self.vp_csv) if (self.vp_csv and self.vp_csv.exists()) else None
            wp_df = pd.read_csv(self.wp_csv) if (self.wp_csv and self.wp_csv.exists()) else None
        except Exception as e:
            self._log(f"※ Vertiport/Waypoint 읽기 실패: {e}")
            return

        if vp_df is not None:
            for _, r in vp_df.iterrows():
                lat = _pick_any(r, "위도", "lat"); lon = _pick_any(r, "경도", "lon")
                name= _pick_any(r, "Vertiport 명", "name", default="")
                if pd.isna(lat) or pd.isna(lon): continue
                folium.CircleMarker([float(lat), float(lon)],
                                    radius=5, color="#1a237e",
                                    fill=True, fill_color="#448aff", fill_opacity=.9,
                                    popup=f"Vertiport {name}").add_to(fmap)
                inr_km = _pick_any(r, "INR(km)", "INR", default=0) or 0
                mtr_km = _pick_any(r, "MTR(km)", "MTR", default=0) or 0
                if inr_km and inr_km > 0:
                    folium.Circle([float(lat), float(lon)],
                                  radius=float(inr_km)*1000.0,
                                  color="#2e7d32", weight=1.5, fill=False, opacity=0.8).add_to(fmap)
                if mtr_km and mtr_km > 0:
                    folium.Circle([float(lat), float(lon)],
                                  radius=float(mtr_km)*1000.0,
                                  color="#8e24aa", weight=1.0, fill=False, opacity=0.5, dash_array="5,5").add_to(fmap)

        if wp_df is not None:
            dgeo = {}
            for _, r in wp_df.iterrows():
                name = _pick_any(r, "Waypoint 명", "name", default="")
                lat  = _pick_any(r, "위도", "lat")
                lon  = _pick_any(r, "경도", "lon")
                if not name or pd.isna(lat) or pd.isna(lon): continue
                dgeo[str(name)] = (float(lat), float(lon))
            for _, r in wp_df.iterrows():
                s = str(_pick_any(r, "Waypoint 명", "name", default=""))
                links = str(_pick_any(r, "Link", "link", default=""))
                if s not in dgeo: continue
                for t in map(str.strip, links.split(",")):
                    if t in dgeo:
                        folium.PolyLine([dgeo[s], dgeo[t]], color="#1565c0", weight=2, opacity=0.35).add_to(fmap)

    def _ensure_base_map(self):
        """지도/격자/VP/WP를 최초 1회만 생성. 이후엔 JS로 레이어 스타일 갱신."""
        if self._base_ready or self._gdf is None:
            return
        fmap = self._base_map()

        # 인구 바탕 색 사전계산
        base = self._gdf.copy()
        pop_col = next((c for c in base.columns if c.lower() in ("val","population","pop","cnt","count")), None)
        if pop_col is None:
            base["__pop__"]=0; pop_col="__pop__"
        vmin = float(base[pop_col].min() if base[pop_col].notna().any() else 0)
        vmax = float(base[pop_col].max() or 1.0)
        cmapP = _linear_cmap(["#ECEFF1","#90CAF9","#1976D2"], vmin, vmax, "Population")
        base["__pop_color__"] = base[pop_col].apply(lambda v: cmapP(v) if pd.notna(v) else "#ECEFF1")

        # 격자 레이어(바탕)
        grid_layer = folium.GeoJson(
            base.to_json(),
            name="grid_pop",
            style_function=lambda feat: {
                "fillColor": feat["properties"].get("__pop_color__","#ECEFF1"),
                "color":"#999","weight":0.2,"fillOpacity":0.55
            },
            tooltip=folium.GeoJsonTooltip(fields=["gid", pop_col], aliases=["Grid ID","Population"])
        )
        grid_layer.add_to(fmap)

        # 소음 오버레이 레이어(처음엔 투명)
        noise_layer = folium.GeoJson(
            base[["gid","geometry"]].to_json(),
            name="grid_noise",
            style_function=lambda feat: {
                "fillColor":"#FF5722","color":"#FF5722","weight":0,"fillOpacity":0.0
            }
        )
        noise_layer.add_to(fmap)

        # 이름 보관(Leaflet 전역 변수명)
        self._map_js_name   = fmap.get_name()
        self._grid_js_name  = grid_layer.get_name()
        self._noise_js_name = noise_layer.get_name()

        # VP/WP, 링
        self._add_vp_wp_and_rings(fmap)

        # 경계 맞춤(1회)
        self._fit_to_bounds(fmap, self._grid_bounds, zoom_bump=POP_ZOOM_BUMP)

        # 저장/로드(1회)
        self._push_map(fmap)
        
    def _on_web_loaded(self, ok: bool):
        self._web_ready = bool(ok)
        if not ok:
            self._log("페이지 로드 실패")
            return
        # JS API 주입
        self._inject_js_api()
        self._base_ready = True
        # 첫 렌더
        self._render_current()
        
    def _inject_js_api(self):
        if not (self._map_js_name and self._grid_js_name and self._noise_js_name):
            return
        js = build_js_api(self._map_js_name, self._grid_js_name, self._noise_js_name)
        self._js_call(js)
    # ───────────────────────────── 렌더 라우팅 ─────────────────────────────
    def _render_current(self):
        if not (self._web_ready and self._base_ready):
            return
        if self._mode == "pop":
            self._render_population_fast()
        elif self._mode == "noise":
            self._render_noise_fast()
        else:
            self._render_mixed_fast()

    def _render_population_fast(self):
        """인구 배경만 표시 + 인구 표"""
        self.btn_top20.setVisible(False)
        self._js_call("window.JY.setMode('pop'); window.JY.setEdge(false);")
        self._js_clear_top()
        self._js_clear_highlight()
        if self._gdf is None:
            self._log("SHP가 없습니다. Analyzer\\Sources 폴더에 넣어주세요.")
            self._fill_table([], ["Grid ID","Population"], "표: 인구 격자 (데이터 없음)")
            return
        self._log(f"인구 격자   |  시간 {self._hour:02d}시")
        self._build_table_population()

    def _render_noise_fast(self):
        """선택 시각 윈도우의 소음 오버레이 + 소음 표"""
        self.btn_top20.setVisible(False)
        self._js_clear_top()
        self._js_clear_highlight()
        if self._gdf is None or self._noise is None:
            self._log("필수 파일이 없습니다. (SHP / noise_log.csv)")
            self._js_call("window.JY.setMode('pop');")
            self._fill_table([], ["Grid ID","dB"], "표: 소음 격자 (데이터 없음)")
            return

        # 표 먼저 생성
        self._build_table_noise_window()

        # 오버레이
        df = self._pick_noise_df(agg="auto")
        if df is None or df.empty or not getattr(self, "_noise_col", None):
            self._js_call("window.JY.updateNoise({}, {}); window.JY.setMode('noise');")
            self._log(f"선택 시각(±{FILTER_WINDOW_S}s)에 해당하는 소음 데이터가 없습니다.")
            return

        vals = {str(int(g)): float(v) for g, v in zip(df["gid"], df["L"]) if pd.notna(g) and pd.notna(v)}
        cmap = _linear_cmap(["#FFE0B2","#FF9800","#FF5722","#BF360C"], NOISE_VMIN, NOISE_VMAX, "")
        cols = {k: cmap(v) for k, v in vals.items()}
        self._last_noise_vals = vals
        self._last_noise_colors = cols

        self._js_update_noise(vals, cols)
        self._js_call("window.JY.setMode('noise'); window.JY.setEdge(false);")
        self._log(f"소음 영향 보기 |  시간 {self._hour:02d}시")

    def _render_mixed_fast(self):
        """
        혼합 보기 = L_den.
        기본은 인구 배경 + 오버레이 비움(깨끗한 지도) + 우측 상위 20(+인구) 텍스트/표.
        '상위 20만 표시' 버튼으로 지도에 상위 20만 칠하고 윤곽선/번호 표시.
        """
        if self._gdf is None or self._noise is None:
            self._log("SHP 또는 noise_log가 없습니다.")
            self._js_call("window.JY.setMode('pop'); window.JY.setEdge(false);")
            self.btn_top20.setVisible(False)
            self._fill_table([], ["Grid ID","L_den(dB)"], "표: L_den (데이터 없음)")
            return

        df = self._compute_daily_level_from_noise(use_lden=True)
        if df is None or df.empty:
            self._js_call("window.JY.updateNoise({}, {}); window.JY.setMode('mix'); window.JY.setEdge(false);")
            self._log("혼합 보기(=L_den): 계산할 데이터가 없습니다.")
            self.btn_top20.setVisible(False)
            self._fill_table([], ["Grid ID","L_den(dB)"], "표: L_den (데이터 없음)")
            return

        # 상위 20 준비
        self._prepare_top20(df)

        # 기본 뷰: 오버레이 비움(지도를 깨끗하게), 버튼 ON/OFF는 유저가 선택
        self._js_call("window.JY.setMode('mix'); window.JY.setEdge(false);")
        self._js_update_noise({}, {})
        self._js_clear_top()
        self._js_clear_highlight()
        self.btn_top20.blockSignals(True); self.btn_top20.setChecked(False); self.btn_top20.blockSignals(False)
        self.btn_top20.setVisible(True)

        # 우측 텍스트: 상위 20(+인구)
        pop_col = next((c for c in self._gdf.columns if c.lower() in ("val","population","pop","cnt","count")), None)
        pop_map = {}
        if pop_col:
            tmp = self._gdf[["gid", pop_col]].copy()
            tmp["gid"] = pd.to_numeric(tmp["gid"], errors="coerce").astype("Int64")
            pop_map = {int(g): (int(v) if pd.notna(v) else 0) for g, v in zip(tmp["gid"], tmp[pop_col])}

        lines = []
        for i, gid in enumerate(self._top20_gids, 1):
            val = self._top20_vals.get(str(gid), np.nan)
            pop = pop_map.get(int(gid), 0)
            lines.append(f"{i:>2}. Grid ID {gid:>6}   {val:6.1f} dB   Population {pop}")

        header = f"혼합 보기(=L_den): 총 {len(df.dropna(subset=['L'])):,} 격자"
        self._log(header + "\n\n[상위 20]\n" + ("\n".join(lines) if lines else "(없음)"))

        # 표: L_den 전체(정렬)
        self._build_table_lden()

    def _prepare_top20(self, df: pd.DataFrame):
        """df[['gid','L']]에서 상위 20 추려서 매핑/리스트 저장 + 색/랭크 지정"""
        dclean = df.dropna(subset=["L"]).copy()
        if dclean.empty:
            self._top20_vals, self._top20_cols, self._top20_gids, self._top20_ranks = {}, {}, [], {}
            return
        dclean["L"] = dclean["L"].astype(float)
        top = dclean.sort_values("L", ascending=False).head(20)
        gids = top["gid"].tolist()
        self._top20_gids  = [int(g) for g in gids]
        self._top20_ranks = {str(int(g)): idx+1 for idx, g in enumerate(gids)}
        cmap = _linear_cmap(["#FFE0B2","#FF9800","#FF5722","#BF360C"], NOISE_VMIN, NOISE_VMAX, "")
        self._top20_vals = {str(int(g)): float(lv) for g, lv in zip(top["gid"], top["L"])}
        self._top20_cols = {k: cmap(v) for k, v in self._top20_vals.items()}

    def _toggle_top20(self, checked: bool):
        """혼합보기일 때만 등장하는 버튼 동작: 상위 20만 지도에 표시/해제"""
        if checked:
            # 데이터가 없으면 계산
            if not self._top20_vals:
                df = self._compute_daily_level_from_noise(use_lden=True)
                if df is None or df.empty:
                    self._log("상위 20을 표시할 데이터가 없습니다.")
                    self.btn_top20.setChecked(False)
                    return
                self._prepare_top20(df)
            # 상위 20만 칠하고, 윤곽선 강조 + 번호 배지
            self._js_update_noise(self._top20_vals, self._top20_cols)
            self._js_call("window.JY.setMode('mix'); window.JY.setEdge(true);")
            self._js_set_ranks(self._top20_ranks)
            self._js_highlight_gids([])  # 기존 하이라이트 제거
        else:
            # 다시 깨끗한 혼합보기(=오버레이 없음)
            self._js_update_noise({}, {})
            self._js_clear_top()
            self._js_call("window.JY.setMode('mix'); window.JY.setEdge(false);")

    # ───────────────────────────── JS 헬퍼 ─────────────────────────────
    def _js_call(self, code: str):
        if getattr(self, "_web_ready", False):
            run_js(self.web, code)

    def _js_update_noise(self, val_map: dict, color_map: dict):
        update_noise(self.web, val_map, color_map)

    def _js_set_ranks(self, rank_map: dict):
        set_ranks(self.web, rank_map)

    def _js_clear_top(self):
        clear_top(self.web)

    def _js_highlight_gids(self, gids, pan=True):
        highlight_gids(self.web, gids, pan=pan)

    def _js_clear_highlight(self):
        clear_highlight(self.web)

    # ───────────────────────────── 모드 전환 ─────────────────────────────
    def _switch(self, mode: str):
        self._mode = mode
        self.btn_top20.setVisible(mode == "mix")
        if mode != "mix":
            if self.btn_top20.isChecked():
                self.btn_top20.setChecked(False)
            self._js_update_noise({}, {})
            self._js_call("try{window.JY.setEdge(false);}catch(e){}")
            self._js_clear_top()
        self._render_current()
        self._js_clear_highlight()

    # ───────────────────────────── 파일 로드 ─────────────────────────────
    def _load_all(self):
        msgs = []

        # SHP
        self._gdf, self._grid_bounds, self._map_center = load_grid(getattr(self, "shp_grid", DEF_GRID_SHP))
        if self._gdf is not None:
            msgs.append(f"✔ SHP: {self.shp_grid.relative_to(ROOT) if self.shp_grid.exists() else self.shp_grid}  ({len(self._gdf):,} grids)")
        else:
            msgs.append(f"⚠ SHP 없음: {getattr(self, 'shp_grid', DEF_GRID_SHP)}")

        # noise_log
        noise_path = Path(self.csv_noise) if self.csv_noise else None
        self._noise = load_noise(noise_path) if noise_path else None
        if self._noise is not None:
            try:
                label = noise_path.resolve().relative_to(ROOT)
            except Exception:
                label = noise_path
            msgs.append(f"✔ noise_log: {label}  ({len(self._noise):,} rows)")
        else:
            msgs.append("⚠ noise_log 없음")

        # Vertiport / Waypoint
        self._vp, self._wp = load_overlays(self.vp_csv, self.wp_csv)
        msgs.append(f"ⓘ Vertiport: {'OK' if self._vp is not None else '없음'} / Waypoint: {'OK' if self._wp is not None else '없음'}")

        # 초기 시간 동기화 (tsec 있으면 첫 값으로)
        if self._noise is not None and "tsec" in self._noise.columns and not self._noise.empty:
            t0 = int(self._noise["tsec"].iloc[0])
            self._tsec, self._hour = t0, t0 // 3600
            if hasattr(self, "sld"):
                self.sld.blockSignals(True); self.sld.setValue(t0); self.sld.blockSignals(False)
            h = (self._tsec // 3600) % 24
            m = (self._tsec % 3600) // 60
            s = self._tsec % 60
            for sp, val in ((self.spin_h, h), (self.spin_m, m), (self.spin_s, s)):
                sp.blockSignals(True); sp.setValue(val); sp.blockSignals(False)

        self._log("\n".join(msgs))

    # ───────────────────────────── 저장/종료 ─────────────────────────────
    def _save_html(self):
        if not self._html:
            self._log("저장할 내용이 없습니다. 먼저 지도를 생성하세요."); return

        db = (MOD3 / "database")
        try: db.mkdir(parents=True, exist_ok=True)
        except Exception: pass
        default_path = str(db / "impact_report.html")

        fn, _ = QFileDialog.getSaveFileName(
            self, "Save report as (HTML)", default_path, "HTML (*.html)"
        )
        if fn:
            body = Path(self._html).read_text(encoding="utf-8")
            inj  = f"""
                <script>
                window.addEventListener('load', function(){{
                try {{
                    if (window.JY) {{
                    window.JY.updateNoise({json.dumps(self._last_noise_vals)}, {json.dumps(self._last_noise_colors)});
                    window.JY.setMode({json.dumps(self._mode)});
                    }}
                }} catch(e){{}}
                }});
                </script>
            """
            Path(fn).write_text(body + inj, encoding="utf-8")
            self._log(f"✔ 저장됨: {fn}")

    def _save_png(self):
        db = (MOD3 / "database")
        try: db.mkdir(parents=True, exist_ok=True)
        except Exception: pass
        default_path = str(db / "impact_report.png")

        fn, _ = QFileDialog.getSaveFileName(
            self, "Save report as (PNG)", default_path, "PNG (*.png)"
        )
        if not fn: return
        pix = self.web.grab()
        pix.save(fn, "PNG")
        self._log(f"✔ 저장됨: {fn}")

    def closeEvent(self, ev):
        try:
            if self._html and os.path.exists(self._html): os.remove(self._html)
        except Exception:
            pass
        super().closeEvent(ev)


# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.environ.setdefault("QTWEBENGINE_CHROMIUM_FLAGS", "--disable-gpu")
    os.environ.setdefault("QTWEBENGINE_DISABLE_SANDBOX", "1")
    import sys
    app = QApplication(sys.argv)
    w = NoiseTab()
    w.show()
    sys.exit(app.exec_())
