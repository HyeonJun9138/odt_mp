# -*- coding: utf-8 -*-
import os, json, re, ast
from pathlib import Path

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QPixmap
from PyQt5.QtWidgets import (
    QMainWindow, QTableWidget, QTableWidgetItem, QHeaderView, QLabel, QGraphicsBlurEffect,
    QDialog, QVBoxLayout, QPlainTextEdit, QPushButton, QHBoxLayout, QApplication,
    QTextEdit, QDialogButtonBox
)

from openai import OpenAI

from styles import BASE_QSS
from grid_util import ROWS, COLS, put_spanned_widget
from widgets.core_panel import CorePanel
from widgets.detail_panel import DetailPanel
from widgets.square_buttons_bar import SquareButtonsBar
from widgets.chat_panel import ChatPanel
from widgets.chat_input import ChatInput
from widgets.spacer import VSpacer
from widgets.display_panel import DisplayPanel

from assistant_worker import AssistantWorker
from uftm_worker import UFTMWorker  

SHOW_DEBUG_NUMBERS = False

class AppWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setStyleSheet(BASE_QSS)
        self._last_uftm_raw = ""
        self._gpt_busy = False
        self._uftm_busy = False
        
        # ✅ 프리엠션 상태 관리
        self._drop_chat_reply = False   # 채팅 응답을 '낚아채기' 처리할지
        self._alert_loading_index = None  # "대응이 필요합니다. 생각중" 말풍선 인덱스
        self._mode = 1  # 1: 일반, 2: UFTM 대응

        # ===== OpenAI 클라이언트: Window에서 단일 생성 → 워커에 주입 =====
        self._openai_client = self._make_openai_client()

        # ── 배경
        bg_path = Path(__file__).resolve().parent / "resource" / "background.png"
        self.bg_label = QLabel(self)
        self.bg_label.setPixmap(QPixmap(str(bg_path)))
        self.bg_label.setScaledContents(True)
        blur = QGraphicsBlurEffect(); blur.setBlurRadius(8)
        self.bg_label.setGraphicsEffect(blur)
        self.bg_label.lower()

        # ── 테이블 레이아웃
        self.table = QTableWidget(ROWS, COLS, self)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionMode(QTableWidget.NoSelection)
        self.table.setFocusPolicy(Qt.NoFocus)
        self.table.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.table.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.table.setShowGrid(False)
        self.table.setStyleSheet("""
            QTableWidget { background: transparent; }
            QTableWidget::item { border: none; background: transparent; }
        """)
        hh = self.table.horizontalHeader(); vh = self.table.verticalHeader()
        hh.hide(); vh.hide()
        hh.setSectionResizeMode(QHeaderView.Stretch)
        vh.setSectionResizeMode(QHeaderView.Stretch)
        self.table.setCornerButtonEnabled(False)
        self.setCentralWidget(self.table)

        if SHOW_DEBUG_NUMBERS:
            self._fill_numbers()

        # -----------------------------
        # 영역 위젯 배치
        # -----------------------------
        # 핵심 정보
        self.core_panel = CorePanel(self)
        put_spanned_widget(self.table, self.core_panel, [(32,49), (152,169)])

        # 세부 정보
        self.detail_panel = DetailPanel(self)
        put_spanned_widget(self.table, self.detail_panel, [(212,219), (482, 489)])

        self.display_panel = DisplayPanel(self)
        put_spanned_widget(self.table, self.display_panel, [(221,229), (491, 499)])

        # 버튼바
        self.sqbar = SquareButtonsBar(self, count=10)
        put_spanned_widget(self.table, self.sqbar, [(512,529), (542,559)])
        self.sqbar.clicked_index.connect(self._on_toolbar_clicked)  # ▶ 10번 입력 트리거

        # 채팅 패널/입력
        self.chat_panel = ChatPanel(self)
        put_spanned_widget(self.table, self.chat_panel, [(51,59), (81,89), (501,509)])
        spacer = VSpacer(self); put_spanned_widget(self.table, spacer, [(531,539)])
        cinput = ChatInput(self); put_spanned_widget(self.table, cinput, [(561,569)])
        cinput.submitted.connect(self._on_submit_chat)

        # 경고 클릭 → actions 표시
        self.core_panel.warningSelected.connect(self._show_actions)
        self.core_panel.rawRequested.connect(self._show_raw_uftm)
        self._pending_lv3 = False

        # ✅ DetailPanel에서 액션 선택 신호 연결 → DisplayPanel 시각화
        self.detail_panel.actionSelected.connect(self._on_action_selected)
        self.detail_panel.tokenClicked.connect(self._on_action_token_clicked)

        # 초기 안내
        QTimer.singleShot(200, lambda: self.chat_panel.add_message("안녕하세요. KADA 교통관리 AI 입니다.", False))
        QTimer.singleShot(400, lambda: self.chat_panel.add_message("우측 아래 입력창에서 메시지를 보내보세요.", False))

    def _on_action_selected(self, action_dict: dict):
        # DisplayPanel에 전체 액션 시각화
        self.display_panel.visualize_action(action_dict)

    def _on_action_token_clicked(self, action_dict: dict, token: str):
        # 특정 토큰(쉼표 분해 단어) 강조 시각화
        self.display_panel.visualize_action(action_dict, focus_token=token)

    # ===== OpenAI 클라이언트 생성(공유) =====
    def _make_openai_client(self) -> OpenAI:
        api_key = (os.getenv("OPENAI_API_KEY") or "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").strip()
        # env에 없을 경우 None을 넘기면 SDK가 내부적으로 에러를 던집니다.
        # 이 동작이 의도된 경우가 아니라면 여기서 UI로 입력받아도 됩니다.
        return OpenAI(api_key=api_key if api_key else None)

    # ✅ 원문 보기 대화상자
    def _show_raw_uftm(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("UFTM 추론 원문")
        lay = QVBoxLayout(dlg); lay.setContentsMargins(10,10,10,10); lay.setSpacing(8)

        edit = QPlainTextEdit(dlg)
        edit.setReadOnly(True)
        edit.setLineWrapMode(QPlainTextEdit.NoWrap)
        f = QFont("Consolas"); f.setPointSize(10)
        edit.setFont(f)
        edit.setPlainText(self._last_uftm_raw or "(원문 없음)")
        lay.addWidget(edit, 1)

        btns = QHBoxLayout(); btns.addStretch(1)
        btnCopy = QPushButton("복사"); btnClose = QPushButton("닫기")
        btns.addWidget(btnCopy); btns.addWidget(btnClose)
        lay.addLayout(btns)

        def _copy():
            QApplication.clipboard().setText(edit.toPlainText())
        btnCopy.clicked.connect(_copy)
        btnClose.clicked.connect(dlg.accept)

        dlg.resize(800, 500)
        dlg.exec_()

    # -----------------------------
    # 기존 대화 GPT
    # -----------------------------
    def _on_submit_chat(self, text: str):
        if self._gpt_busy:
            return
        self._gpt_busy = True

        text = (text or "").strip()
        if not text:
            self._gpt_busy = False
            return

        # 사용자 말풍선
        self.chat_panel.add_message(text, True)  # mine=True  :contentReference[oaicite:5]{index=5}

        # 어시스턴트 '입력 중...' 말풍선
        self.loading_index = self.chat_panel.add_message("입력 중...", False)  # mine=False  :contentReference[oaicite:6]{index=6}

        # 🔹 일반 대화는 항상 mode=1
        self._mode = 1
        self.chat_thread = AssistantWorker(text, client=self._openai_client, mode=1)
        self.chat_thread.finished.connect(self._on_chat_reply)
        self.chat_thread.finished.connect(self.chat_thread.deleteLater)
        self.chat_thread.start()

    def _on_chat_reply(self, reply: str):
        # 프리엠션(낚아채기)되었으면 이 응답은 버린다.
        if self._drop_chat_reply:
            # 진행중 응답을 무시하고 플래그만 해제
            self._drop_chat_reply = False
            self._gpt_busy = False
            return

        # 정상 응답이면 '입력 중...' 말풍선 대체
        if hasattr(self, "loading_index") and self.loading_index is not None:
            self.chat_panel.replace_message(self.loading_index, reply, False)  # :contentReference[oaicite:7]{index=7}
            self.loading_index = None
        else:
            # 방어적: 혹시 인덱스 없으면 그냥 추가
            self.chat_panel.add_message(reply, False)

        self._gpt_busy = False
        self._mode = 1  # 원복

    # -----------------------------
    # 핵심 정보 GPT (버튼 10 → 입력 → 호출 → CorePanel 출력)
    # -----------------------------
    def _on_toolbar_clicked(self, idx: int):
        if idx != 10:
            return
        if self._uftm_busy:
            return

        MAX_CHARS = 6000  # 과도한 입력 방지(컨텍스트 예산 보호)

        # 입력 다이얼로그
        dlg = QDialog(self)
        dlg.setWindowTitle("핵심 정보 입력")
        lay = QVBoxLayout(dlg); lay.setContentsMargins(10,10,10,10); lay.setSpacing(8)

        lab = QLabel("입력값:")
        te = QTextEdit(dlg)
        te.setAcceptRichText(False)
        te.setPlaceholderText(f"최대 {MAX_CHARS}자까지 입력 가능합니다")
        lay.addWidget(lab)
        lay.addWidget(te, 1)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, dlg)
        lay.addWidget(btns)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)

        if dlg.exec_() != QDialog.Accepted:
            return

        text = (te.toPlainText() or "").strip()
        if not text:
            return
        if len(text) > MAX_CHARS:
            text = text[:MAX_CHARS]

        # 호출 직전 사용자에게 표시 및 중복 방지
        self._uftm_busy = True
        self.core_panel.begin_wait("분석중")

        # Window에서 생성한 client 주입
        self.uftm_thread = UFTMWorker(
            user_input=text,
            client=self._openai_client,
            prompt_path=r"UATMSP\prompt\uftm.txt",
        )
        self.uftm_thread.finished.connect(self._on_uftm_done)
        self.uftm_thread.finished.connect(self.uftm_thread.deleteLater)
        self.uftm_thread.start()

    def _on_uftm_done(self, raw: str):
        self.core_panel.end_wait()
        self._uftm_busy = False
        self._last_uftm_raw = raw

        items = self._parse_uftm_output(raw)
        self.core_panel.set_warnings(items or [{
            "priority":"Level 0","warning":"결과 없음","reason":raw[:200]
        }])

        has_lv3 = any("3" in str(d.get("priority","")) for d in items)
        self._pending_lv3 = has_lv3
        if has_lv3: self.core_panel.start_alert_blink()
        else:       self.core_panel.stop_alert_blink()

        # ✅ 여기서 즉시 "비상 대응 모드"로 전환 실행
        self._kickoff_assistant_for_uftm(raw)

    def _kickoff_assistant_for_uftm(self, raw: str):
        # 1) 진행 중인 일반 채팅 응답이 있다면 낚아채기
        if self._gpt_busy:
            self._drop_chat_reply = True  # 이후 도착하는 채팅 응답은 폐기

            # 기존 '입력 중...' 말풍선이 있으면 그것을 치환
            if hasattr(self, "loading_index") and self.loading_index is not None:
                self.chat_panel.replace_message(self.loading_index, "대응이 필요합니다. 생각중", False)  # :contentReference[oaicite:9]{index=9}
                self._alert_loading_index = self.loading_index
                self.loading_index = None
            else:
                # 없으면 새로 띄움
                self._alert_loading_index = self.chat_panel.add_message("대응이 필요합니다. 생각중", False)
        else:
            # 2) 일반 채팅 진행 중이 아니어도 비상 말풍선 시작
            self._alert_loading_index = self.chat_panel.add_message("대응이 필요합니다. 생각중", False)

        # 3) AssistantWorker(mode=2)로 원문 그대로 전달 (채팅 입력 없이 비공개 트리거)
        self._mode = 2
        self.alert_thread = AssistantWorker(raw, client=self._openai_client, mode=2)
        self.alert_thread.finished.connect(self._on_alert_reply)
        self.alert_thread.finished.connect(self.alert_thread.deleteLater)
        self.alert_thread.start()


    def _on_alert_reply(self, reply: str):
        # '대응이 필요합니다. 생각중' 말풍선을 결과로 치환
        if self._alert_loading_index is not None:
            self.chat_panel.replace_message(self._alert_loading_index, reply, False)  # :contentReference[oaicite:10]{index=10}
            self._alert_loading_index = None
        else:
            self.chat_panel.add_message(reply, False)

        # 모드 원복
        self._mode = 1
        
    def _show_actions(self, warn_item: dict):
        self.detail_panel.set_actions(warn_item.get("actions") or [])
        # 사용자가 LV3 항목을 클릭하면 깜빡임 중지
        if self._pending_lv3 and ("3" in str(warn_item.get("priority",""))):
            self._pending_lv3 = False
            self.core_panel.stop_alert_blink()

    # =============================
    # UFTM 출력 파싱 유틸
    # =============================
    def _normalize_jsonish(self, s: str) -> str:
        # 코드펜스/스마트쿼트 정리
        s = re.sub(r"```(?:json)?", "", s)
        s = s.replace("```", "")
        s = s.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
        return s

    def _extract_top_level_json(self, s: str) -> list:
        # { } 중첩 깊이로 맨 위 레벨만 추출
        objs, depth, start = [], 0, None
        for i, ch in enumerate(s):
            if ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}":
                if depth > 0:
                    depth -= 1
                    if depth == 0 and start is not None:
                        objs.append(s[start:i+1])
                        start = None
        if objs:
            return objs
        # top-level이 전혀 없을 때만 줄단위 JSONL 처리
        for ln in s.splitlines():
            ln = ln.strip()
            if ln.startswith("{") and ln.endswith("}"):
                objs.append(ln)
        return objs

    def _safe_load_obj(self, chunk: str):
        t = self._normalize_jsonish(chunk)
        # 1차: 표준 JSON
        try:
            return json.loads(t)
        except Exception:
            pass
        # 2차: 파이썬 리터럴(덜 엄격)
        try:
            return ast.literal_eval(t)
        except Exception:
            pass
        # 3차: 최소 필드만 복구(정규식)
        def grab(key):
            m = re.search(rf'["“”]{key}["“”]\s*:\s*["“”](.*?)["“”]', t)
            return m.group(1).strip() if m else ""
        actions = []
        m = re.search(r'["“”]actions["“”]\s*:\s*\[(.*?)\]', t, re.S)
        if m:
            for it in re.findall(r'\{(.*?)\}', m.group(1), re.S):
                a = re.search(r'["“”]action["“”]\s*:\s*["“”](.*?)["“”]', it)
                w = re.search(r'["“”]why["“”]\s*:\s*["“”](.*?)["“”]', it)
                if a or w:
                    actions.append({"action": a.group(1) if a else "", "why": w.group(1) if w else ""})
        d = {"warning": grab("warning"), "reason": grab("reason"), "priority": grab("priority"), "actions": actions}
        return d if any(d.values()) else None

    def _parse_uftm_output(self, text: str) -> list:
        t = self._normalize_jsonish(text)
        chunks = self._extract_top_level_json(t)
        items = []

        def add_if_warn(obj):
            # obj가 dict이면 경고키가 있으면 추가
            if isinstance(obj, dict):
                # (1) 직접 경고 객체
                if "warning" in obj or "priority" in obj:
                    items.append(obj); return
                # (2) warnings/items/alerts 같은 리스트 내부
                for k in ("warnings", "items", "alerts"):
                    if k in obj and isinstance(obj[k], list):
                        for it in obj[k]:
                            if isinstance(it, dict) and ("warning" in it or "priority" in it):
                                items.append(it)
                        return
            # obj가 리스트면 각 원소 검사
            if isinstance(obj, list):
                for it in obj:
                    if isinstance(it, dict) and ("warning" in it or "priority" in it):
                        items.append(it)

        for ch in chunks:
            obj = self._safe_load_obj(ch)
            if obj is None:
                continue
            add_if_warn(obj)

        # 보정: top-level이 전혀 없을 때 줄 단위로 재시도
        if not items:
            for ln in t.splitlines():
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    parsed = self._safe_load_obj(ln)
                    if parsed:
                        add_if_warn(parsed)
                except Exception:
                    pass

        return items

    # -----------------------------
    # 유틸 (디버그 번호 등)
    # -----------------------------
    def showEvent(self, e):
        super().showEvent(e)
        self._fit_font()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self.bg_label.resize(self.size())
        self._fit_font()

    def _fit_font(self):
        if not SHOW_DEBUG_NUMBERS:
            return
        vp = self.table.viewport().size()
        if vp.width() == 0 or vp.height() == 0:
            return
        cell_w = vp.width() / float(COLS)
        cell_h = vp.height() / float(ROWS)
        px = int(min(cell_w, cell_h) * 0.45)
        px = max(px, 10)
        f = QFont(); f.setPixelSize(px)
        self.table.setFont(f)
