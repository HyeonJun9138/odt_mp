# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path
from PyQt5.QtCore import QThread, pyqtSignal
from openai import OpenAI

MODEL_NAME = "gpt-5-nano"

class AssistantWorker(QThread):
    finished = pyqtSignal(str)

    def __init__(self, user_message: str, client: OpenAI,
                 mode: int = 1,
                 prompt_path: str = r"UATMSP\prompt\supporter.txt"):
        super().__init__()
        self.user_message = user_message
        self.client = client
        self.mode = mode
        self.prompt_path = Path(prompt_path)

    def _load_system_prompt(self) -> str:
        base = "You are a helpful assistant."
        if self.prompt_path.exists():
            try:
                base = self.prompt_path.read_text(encoding="utf-8")
            except Exception:
                base = self.prompt_path.read_text(encoding="utf-8", errors="ignore")
        # ✅ 모드 주입
        return f"""{base}

### RUNTIME_MODE: {self.mode}
# Mode 1: 일반 대화(교통 관련 대화만), 친절/간결/사실 위주
# Mode 2: UFTM 사건 대응 집중 모드. 입력은 'UFTM RAW' 그대로 들어옴.
# - 핵심 위험/우선순위/즉시 실행 조치 요약
# - 오판 최소화(보수적), 짧은 근거 포함
# - 최종: 실행 항목 bullet로 정리
"""

    def run(self):
        try:
            system_prompt = self._load_system_prompt()
            resp = self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    # mode==2 일 때도 user에 '원문 그대로' 전달 (비공개 트리거 용)
                    {"role": "user", "content": self.user_message},
                ],
            )
            out = getattr(resp.choices[0].message, "content", "") or ""
            if isinstance(out, bytes):
                out = out.decode("utf-8", errors="ignore")
            else:
                out = str(out).encode("utf-8", errors="ignore").decode("utf-8")
        except Exception as e:
            out = f"[오류] {e}"
        self.finished.emit(out)
