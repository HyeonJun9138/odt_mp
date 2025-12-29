# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path
from PyQt5.QtCore import QThread, pyqtSignal
from openai import OpenAI

MODEL_NAME = "gpt-5-mini"

class UFTMWorker(QThread):
    finished = pyqtSignal(str)  # 원문 그대로 방출(창에서 파싱)

    def __init__(self, user_input: str, client: OpenAI,
                 prompt_path: str = r"UATMSP\prompt\uftm.txt"):
        super().__init__()
        self.user_input = user_input
        self.client = client
        self.prompt_path = Path(prompt_path)

    def run(self):
        try:
            if not self.prompt_path.exists():
                self.finished.emit(f"[오류] 프롬프트 파일이 없습니다: {self.prompt_path}")
                return

            system_prompt = self.prompt_path.read_text(encoding="utf-8")
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": self.user_input},
            ]

            # 기본 1패스 호출 (추가 파라미터 전부 없음)
            resp = self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
            )
            out = getattr(resp.choices[0].message, "content", "") or ""
            if isinstance(out, bytes):
                out = out.decode("utf-8", errors="ignore")
            else:
                out = str(out).encode("utf-8", errors="ignore").decode("utf-8")
        except Exception as e:
            out = f"[오류] {e}"

        self.finished.emit(out)
