# 파일: llm_skeleton/llm_client.py
from __future__ import annotations
from typing import Optional
from openai import OpenAI


class LLMClient:
    """OpenAI Responses API 래퍼 (필요 시 교체 가능)."""

    def __init__(self) -> None:
        self._client: Optional[OpenAI] = None

    # --- 메서드: API Key 설정 ---
    def set_api_key(self, api_key: str) -> None:
        if not api_key or not api_key.strip():
            raise ValueError("유효한 OpenAI API Key를 입력하세요.")
        self._client = OpenAI(api_key=api_key.strip())

    # --- 메서드: 실행 ---
    def run(
        self,
        model: str,
        instructions: str,
        prompt_body: str,
        max_output_tokens: int,
        temperature: float,
    ) -> str:
        if self._client is None:
            raise RuntimeError("클라이언트가 초기화되지 않았습니다. set_api_key()를 먼저 호출하세요.")

        resp = self._client.responses.create(
            model=model,
            instructions=instructions,
            input=prompt_body,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
        return (resp.output_text or "").strip()
