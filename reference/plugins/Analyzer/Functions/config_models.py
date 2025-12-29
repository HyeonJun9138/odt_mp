# -*- coding: utf-8 -*-
"""
모델/설정 상수 모듈
- UI에서는 가격을 표기하지 않지만, 내부 참조용으로 PRICING은 유지
- MODEL_IDS: 콤보 박스에 표기/선택할 모델 리스트
"""

PRICING = {
    # Flagship
    "gpt-5":            {"input": 1.25, "cached_input": 0.125, "output": 10.00},
    "gpt-5-mini":       {"input": 0.25, "cached_input": 0.025, "output": 2.00},
    "gpt-5-nano":       {"input": 0.05, "cached_input": 0.005, "output": 0.40},
    # 4o 계열
    "gpt-4o":           {"input": 5.00, "cached_input": 2.50, "output": 20.00},
    "gpt-4o-mini":      {"input": 0.60, "cached_input": 0.30, "output": 2.40},
    # 4.1 계열
    "gpt-4.1":          {"input": 2.00, "cached_input": 0.50, "output": 8.00},
    "gpt-4.1-mini":     {"input": 0.40, "cached_input": 0.10, "output": 1.60},
    # Reasoning 계열
    "o3":               {"input": 2.00, "cached_input": 0.50, "output": 8.00},
    "o3-pro":           {"input": 20.00, "cached_input": None, "output": 80.00},
    "o4-mini":          {"input": 1.10, "cached_input": 0.28, "output": 4.40},
}

# UI 콤보박스에 노출할 순서 (가격 미표기)
MODEL_IDS = [
    "gpt-5",
    "gpt-5-mini",
    "gpt-5-nano",
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4.1",
    "gpt-4.1-mini",
    "o3",
    "o3-pro",
    "o4-mini",
]

# === [AI API Key Global Accessors] ===========================================
import os
from typing import Optional

# --- OpenAI API Key (runtime store) -----------------------------------------
_OPENAI_API_KEY: str | None = None

def set_openai_api_key(key: str | None) -> None:
    """메인에서 입력한 키를 전역으로 보관."""
    global _OPENAI_API_KEY
    _OPENAI_API_KEY = (key or "").strip() or None

def get_openai_api_key() -> str | None:
    """어디서든 현재 설정된 키를 조회."""
    return _OPENAI_API_KEY
