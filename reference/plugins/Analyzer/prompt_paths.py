# -*- coding: utf-8 -*-
from __future__ import annotations

def prompt_for_mode(mode: str) -> str:
    """
    mode에 따른 프롬프트 파일 경로 반환.
    - single  -> Analyzer\prompt\single.txt
    - compare -> Analyzer\prompt\compare.txt
    """
    if mode == "compare":
        return r"Analyzer\prompt\compare.txt"
    # 기본: 단일
    return r"Analyzer\prompt\single.txt"
