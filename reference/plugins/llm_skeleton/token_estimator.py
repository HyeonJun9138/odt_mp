# 파일: llm_skeleton/token_estimator.py
from __future__ import annotations
from typing import Optional

try:
    import tiktoken  # 선택 사항
except Exception:
    tiktoken = None


class TokenEstimator:
    """
    토큰 수 추정기.
    - tiktoken 사용 가능하면 모델별 인코딩으로 정확 추정
    - 없으면 문자수/3(대략치) 휴리스틱
    """

    def __init__(self, fallback_chars_per_token: float = 3.0) -> None:
        self.fallback_cpt = fallback_chars_per_token

    def estimate(self, text: str, model: str) -> int:
        if not text:
            return 0
        if tiktoken is None:
            return max(1, int(len(text) / self.fallback_cpt))

        try:
            enc = tiktoken.encoding_for_model(model)
        except Exception:
            # 최신 모델명 미인지 등 -> 보편 인코딩 사용
            try:
                enc = tiktoken.get_encoding("o200k_base")
            except Exception:
                enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))

    @staticmethod
    def words_to_tokens(words: int) -> int:
        # 영어기준 ~1.3x, 한국어는 다소 차이. 안전상 여유 포함
        return max(1, int(words * 1.5))
