# 파일: llm_skeleton/pricing.py
from __future__ import annotations
from typing import Dict, Tuple
from .config import PRICING

def estimate_price_usd(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_ratio: float = 0.0,
) -> Tuple[float, Dict[str, float]]:
    """
    모델/토큰 수 기반 비용 추정(USD).
    - 단위는 'per 1M tokens' 요율 사용.
    - cache_ratio: 입력 토큰 중 캐시로 청구될 비율(0~1).
    """
    if model not in PRICING:
        return 0.0, {"notice": "Unknown model pricing; returned 0.0."}

    rates = PRICING[model]
    in_rate = float(rates["input"])
    cached_rate = float(rates["cached_input"]) if rates.get("cached_input") is not None else in_rate
    out_rate = float(rates["output"])

    cached_tokens = int(max(0, min(1.0, cache_ratio)) * input_tokens)
    normal_tokens = max(0, input_tokens - cached_tokens)

    input_cost = (normal_tokens * in_rate) / 1_000_000.0
    cached_cost = (cached_tokens * cached_rate) / 1_000_000.0
    output_cost = (output_tokens * out_rate) / 1_000_000.0

    total = input_cost + cached_cost + output_cost
    breakdown = {
        "input_tokens": float(input_tokens),
        "output_tokens": float(output_tokens),
        "cached_tokens": float(cached_tokens),
        "normal_input_cost": input_cost,
        "cached_input_cost": cached_cost,
        "output_cost": output_cost,
        "total": total,
    }
    return total, breakdown
