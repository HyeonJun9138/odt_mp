# 파일: llm_skeleton/config.py

# 단위: USD per 1M tokens (입력/출력/캐시입력)
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

CURRENCY = "USD"
PRICING_UNIT = "per 1M tokens"

# 기본값 (원하면 외부에서 바꾸세요)
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TEMPERATURE = 0.3
DEFAULT_MAX_OUTPUT_TOKENS = 256  # 비용 추정/제한용
