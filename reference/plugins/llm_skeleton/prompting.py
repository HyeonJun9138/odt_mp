# 파일: llm_skeleton/prompting.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class Example:
    """프롬프트 예시 (입·출력)."""
    input_text: str
    output_text: str


@dataclass
class PromptSpec:
    """
    프롬프트 스펙(틀).
    - task: 작업 정의(뭘 할 것인지)
    - inputs_definition: 입력 항목 정의/예시/유의점
    - thinking_guidelines: 사고(내적 과정) 가이드 (최종 답변에 노출 X)
    - constraints: 필수 제약 조건 (형식, 금지사항 등)
    - output_format: 출력 스키마/예시 (최종 출력 형태)
    - style_guidelines: 문체/톤/스타일 지침
    - examples: 예시 입력/출력
    """
    name: str
    task: str
    inputs_definition: str
    thinking_guidelines: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    output_format: str = ""
    style_guidelines: List[str] = field(default_factory=list)
    examples: List[Example] = field(default_factory=list)

    # --- 메서드: 시스템 인스트럭션 생성 ---
    def system_instructions(self) -> str:
        return (
            "You are a precise, concise professional assistant. "
            "Follow the provided TASK, CONSTRAINTS, and OUTPUT FORMAT exactly. "
            "Do not reveal internal reasoning steps. Provide only the final answer."
        )

    # --- 메서드: 프롬프트 컴파일 ---
    def compile(self, **payload: Dict[str, str]) -> str:
        """
        섹션별로 명확히 구획한 프롬프트 바디 문자열을 생성한다.
        payload에는 실제 입력 값(예: source_text 등)을 전달.
        """
        lines = []
        lines.append("## TASK\n" + self.task.strip())

        lines.append("\n## INPUTS & DEFINITIONS\n" + self.inputs_definition.strip())

        if self.thinking_guidelines:
            lines.append("\n## THINKING GUIDELINES (do not reveal)\n- " + "\n- ".join(self.thinking_guidelines))

        if self.constraints:
            lines.append("\n## CONSTRAINTS\n- " + "\n- ".join(self.constraints))

        if self.output_format:
            lines.append("\n## OUTPUT FORMAT\n" + self.output_format.strip())

        if self.style_guidelines:
            lines.append("\n## STYLE GUIDELINES\n- " + "\n- ".join(self.style_guidelines))

        if self.examples:
            ex_lines = []
            for i, ex in enumerate(self.examples, 1):
                ex_lines.append(f"[Example {i}]")
                ex_lines.append("Input:\n" + ex.input_text.strip())
                ex_lines.append("Output:\n" + ex.output_text.strip())
            lines.append("\n## EXAMPLES\n" + "\n\n".join(ex_lines))

        if payload:
            # 실제 실행에 사용되는 입력 페이로드
            pay_lines = ["\n## PROVIDED INPUT"]
            for k, v in payload.items():
                pay_lines.append(f"- {k}:\n{v.strip()}")
            lines.append("\n".join(pay_lines))

        # 최종 출력 지시
        lines.append(
            "\n## FINAL INSTRUCTION\n"
            "Return ONLY the content requested in OUTPUT FORMAT. No preamble or meta commentary."
        )
        return "\n".join(lines)
