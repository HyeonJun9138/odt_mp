# -*- coding: utf-8 -*-

DARK_BG = "#2b2f36"
CARD_BG = "#343a46"
CARD_BG_SOFT = "#3b4250"
CHAT_MINE = "#e6f4ff"
CHAT_PEER = "#f0f2f5"
ACCENT = "#2f77ff"          # 전송 버튼 액센트
ACCENT_SOFT = "#e8f1ff"     # 옅은 파랑 배경(원형 버튼 바탕)

RADIUS_TINY = 6
RADIUS_SMALL = 10
RADIUS_LARGE = 18
RADIUS_PILL = 24            # 입력부 필 라운드

BASE_QSS = f"""
QMainWindow {{
  background: {DARK_BG};
}}
"""

PANEL_TINY_ROUND = f"""
QFrame {{
  background: {CARD_BG};
  border-radius: {RADIUS_TINY}px;
}}
"""

PANEL_SMALL_ROUND = f"""
QFrame {{
  background: {CARD_BG_SOFT};
  border-radius: {RADIUS_SMALL}px;
}}
"""

# 입력부: 흰 배경 / 옅은 테두리 / 필 형태
INPUT_ROUND = f"""
QFrame#InputWrap {{
  background: #ffffff;
  border: 1px solid #e3e6ed;
  border-radius: {RADIUS_PILL}px;
}}
QLineEdit {{
  background: transparent;
  border: 0;
  padding: 4px 8px;
  color: #1f2633;
  font-size: 15px;
}}
QLineEdit::placeholder {{
  color: #9aa4b2;
}}
QToolButton#IconBtn {{
  background: transparent;
  border: 0;
  padding: 0px;
  margin: 0px;
}}
QToolButton#IconBtn:hover {{
  background: rgba(0,0,0,0.04);
  border-radius: 16px;
}}
QPushButton#MicBtn {{
  background: transparent;
  border: 0;
  padding: 0px;
  margin: 0px;
}}
QPushButton#MicBtn:hover {{
  background: rgba(0,0,0,0.04);
  border-radius: 16px;
}}
QPushButton#SendCircle {{
  background: {ACCENT_SOFT};
  color: {ACCENT};
  border: 0;
  border-radius: 18px;      /* 지름 36px 원형 */
  font-weight: 700;
}}
QPushButton#SendCircle:pressed {{
  opacity: .9;
}}
"""
# 말풍선 스타일은 이전과 동일
# 기존 BUBBLE_MINE / BUBBLE_PEER 교체
BUBBLE_MINE = """
QFrame {
  background-color: rgba(230, 244, 255, 220);  /* 옅은 하늘색, 약 86% 불투명 */
  border-radius: 18px;
}
QLabel {
  background: transparent;  /* ✅ 라벨 배경 투명 */
  border: 0;
  padding: 8px 10px;
  color: #0e1726;
}
"""

BUBBLE_PEER = """
QFrame {
  background-color: rgba(240, 242, 245, 210);  /* 옅은 회색, 약 82% 불투명 */
  border-radius: 18px;
}
QLabel {
  background: transparent;  /* ✅ 라벨 배경 투명 */
  border: 0;
  padding: 8px 10px;
  color: #0e1726;
}
"""