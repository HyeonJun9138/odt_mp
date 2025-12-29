# UAMParameters.py
# 각 eVTOL 비행체 타입별 수평 및 수직 속도 (m/s)
UAM_SPEEDS = {
    "lift_and_cruise": {"V_horiz": 70, "V_vert": 2.5},  
    "tiltrotor":      {"V_horiz": 70, "V_vert": 2.5}, 
    "multirotor":     {"V_horiz": 70, "V_vert": 2.5}
}

# 이동하면서 상승(대각선 상승)의 경우, 500 ft/min ≒ 2.54 m/s 사용
DIAGONAL_ASCEND_SPEED = 2.54

# 항공기 간 감속 및 holding 임계값
NM = 1852              # 1 NM (미터)
THRESHOLD_DECEL = NM   # 1.0 NM
THRESHOLD_HOLD  = NM * 0.5  # 0.5 NM

# ────────────────────────────────────────────────────────────
# “300 m / 10 s · 150 m / 5 s” 기준
# ────────────────────────────────────────────────────────────
# 최소 종적 분리
MIN_LONGITUDINAL_SEP_M   = 300.0      # 1 000 ft ≈ 304.8 m
MIN_LONGITUDINAL_SEP_TTC = 10.0       # s

# (수평/수직) 충돌 예상
COLLISION_EXPECT_DIST_M  = 150.0      # 500 ft ≈ 152.4 m
COLLISION_EXPECT_TTC_S   = 5.0        # s

# (수평/수직) 충돌 위험
COLLISION_RISK_DIST_M    = 300.0
COLLISION_RISK_TTC_S     = 10.0