# STAComputer.py  ───────────────────────────────────────────────────
"""
origin, dest 이름만 넘기면
① PathPlanner → 최단 경로(raw) + 절차(full)
② full → MissionProfile(seg-list)
③ seg-list 길이·기본 속도로 총 비행시간(초) 추정
   (AircraftAgent 의 build_route_from_mission() 논리 재사용)
"""
from pathlib import Path
import datetime as dt
from math import ceil
from typing import Optional, Union
# --- 내부 import(패키지/독립 실행 모두 지원) -------------------------
try:
    from Monitoring.Functions.PathPlanning   import PathPlanner, rebuild_route
    from Monitoring.Functions.MissionProfile import MissionSegment, MissionProfile
    from Monitoring.Functions.UAM_Path2Sim   import path_to_profile   # ‹재사용›
except ImportError:   # CLI 단독 실행
    import sys, pathlib
    ROOT = Path(__file__).resolve().parents[2]   # 프로젝트 루트
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from Monitoring.Functions.PathPlanning   import PathPlanner, rebuild_route
    from Monitoring.Functions.MissionProfile import MissionSegment, MissionProfile
    from Monitoring.Functions.UAM_Path2Sim   import path_to_profile

# -------------------------------------------------------------------

def compute_eta(
        origin: str,
        dest: str,
        planner: Optional[PathPlanner] = None,          # ← PathPlanner | None → Optional[…]
        vp_csv: Optional[Union[str, Path]] = None,      # ← str | Path | None  → Optional[Union[…]]
        wp_csv: Optional[Union[str, Path]] = None,
        cruise_type: str = "tiltrotor",
    ) -> dt.timedelta:
    """
    • planner가 전달되면 그 그래프를 그대로 사용
    • 아니면 vp_csv, wp_csv 경로로 새 PathPlanner를 만든다.
    • 둘 다 없으면 ValueError
    """
    # 0) PathPlanner 확보 ------------------------------------------------
    if planner is None:
        if vp_csv is None or wp_csv is None:
            raise ValueError("planner 인스턴스가 없으면 vp_csv·wp_csv 경로를 지정해야 합니다.")
        planner = PathPlanner(vp_csv, wp_csv)

    # 1) 네트워크 & 절차 포함 경로 ---------------------------------------
    dist, prev = planner.dijkstra(origin, dest)
    if origin == dest or dest not in dist or dist[dest] == float("inf"):
        raise ValueError(f"No path {origin}→{dest}")
    raw  = planner.reconstruct(prev, origin, dest)
    full = rebuild_route(planner, raw)

    # 2) MissionProfile (Path2Sim·SEG_RULES 이용)
    prof = path_to_profile(full, planner.nodes)      # km 좌표 전달

    # 3) 세그먼트 길이·속도합산
    tot_sec = 0.0
    for seg in prof.get_segments():
        # km→m 좌표 이미 변환돼 있음
        # AircraftAgent 의 build_route_from_mission() 공식을 간단히 복제
        # (z-변화 고려해 effective_speed = √(Vh²+Vv²) 또는 Vh/Vv)
        # -> 여기서는 seg별 base_effective_speed 만 필요한데,
        #    Path2Sim 코드를 그대로 사용하기 위해 한 번 더 계산
        wp_prev = (0, 0, 0) if tot_sec == 0 else last_wp
        wp_cur  = (seg.end_point["x"], seg.end_point["y"], seg.ending_altitude*0.3048)
        dx, dy, dz = [wp_cur[i]-wp_prev[i] for i in range(3)]
        length = (dx**2 + dy**2 + dz**2) ** 0.5
        # 단순-모델: 수평 70 m/s, 수직 2.5 m/s (tiltrotor 기본) → 대각 70
        if abs(dz) < 1e-3:   eff_spd = 70
        elif abs(dx)>1e-3 or abs(dy)>1e-3:
            eff_spd = (70**2 + 2.5**2) ** 0.5
        else:
            eff_spd = 2.5
        tot_sec += length / eff_spd
        last_wp  = wp_cur

    # 4) 여유버퍼 (상승/하강 절차·FATO 이탈/진입)  +90 sec
    tot_sec += 90
    return dt.timedelta(seconds=ceil(tot_sec))
