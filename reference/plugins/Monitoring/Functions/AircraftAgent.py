from __future__ import annotations      # 이미 있으면 그대로 두세요
from typing import Optional
import math, itertools
from typing import Tuple

# ── 패키지/스크립트 양쪽 실행 지원 ─────────────────────────────────────
try:
    from .UAMParameters import (
        UAM_SPEEDS, DIAGONAL_ASCEND_SPEED,
        MIN_LONGITUDINAL_SEP_M, MIN_LONGITUDINAL_SEP_TTC,
        COLLISION_EXPECT_DIST_M, COLLISION_EXPECT_TTC_S,
        COLLISION_RISK_DIST_M,   COLLISION_RISK_TTC_S
    )
except ImportError:
    import sys, pathlib
    THIS = pathlib.Path(__file__).resolve()
    PKG  = THIS.parent          # …/Monitoring/Functions
    if str(PKG) not in sys.path:
        sys.path.insert(0, str(PKG))

    from UAMParameters import (
        UAM_SPEEDS, DIAGONAL_ASCEND_SPEED, THRESHOLD_DECEL, THRESHOLD_HOLD
    )


R_EARTH_M = 6_378_137.0              # WGS-84 반지름 [m]
from .geo_utils import xy_to_lonlat, bearing_deg

import itertools



class AircraftAgent:
    def __init__(self, mission_segments, dt, flight_type, initial_progress=0.0, a_max=0.5):
        """
        mission_segments: MissionSegment 객체들의 리스트.
            각 MissionSegment는 end_point (x, y)만 입력받으며 내부 DEFAULTS는 그대로 유지됩니다.
        dt: 시뮬레이션 시간 간격 (s)
        flight_type: 'lift_and_cruise', 'tiltrotor', 'multirotor' 중 하나.
        initial_progress: 초기 진행 거리 (m)
        a_max: 최대 가속도 (m/s²), 기본값 0.5 m/s²
        """
        self.dt = dt
        self.flight_type = flight_type
        self.V_horiz = UAM_SPEEDS[flight_type]["V_horiz"]
        self.V_vert = UAM_SPEEDS[flight_type]["V_vert"]
        self.mission_segments = mission_segments
        self.waypoints, self.segments_info = self.build_route_from_mission()
        self.progress = initial_progress
        self.position = self.get_position(self.progress)
        self.a_max = a_max
        self.current_speed = 0.0  # 현재 effective speed (m/s)
        self.sensor_ttc      = None

        self.seg_idx     = 0
        self.cur_segment = mission_segments[0]

        # 누적 길이 배열 + 총 길이
        self._cum_dist = []
        acc = 0.0
        for info in self.segments_info:
            acc += info["length"]
            self._cum_dist.append(acc)
        self._total_length = acc               # ← 총 길이 캐시

        # Phase 저장용 내부 변수
        self._phase = self.cur_segment.segment_id.upper()

        # 센서 관련 속성: 진행 방향 상 장애물(다른 비행체)와의 거리를 (미터 단위) 저장합니다.
        # 값이 None이면 센서 입력이 없는 것으로 간주합니다.
        self.sensor_distance = None

    def _update_segment(self):
        # progress가 다음 경계 넘으면 세그먼트 인덱스 증가
        while self.seg_idx < len(self._cum_dist) and self.progress >= self._cum_dist[self.seg_idx]:
            self.seg_idx += 1
            if self.seg_idx < len(self.mission_segments):
                self.cur_segment = self.mission_segments[self.seg_idx]
    # ───────────────────────────────────────────────────────────
    #  남은 비행 시간(초) 추정
    # ───────────────────────────────────────────────────────────
    def estimate_remaining_time(self) -> float:
        """
        • 현재 세그먼트의 남은 길이 + 이후 세그먼트 전체 길이를
          base_effective_speed 로 나눠 합산한다.
        • 이미 도착( ARRIVED ) 했으면 0 반환.
        """
        # ── ARRIVED( or seg_idx overflow ) 보호 ───────────────
        if self.seg_idx >= len(self.segments_info):
            return 0.0

        rem_time = 0.0

        # ① 현재 세그먼트 남은 길이
        cur_len  = self.segments_info[self.seg_idx]["length"]
        prev_len = self._cum_dist[self.seg_idx - 1] if self.seg_idx else 0.0
        left_in_seg = max(cur_len - (self.progress - prev_len), 0.0)

        v_eff = self.segments_info[self.seg_idx]["base_effective_speed"]
        if v_eff > 0:
            rem_time += left_in_seg / v_eff

        # ② 이후 세그먼트 전체
        for seg in self.segments_info[self.seg_idx + 1:]:
            rem_time += seg["length"] / seg["base_effective_speed"]

        return rem_time            # seconds

    def update_sensor(self, distance: Optional[float], ttc: Optional[float] = None):
        """
        · distance : 앞 기체(또는 장애물)까지 수평 거리 [m]  (None = 정보 없음)
        · ttc      : Time-to-Collision, 충돌 예상까지 남은 시간 [s]
                     (None 이면 속도·방향상 충돌 경로가 아님을 의미)
        """
        self.sensor_distance = distance
        self.sensor_ttc      = ttc

    def get_current_segment_index(self):
        """
        현재 진행 중인 세그먼트 인덱스를 반환합니다.
        (누적 진행 거리를 바탕으로 현재 어느 세그먼트에 있는지 계산)
        """
        progress = self.progress
        cum = 0.0
        for i, seg in enumerate(self.segments_info):
            if progress < cum + seg["length"]:
                return i
            cum += seg["length"]
        return len(self.segments_info) - 1


    _last_hdg: float = 0.0

    def get_heading_deg(self) -> float:
        """
        지오데식 bearing 사용 – 좌표 스케일 불일치 문제 제거.
        수직 구간(B/J)에서는 직전 heading 유지.
        """
        pos_lon, pos_lat = xy_to_lonlat(self.position["x"], self.position["y"])

        # 앞쪽에서 첫 수평 WP 찾기
        for idx in range(self.seg_idx + 1, len(self.waypoints)):
            wp = self.waypoints[idx]
            if abs(wp["x"] - self.position["x"]) > 0.01 or \
               abs(wp["y"] - self.position["y"]) > 0.01:
                wp_lon, wp_lat = xy_to_lonlat(wp["x"], wp["y"])
                hdg = bearing_deg(pos_lon, pos_lat, wp_lon, wp_lat)
                self._last_hdg = hdg
                return hdg

        # 못 찾으면 이전 값 유지
        return self._last_hdg



    def build_route_from_mission(self):
        """
        MissionSegment 리스트를 기반으로 전체 경로(waypoints)와 세그먼트 정보를 구성
        - 첫 세그먼트의 start_point 또는 end_point를 출발점으로 사용
        - A/K 세그먼트(지상이동)는 z=0 경로로 유지
        - F 세그먼트는 lane_type(L/R)에 따라 100m 차선 오프셋 적용
        """
        if not self.mission_segments:
            return [], []

        waypoints: list[dict[str, float]] = []
        segment_edges: list[tuple[dict[str, float], dict[str, float], 'MissionSegment']] = []
        ft_to_m = 0.3048
        offset_distance = 100.0  # 차선 폭(좌·우 오프셋) [m]

        def append_wp(pt: dict[str, float]):
            copy = {"x": pt["x"], "y": pt["y"], "z": pt["z"]}
            waypoints.append(copy)
            return copy

        first = self.mission_segments[0]
        start_base = first.start_point or first.end_point
        append_wp({"x": start_base["x"], "y": start_base["y"], "z": 0.0})

        for seg in self.mission_segments:
            seg_id = seg.segment_id.upper()
            wp_z = seg.ending_altitude * ft_to_m

            if seg_id in {"A", "K"}:
                candidate = None
                if seg.start_point:
                    candidate = {"x": seg.start_point["x"],
                                 "y": seg.start_point["y"],
                                 "z": 0.0}
                start_node = waypoints[-1]
                if candidate is not None:
                    last = waypoints[-1]
                    if math.hypot(candidate["x"] - last["x"], candidate["y"] - last["y"]) > 1e-6:
                        start_node = append_wp(candidate)
                    else:
                        start_node = last
                end_node = append_wp({"x": seg.end_point["x"],
                                      "y": seg.end_point["y"],
                                      "z": 0.0})
                segment_edges.append((start_node, end_node, seg))
                continue

            if seg_id == "F" and waypoints:
                start_node = waypoints[-1]
                lane = (seg.lane_type or "").upper()
                if lane.startswith(("L", "R")):
                    dx = seg.end_point["x"] - start_node["x"]
                    dy = seg.end_point["y"] - start_node["y"]
                    norm = math.hypot(dx, dy) or 1.0
                    ux, uy = dx / norm, dy / norm
                    right_x, right_y = uy, -ux
                    sign = 1.0 if lane.startswith("R") else -1.0
                    offset_start = {
                        "x": start_node["x"] + sign * right_x * offset_distance,
                        "y": start_node["y"] + sign * right_y * offset_distance,
                        "z": start_node["z"]
                    }
                    waypoints[-1] = offset_start
                    start_node = waypoints[-1]
                    adjusted_end = {
                        "x": seg.end_point["x"] + sign * right_x * offset_distance,
                        "y": seg.end_point["y"] + sign * right_y * offset_distance,
                        "z": wp_z
                    }
                    end_node = append_wp(adjusted_end)
                    segment_edges.append((start_node, end_node, seg))
                    continue

            start_node = waypoints[-1]
            end_node = append_wp({"x": seg.end_point["x"], "y": seg.end_point["y"], "z": wp_z})
            segment_edges.append((start_node, end_node, seg))

        KT_TO_MPS = 0.514444
        FTPM_TO_MPS = 0.3048 / 60.0
        segments_info = []
        for start_node, end_node, seg in segment_edges:
            dx = end_node["x"] - start_node["x"]
            dy = end_node["y"] - start_node["y"]
            dz = end_node["z"] - start_node["z"]
            length = math.sqrt(dx * dx + dy * dy + dz * dz)
            if length <= 1e-6:
                length = 1e-6

            horiz_speed = seg.target_horizontal_speed * KT_TO_MPS if seg.target_horizontal_speed else 0.0
            if horiz_speed <= 0:
                horiz_speed = self.V_horiz

            vert_speed = seg.target_vertical_speed * FTPM_TO_MPS if seg.target_vertical_speed else 0.0
            if vert_speed <= 0:
                vert_speed = self.V_vert if abs(dz) > 1e-6 else 0.0

            if abs(dx) > 1e-6 or abs(dy) > 1e-6:
                if abs(dz) > 1e-6 and vert_speed > 0:
                    effective_speed = math.sqrt(horiz_speed ** 2 + vert_speed ** 2)
                else:
                    effective_speed = horiz_speed
            else:
                effective_speed = vert_speed if vert_speed > 0 else horiz_speed

            if effective_speed <= 0:
                effective_speed = 0.1

            segments_info.append({
                "length": length,
                "base_effective_speed": effective_speed
            })

        return waypoints, segments_info


    def get_total_route_length(self):
        return sum(seg["length"] for seg in self.segments_info)

    def get_position(self, progress):
        total = self.get_total_route_length()
        if progress >= total:
            return self.waypoints[-1]
        remaining = progress
        for i, seg in enumerate(self.segments_info):
            if remaining <= seg["length"]:
                frac = remaining / seg["length"] if seg["length"] > 0 else 1
                start = self.waypoints[i]
                end = self.waypoints[i+1]
                x = start["x"] + (end["x"] - start["x"]) * frac
                y = start["y"] + (end["y"] - start["y"]) * frac
                z = start["z"] + (end["z"] - start["z"]) * frac
                return {"x": x, "y": y, "z": z}
            else:
                remaining -= seg["length"]
        return self.waypoints[-1]

    def get_desired_speeds(self):
        """
        현재 진행 상태(self.progress)를 기반으로 세그먼트 내 진행 비율(fraction)을 계산한 후,
        미션 세그먼트 ID에 따라 원하는 수직(v_target) 및 수평(h_target) 속도를 반환합니다.
        (각 세그먼트에 대한 예시 프로파일은 내부 주석을 참고)
        """
        total_length = self.get_total_route_length()
        progress = self.progress
        cum = 0.0
        for i, seg_info in enumerate(self.segments_info):
            if progress < cum + seg_info["length"]:
                f = (progress - cum) / seg_info["length"]
                seg_id = self.mission_segments[i].segment_id.upper()
                Vh = self.V_horiz
                Vv = self.V_vert
                if seg_id == "B":
                    v_target = Vv * (0.1 + 0.9 * f)
                    h_target = 0
                elif seg_id == "C":
                    if f <= 0.8:
                        v_target = Vv
                        h_target = Vh * f
                    else:
                        v_target = Vv * (1 - (f - 0.8) / 0.2)
                        h_target = 0.8 * Vh
                elif seg_id == "D":
                    v_target = 0
                    h_target = 0.8 * Vh
                elif seg_id == "E":
                    if f <= 0.5:
                        v_target = 2 * Vv * f
                    else:
                        v_target = Vv * (2 - 2 * f)
                    h_target = 0.8 * Vh + 0.2 * Vh * f
                elif seg_id == "F":
                    v_target = 0
                    h_target = Vh
                elif seg_id == "G":
                    v_target = -Vv * f
                    h_target = Vh - 0.2 * Vh * f
                elif seg_id == "H":
                    v_target = 0
                    h_target = 0.8 * Vh
                elif seg_id == "I":
                    v_target = -0.2 * Vv
                    h_target = Vh * (1 - f)
                elif seg_id == "J":
                    v_target = -0.2 * Vv * (1 - f)
                    h_target = 0
                else:
                    v_target = Vv
                    h_target = Vh
                return v_target, h_target
            else:
                cum += seg_info["length"]
        return 0, 0

    def current_desired_effective_speed(self):
        v_target, h_target = self.get_desired_speeds()
        return math.sqrt(v_target**2 + h_target**2)

    # ───────────────────────────────────────────────────────────
    #  새 step() – Phase 추적 · 센서 감속 · 가속 한계 통합
    # ───────────────────────────────────────────────────────────
    def step(self, speed_factor: float = 1.0):
        # ── 0. ARRIVED 상태면 더 계산하지 않고 바로 리턴 ───────────
        if self._phase == "ARRIVED":
            return
        # --------------------------------------------------------------

        """
        1) 현재 세그먼트·진행률 → 목표 수평·수직 속도 계산
        2) 센서 거리( seg F 구간 )에 따라 감속/HOLD
        3) 최대 가속도(a_max) 제약으로 current_speed 조정
        4) dt 만큼 진행(progress)하고 위치·Phase 갱신
        """
        # ---- 1. 센서 감속 계수 ---------------------------------------
        sensor_factor = 1.0
        seg_id = self.mission_segments[self.seg_idx].segment_id.upper()

        if seg_id == "F":
            d   = self.sensor_distance
            ttc = self.sensor_ttc

            # 충돌 예상 → 즉시 HOLD
            if (d is not None and d < COLLISION_EXPECT_DIST_M) or \
               (ttc is not None and ttc < COLLISION_EXPECT_TTC_S):
                sensor_factor = 0.0        # HOLD

            # 충돌 위험 → 감속
            elif (d is not None and d < COLLISION_RISK_DIST_M) or \
                 (ttc is not None and ttc < COLLISION_RISK_TTC_S):
                sensor_factor = 0.7        # Decel
                
        # ---- 2. 목표 effective speed -------------------------------
        time_factor = max(speed_factor, 0.0)
        desired_speed = (
            self.current_desired_effective_speed() *
            sensor_factor
        )

        # ---- 3. 가속도 제한 -----------------------------------------
        effective_dt = self.dt * time_factor
        max_delta = self.a_max * effective_dt
        delta = max(-max_delta,
                    min(max_delta, desired_speed - self.current_speed))
        self.current_speed += delta

        # ---- 4. 진행도 & 위치 갱신 ----------------------------------
        self.progress = min(self.progress + self.current_speed * effective_dt,
                            self._total_length)

        # 세그먼트 경계 통과 시 Phase 업데이트
        while (self.seg_idx < len(self._cum_dist) and
               self.progress >= self._cum_dist[self.seg_idx]):
            self.seg_idx += 1
            if self.seg_idx < len(self.mission_segments):
                self.cur_segment = self.mission_segments[self.seg_idx]

        self.position = self.get_position(self.progress)

        # Phase 문자열 저장 → _phase
        self._phase = ("ARRIVED"
                       if self.progress >= self._total_length - 1
                       else self.cur_segment.segment_id.upper())


    @property
    def phase(self) -> str:
        """현재 Mission Segment ID(B~J) 또는 ARRIVED 반환 (read-only)"""
        return self._phase
