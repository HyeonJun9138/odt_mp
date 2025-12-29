"""
MissionProfile.py

이 모듈은 UAM 운용 시 적용할 Mission Profile을 구성하기 위한 클래스들을 정의합니다.
MissionSegment는 개별 미션 단계(예: Hover Climb, Transition + Climb, ... Hover Descend)를 나타내며,
각 세그먼트는 종료점(x, y)만 입력받고, 목표 수직/수평 속도와 종료 AGL 고도(단위: ft)는 내부 DEFAULTS에서 자동으로 설정됩니다.
MissionProfile은 여러 MissionSegment를 순서대로 연결하여 전체 운용 경로(waypoint 리스트)를 생성합니다.
"""

import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 3D 플롯 지원

class MissionSegment:
    # 각 세그먼트에 대한 기본 설정 (미션 프로필 표에 기초)
    DEFAULTS = {
    "B": {"target_vertical_speed": 500, "target_horizontal_speed":   0, "ending_altitude":  15},
    "C": {"target_vertical_speed": 500, "target_horizontal_speed":  60, "ending_altitude": 300},
    "D": {"target_vertical_speed":   0, "target_horizontal_speed": 120, "ending_altitude": 300},
    "E": {"target_vertical_speed": 500, "target_horizontal_speed": 125, "ending_altitude":1000},
    "F": {"target_vertical_speed":   0, "target_horizontal_speed": 130, "ending_altitude":1000},
    "G": {"target_vertical_speed": 500, "target_horizontal_speed": 125, "ending_altitude": 300},
    "H": {"target_vertical_speed": 500, "target_horizontal_speed": 120, "ending_altitude": 300},
    "I": {"target_vertical_speed": 400, "target_horizontal_speed":  60, "ending_altitude":  65},
    "J": {"target_vertical_speed": 300, "target_horizontal_speed":   0, "ending_altitude":   0},    
}
    
#     DEFAULTS = {
#     "A": {"target_vertical_speed": 0, "target_horizontal_speed":   0, "ending_altitude":  0},
#     "B": {"target_vertical_speed": 500, "target_horizontal_speed":   0, "ending_altitude":  15},
#     "C": {"target_vertical_speed": 500, "target_horizontal_speed":  60, "ending_altitude": 300},
#     "D": {"target_vertical_speed":   0, "target_horizontal_speed": 120, "ending_altitude": 300},
#     "E": {"target_vertical_speed": 500, "target_horizontal_speed": 125, "ending_altitude":1000},
#     "F": {"target_vertical_speed":   0, "target_horizontal_speed": 130, "ending_altitude":1000},
#     "G": {"target_vertical_speed": 500, "target_horizontal_speed": 125, "ending_altitude": 300},
#     "H": {"target_vertical_speed": 500, "target_horizontal_speed": 120, "ending_altitude": 300},
#     "I": {"target_vertical_speed": 400, "target_horizontal_speed":  60, "ending_altitude":  65},
#     "J": {"target_vertical_speed": 300, "target_horizontal_speed":   0, "ending_altitude":   0},
#     "K": {"target_vertical_speed": 0, "target_horizontal_speed":   0, "ending_altitude":  0},
# }
    def __init__(self, segment_id: str, end_point: dict[str, float],
                    segment_name: str | None = None, lane_type: str | None = None,
                 start_point: dict[str, float] | None = None,
                 duration_override_sec: float | None = None):
        """
        segment_id : "B"~"J"
        end_point  : {"x": …, "y": …}
        lane_type  : "L-1000" / "R-1000" / "L-2000" / "R-2000"  (선택)
                     └─ 경로 계획 단계(path_to_profile)에서 1회만 지정해 주면
                        모든 F-세그먼트가 동일 lane 으로 가도록 전파된다.
        """
        self.segment_id   = segment_id.upper()
        self.segment_name = segment_name if segment_name else self.segment_id
        self.end_point    = end_point
        self.lane_type    = lane_type                 # ② 저장
        self.duration_override_sec = duration_override_sec        # ← 추가
        self.start_point  = start_point         # ← 추가

        # 기본 프로파일 값
        defs = MissionSegment.DEFAULTS.get(
            self.segment_id,
            {"target_vertical_speed": 0,
             "target_horizontal_speed": 0,
             "ending_altitude": 0},
        )
        self.target_vertical_speed  = defs["target_vertical_speed"]
        self.target_horizontal_speed = defs["target_horizontal_speed"]
        self.ending_altitude        = defs["ending_altitude"]

        # ③ F-구간은 lane_type 으로 고도(1000/2000 ft) 덮어쓰기
        if self.segment_id == "F" and lane_type:
            try:
                alt_ft = int(lane_type.split("-")[1])
                self.ending_altitude = alt_ft
            except (ValueError, IndexError):
                pass  # 잘못된 형식이면 무시

    def __repr__(self):
        return (
            f"<MissionSegment {self.segment_id} {self.segment_name}"
            f" | EndPoint: {self.end_point}"
            f", Lane: {self.lane_type}"
            f", V_speed: {self.target_vertical_speed} ft/min"
            f", H_speed: {self.target_horizontal_speed} kt"
            f", Ending Altitude: {self.ending_altitude} ft>"
        )


class MissionProfile:
    def __init__(self, segments):
        """
        생성자:
            segments: MissionSegment 객체들의 리스트 (예: [segB, segC, ..., segJ])
                      리스트 순서대로 미션이 진행됩니다.
        """
        self.segments = segments

    def get_segments(self):
        """미션 프로필에 포함된 MissionSegment 리스트 반환"""
        return self.segments

    def __repr__(self):
        seg_str = "\n".join(repr(seg) for seg in self.segments)
        return f"MissionProfile:\n{seg_str}"

    def generate_waypoints(self):
        """
        각 MissionSegment 종료점을 순서대로 연결해 waypoint 리스트 반환.
        ● lane_type ─ "L-1000" / "R-1000" / "L-2000" / "R-2000"
          · L / R  : 경로 기준 왼쪽 / 오른쪽 차선 (좌·우 100 m 오프셋)
          · 1000 / 2000 : 크루즈(F) 세그먼트 목표 고도(ft)
        """
        waypoints = []
        ft_to_m = 0.3048
        offset_distance = 100.0  # 차선 폭(좌·우 오프셋) [m]

        # ── 1. 출발점(B) 추가 ──────────────────────────
        if self.segments:
            first = self.segments[0]
            if first.segment_id in ("A","B"):
                if first.segment_id == "A" and first.start_point:
                    waypoints.append({"x": first.start_point["x"], "y": first.start_point["y"], "z": 0.0})
                else:  # B 또는 start_point 미지정
                    waypoints.append({"x": first.end_point["x"], "y": first.end_point["y"], "z": 0.0})

        # ── 2. 세그먼트 순회 ───────────────────────────
        for seg in self.segments:
            seg_id = seg.segment_id
            wp_z   = seg.ending_altitude * ft_to_m

            # ── F-구간만 차선 오프셋 적용 ───────────────
            if seg_id == "F" and len(waypoints) > 0:
                prev = waypoints[-1]
                dx, dy = seg.end_point["x"] - prev["x"], seg.end_point["y"] - prev["y"]
                norm = math.hypot(dx, dy) or 1.0
                # 진행 방향 단위벡터
                ux, uy = dx / norm, dy / norm
                # 오른쪽(+)·왼쪽(-) 벡터
                right_x, right_y =  uy, -ux
                sign = 1.0 if (seg.lane_type or "").upper().startswith("R") else -1.0

                # 시작점 오프셋
                waypoints[-1] = {
                    "x": prev["x"] + sign * right_x * offset_distance,
                    "y": prev["y"] + sign * right_y * offset_distance,
                    "z": prev["z"],
                }
                # 종료점 오프셋
                waypoints.append(
                    {"x": seg.end_point["x"] + sign * right_x * offset_distance,
                     "y": seg.end_point["y"] + sign * right_y * offset_distance,
                     "z": wp_z}
                )
            else:
                # 그대로 추가
                waypoints.append(
                    {"x": seg.end_point["x"],
                     "y": seg.end_point["y"],
                     "z": wp_z}
                )
        return waypoints

    def plot_profile(self):
        """
        생성된 waypoint 리스트를 3D 플롯으로 시각화하는 테스트용 함수.
        각 waypoint 옆에 해당 세그먼트의 ID를 텍스트 레이블로 표시합니다.
        """
        waypoints = self.generate_waypoints()
        xs = [wp["x"] for wp in waypoints]
        ys = [wp["y"] for wp in waypoints]
        zs = [wp["z"] for wp in waypoints]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(xs, ys, zs, marker='o', linestyle='-', color='blue', label="Mission Path")
        ax.scatter(xs, ys, zs, color='red', s=50, label="Waypoints")
        # 레이블: 첫 waypoint가 지상이면 "B (Start)"로 표시 후, 각 세그먼트 ID 표시
        labels = []
        if self.segments and self.segments[0].segment_id == "B":
            labels.append("B (Start)")
            for seg in self.segments:
                labels.append(seg.segment_id)
        else:
            for seg in self.segments:
                labels.append(seg.segment_id)
        for i, label in enumerate(labels):
            ax.text(xs[i], ys[i], zs[i], f" {label}", fontsize=10, color="black")
        ax.set_title("3D Mission Profile")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.legend()
        plt.show()


if __name__ == "__main__":
    # 예시: 미션 프로필 단계(B~J) 생성 (입력은 오직 x,y 값만 제공합니다.)
    segments = [
        MissionSegment("B", end_point={"x": 1000, "y": 2000}),
        MissionSegment("C", end_point={"x": 3000, "y": 4000}),
        MissionSegment("D", end_point={"x": 5000, "y": 4000}),
        MissionSegment("E", end_point={"x": 7000, "y": 5000}),
        MissionSegment("F", end_point={"x": 12000, "y": 5000}),
        MissionSegment("G", end_point={"x": 14000, "y": 4000}),
        MissionSegment("H", end_point={"x": 16000, "y": 4000}),
        MissionSegment("I", end_point={"x": 18000, "y": 2500}),
        MissionSegment("J", end_point={"x": 18000, "y": 2500})
    ]
    mission_profile = MissionProfile(segments)
    print(mission_profile)
    print("생성된 Waypoints:")
    for wp in mission_profile.generate_waypoints():
        print(wp)
    # 테스트용 3D 플롯 호출
    mission_profile.plot_profile()
