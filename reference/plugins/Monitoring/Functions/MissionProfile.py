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
        "A": {"target_vertical_speed":   0, "target_horizontal_speed":   3, "ending_altitude":   0},  # ≈1.5 m/s 지상 이동
        "B": {"target_vertical_speed": 500, "target_horizontal_speed":   0, "ending_altitude":  15},
        "C": {"target_vertical_speed": 500, "target_horizontal_speed":  60, "ending_altitude": 450},  # 원래 300 임 
        "D": {"target_vertical_speed":   0, "target_horizontal_speed": 120, "ending_altitude": 450},
        "E": {"target_vertical_speed": 500, "target_horizontal_speed": 125, "ending_altitude":1000},
        "F": {"target_vertical_speed":   0, "target_horizontal_speed": 130, "ending_altitude":1000},
        "G": {"target_vertical_speed": 500, "target_horizontal_speed": 125, "ending_altitude": 450},
        "H": {"target_vertical_speed": 500, "target_horizontal_speed": 120, "ending_altitude": 450},
        "I": {"target_vertical_speed": 400, "target_horizontal_speed":  60, "ending_altitude":  65},
        "J": {"target_vertical_speed": 300, "target_horizontal_speed":   0, "ending_altitude":   0},
        "K": {"target_vertical_speed":   0, "target_horizontal_speed":   3, "ending_altitude":   0},  # ≈1.5 m/s 지상 이동
    }
    def __init__(
        self,
        segment_id: str,
        end_point: dict[str, float],
        segment_name: str | None = None,
        lane_type: str | None = None,          # ① ← 새 파라미터 (예: "R-1000")
        start_point: dict[str, float] | None = None,
    ):
        """
        segment_id : "A"~"K"
        end_point  : {"x": …, "y": …}
        lane_type  : "L-1000" / "R-1000" / "L-2000" / "R-2000"  (선택)
                     └─ 경로 계획 단계(path_to_profile)에서 1회만 지정해 주면
        start_point : (선택) 해당 세그먼트 시작 지점 {"x", "y"}.
                      지상 이동 구간(A, K 등)에서 실제 출발 위치를 지정할 때 사용.
                        모든 F-세그먼트가 동일 lane 으로 가도록 전파된다.
        """
        self.segment_id   = segment_id.upper()
        self.segment_name = segment_name if segment_name else self.segment_id
        self.end_point    = end_point
        self.lane_type    = lane_type                 # ② 저장
        self.start_point   = start_point

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
        start = f", Start: {self.start_point}" if self.start_point else ""
        return (
            f"<MissionSegment {self.segment_id} {self.segment_name}"
            f" | EndPoint: {self.end_point}"
            f"{start}"
            f", Lane: {self.lane_type}"
            f", V_speed: {self.target_vertical_speed} ft/min"
            f", H_speed: {self.target_horizontal_speed} kt"
            f", Ending Altitude: {self.ending_altitude} ft>"
        )


class MissionProfile:
    def __init__(self, segments):
        """
        생성자:
            segments: MissionSegment 객체들의 리스트 (예: [segA, segB, ..., segK])
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
        각 MissionSegment 종료점을 순서대로 연결해 waypoint 리스트를 반환.
        ● lane_type ─ "L-1000" / "R-1000" / "L-2000" / "R-2000"
          · L / R  : 경로 기준 왼쪽 / 오른쪽 차선 (좌·우 100 m 오프셋)
          · 1000 / 2000 : 크루즈(F) 세그먼트 목표 고도(ft)
        지상 세그먼트(A, K)는 start_point/end_point 조합으로 실제 택싱 경로를 구성한다.
        """
        waypoints: list[dict[str, float]] = []
        ft_to_m = 0.3048
        offset_distance = 100.0  # 차선 폭(좌·우 오프셋) [m]

        if not self.segments:
            return waypoints

        def append_wp(pt: dict[str, float]):
            if not waypoints:
                waypoints.append(pt)
                return
            last = waypoints[-1]
            if (abs(last["x"] - pt["x"]) < 1e-6 and
                abs(last["y"] - pt["y"]) < 1e-6 and
                abs(last["z"] - pt["z"]) < 1e-6):
                return
            waypoints.append(pt)

        first = self.segments[0]
        start_base = first.start_point or first.end_point
        append_wp({"x": start_base["x"], "y": start_base["y"], "z": 0.0})

        for seg in self.segments:
            seg_id = seg.segment_id
            wp_z   = seg.ending_altitude * ft_to_m

            if seg_id in {"A", "K"}:
                ground_start = seg.start_point
                if ground_start:
                    append_wp({"x": ground_start["x"],
                               "y": ground_start["y"],
                               "z": 0.0})
                append_wp({"x": seg.end_point["x"],
                           "y": seg.end_point["y"],
                           "z": 0.0})
                continue

            if seg_id == "F" and waypoints:
                prev = waypoints[-1]
                dx, dy = seg.end_point["x"] - prev["x"], seg.end_point["y"] - prev["y"]
                norm = math.hypot(dx, dy) or 1.0
                ux, uy = dx / norm, dy / norm
                right_x, right_y =  uy, -ux
                sign = 1.0 if (seg.lane_type or "").upper().startswith("R") else -1.0

                waypoints[-1] = {
                    "x": prev["x"] + sign * right_x * offset_distance,
                    "y": prev["y"] + sign * right_y * offset_distance,
                    "z": prev["z"],
                }
                append_wp({
                    "x": seg.end_point["x"] + sign * right_x * offset_distance,
                    "y": seg.end_point["y"] + sign * right_y * offset_distance,
                    "z": wp_z
                })
                continue

            append_wp({"x": seg.end_point["x"],
                       "y": seg.end_point["y"],
                       "z": wp_z})

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
        labels: list[str] = []
        if self.segments:
            first_id = self.segments[0].segment_id
            if first_id in {"A", "B"}:
                labels.append(f"{first_id} (Start)")
            else:
                labels.append(first_id)
            for seg in self.segments[1:]:
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
    # 예시: 미션 프로필 단계(A~K) 생성 (입력은 오직 x,y 값만 제공합니다.)
    segments = [
        MissionSegment("A", end_point={"x": 1000, "y": 2000}, start_point={"x": 0, "y": 0}),
        MissionSegment("B", end_point={"x": 1000, "y": 2000}),
        MissionSegment("C", end_point={"x": 3000, "y": 4000}),
        MissionSegment("D", end_point={"x": 5000, "y": 4000}),
        MissionSegment("E", end_point={"x": 7000, "y": 5000}),
        MissionSegment("F", end_point={"x": 12000, "y": 5000}),
        MissionSegment("G", end_point={"x": 14000, "y": 4000}),
        MissionSegment("H", end_point={"x": 16000, "y": 4000}),
        MissionSegment("I", end_point={"x": 18000, "y": 2500}),
        MissionSegment("J", end_point={"x": 18000, "y": 2500}),
        MissionSegment("K", end_point={"x": 19000, "y": 1500}, start_point={"x": 18000, "y": 2500})
    ]
    mission_profile = MissionProfile(segments)
    print(mission_profile)
    print("생성된 Waypoints:")
    for wp in mission_profile.generate_waypoints():
        print(wp)
    # 테스트용 3D 플롯 호출
    mission_profile.plot_profile()
