from __future__ import annotations
# -*- coding: utf-8 -*-
"""
UAM_Path2Sim.py
-----------------------------------------------------------
1) vertiport.xlsx / waypoint.xlsx → PathPlanner
2) start, end Vertiport 이름 지정 → 최단 경로(raw) → rebuild_route(full)
3) full 경로  →  MissionSegment(B~J) 자동 매핑
4) MissionProfile → AircraftAgent 시뮬레이션 → 3-D 애니메이션
"""
from typing import Tuple
import pandas as pd
import os, math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.animation import FuncAnimation

try:
    # 패키지 내부 호출(권장)
    from .PathPlanning   import PathPlanner, rebuild_route
    from .MissionProfile import MissionSegment, MissionProfile
    from .AircraftAgent  import AircraftAgent
    from .UAMParameters  import THRESHOLD_DECEL, THRESHOLD_HOLD
except ImportError:
    # 단독 실행 시: sys.path 패치 → 절대 import
    import sys, pathlib
    THIS = pathlib.Path(__file__).resolve()
    PKG  = THIS.parent          # …/Monitoring/Functions
    ROOT = PKG.parent           # …/Simulation
    for p in (PKG, ROOT):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    from GateResources import load_resources, find_lonlat, lonlat_to_xy_m 
    from PathPlanning   import PathPlanner, rebuild_route
    from MissionProfile import MissionSegment, MissionProfile
    from AircraftAgent  import AircraftAgent
    from UAMParameters  import THRESHOLD_DECEL, THRESHOLD_HOLD
# ───────────────────────────────────────────────────────────────────────

# ----------------------------------------------------------------------
# 0. 경로 → MissionSegment 변환 함수
# ----------------------------------------------------------------------
SEG_RULES = [
    (r'.*_TO_1$',  "C"),
    (r'.*_TO_2$',  "D"),
    (r'^CL\d+_.+$',"E"),
    (r'^WP.*',     "F"),
    (r'.*_LD_2$',  "G"),
    (r'.*_LD_1$',  "H"),
]

# ----------------------------------------------------------------------
# 0. 경로 → MissionProfile  (우측 주행‧차선 선택 완전판)
# ----------------------------------------------------------------------

def inject_ground_and_fato(profile, planner, dep_port: str, arr_port: str,
                           dep_gate_no: int, dep_fato_no: int,
                           arr_fato_no: int, arr_gate_no: int,
                           res_csv_path: str, taxi_minutes: float = 5.0):
    """
    MissionProfile(B~J) → B/I/J FATO-XY 보정 + A/K 삽입(각 5분)
    """
    df = load_resources(res_csv_path)

    # 1) FATO/GATE lon/lat
    lonF_dep, latF_dep = find_lonlat(df, dep_port, "FATO", dep_fato_no)
    lonG_dep, latG_dep = find_lonlat(df, dep_port, "GATE", dep_gate_no)
    lonF_arr, latF_arr = find_lonlat(df, arr_port, "FATO", arr_fato_no)
    lonG_arr, latG_arr = find_lonlat(df, arr_port, "GATE", arr_gate_no)

    # 2) lon/lat → 내부 평면 XY[m] (각 포트 중심을 기준)
    refDep_lon, refDep_lat = planner.nodes_geo[dep_port]
    refDep_xkm, refDep_ykm = planner.nodes[dep_port]
    xf_dep, yf_dep = lonlat_to_xy_m(lonF_dep, latF_dep, refDep_lon, refDep_lat, refDep_xkm, refDep_ykm)
    xg_dep, yg_dep = lonlat_to_xy_m(lonG_dep, latG_dep, refDep_lon, refDep_lat, refDep_xkm, refDep_ykm)

    refArr_lon, refArr_lat = planner.nodes_geo[arr_port]
    refArr_xkm, refArr_ykm = planner.nodes[arr_port]
    xf_arr, yf_arr = lonlat_to_xy_m(lonF_arr, latF_arr, refArr_lon, refArr_lat, refArr_xkm, refArr_ykm)
    xg_arr, yg_arr = lonlat_to_xy_m(lonG_arr, latG_arr, refArr_lon, refArr_lat, refArr_xkm, refArr_ykm)

    # 3) 세그먼트 보정 & 삽입
    segs = profile.get_segments()

    # (B) 출발 FATO 보정
    if segs and segs[0].segment_id == "B":
        segs[0].end_point = {"x": xf_dep, "y": yf_dep}

    # (I,J) 도착 FATO 보정
    if len(segs) >= 2 and segs[-2].segment_id == "I" and segs[-1].segment_id == "J":
        segs[-2].end_point = {"x": xf_arr, "y": yf_arr}   # I 종료점=FATO
        segs[-1].end_point = {"x": xf_arr, "y": yf_arr}   # J 전체=FATO

    # (A) Gate→FATO 세그먼트(5분)
    A = MissionSegment("A",
        end_point  = {"x": xf_dep, "y": yf_dep},
        start_point= {"x": xg_dep, "y": yg_dep},
        duration_override_sec = taxi_minutes*60)

    # (K) FATO→Gate 세그먼트(5분)
    K = MissionSegment("K",
        end_point = {"x": xg_arr, "y": yg_arr},
        duration_override_sec = taxi_minutes*60)

    new_segments = [A] + segs + [K]
    return MissionProfile(new_segments)


def path_to_profile(path, node_xy,
                    dep_ground: dict | None = None,
                    arr_ground: dict | None = None,
                    taxi_out_min: float = 5.0,
                    taxi_in_min:  float = 5.0):
    """
    full_path → MissionProfile
    ───────────────────────────────────────────────────────────────
    · alt_ft : 최초 헤딩이 0°~179°(동쪽) → 1000 | 180°~359°(서쪽) → 2000
    · side   : 각 링크(prev→curr) 가 y+ 방향(북상) → 'R' | y- 방향(남하) → 'L'
               (항로는 남→북으로 그려져 있고, 비행체는 '우측 주행' 규칙)
    · lane_type = f"{side}-{alt_ft}"   모든 MissionSegment 로 전파
    """
    # ── 1. alt_ft 결정 (항공편 전 구간 동일) ──────────────────────
    def _xy(n):            # name/tuple → (xkm, ykm)
        return node_xy[n] if isinstance(n, str) else n

    if len(path) < 2:
        raise ValueError("경로 길이가 2 이상이어야 합니다.")
    x0, y0 = _xy(path[0]);  x1, y1 = _xy(path[1])
    init_hdg = (math.degrees(math.atan2(y1 - y0, x1 - x0)) + 360) % 360
    alt_ft   = 1000 if 0 <= init_hdg < 180 else 2000

    # ── 2. 세그먼트 변환 루프 ────────────────────────────────────
    phase = "TAKEOFF"
    segs: list[MissionSegment] = []
    last_id = None
    prev_coord = _xy(path[0])

    for i, node in enumerate(path):
        is_str = isinstance(node, str);  name = node if is_str else ""
        cur_coord = _xy(node)

        # ── side(R/L) 결정 : y 증감으로 판단 ────────────────
        dy = cur_coord[1] - prev_coord[1]
        side = "R" if dy >= 0 else "L"
        lane_type = f"{side}-{alt_ft}"

        # ── phase 전환 규칙 (기존 로직 그대로) ────────────────
        if phase == "TAKEOFF":
            if is_str and name.startswith("CL"):
                phase = "CRUISE"
        elif phase == "CRUISE":
            if is_str and name.startswith("CL"):
                phase = "ARRIVE_G"
        elif phase == "ARRIVE_G":
            if is_str and name.endswith("_LD_2"):
                phase = "ARRIVE_H"
        elif phase == "ARRIVE_H":
            if is_str and name.endswith("_LD_1"):
                phase = "FINAL"

        # ── seg_id 매핑 (기존 로직) ──────────────────────────
        if i == 0:                 seg_id = "B"
        elif i == len(path) - 1:   seg_id = "J"
        elif phase == "TAKEOFF":
            if is_str and name.endswith("_TO_1"): seg_id = "C"
            elif is_str and name.endswith("_TO_2"): seg_id = "D"
            else:                                  seg_id = "E"
        elif phase == "CRUISE":     seg_id = "F"
        elif phase == "ARRIVE_G":   seg_id = "G"
        elif phase == "ARRIVE_H":   seg_id = "H"
        else:                       seg_id = "I"

        # ── I/J 처리 (착지점 두 번) ────────────────────────
        if seg_id == "J" and last_id not in ("I", "J"):
            seg_id = "I"

        if seg_id == "I" and i == len(path) - 1:
            xkm, ykm = cur_coord
            segs.append(MissionSegment("I",
                        end_point={"x": xkm*1_000, "y": ykm*1_000},
                        lane_type=lane_type))
            segs.append(MissionSegment("J",
                        end_point={"x": xkm*1_000, "y": ykm*1_000},
                        lane_type=lane_type))
            break

        # ── 중복 제거 (B,C,D,I) ───────────────────────────
        if seg_id in ("B", "C", "D", "I") and seg_id == last_id:
            prev_coord = cur_coord; continue

        # ── 세그먼트 생성 (km→m 변환) ─────────────────────
        xkm, ykm = cur_coord
        segs.append(MissionSegment(seg_id,
                    end_point={"x": xkm*1_000, "y": ykm*1_000},
                    lane_type=lane_type))
        last_id = seg_id
        prev_coord = cur_coord

    # ── (A) 출발 FATO 보정 + A 세그먼트 삽입 ──────────────────
    if dep_ground and segs and segs[0].segment_id == "B":
        fxm = dep_ground["fato_xy_km"][0]*1_000
        fym = dep_ground["fato_xy_km"][1]*1_000
        # B의 XY를 FATO로
        segs[0].end_point["x"] = fxm; segs[0].end_point["y"] = fym
        # A 삽입 (Gate→FATO)
        gxm = dep_ground["gate_xy_km"][0]*1_000
        gym = dep_ground["gate_xy_km"][1]*1_000
        segs.insert(0, MissionSegment("A",
                      end_point={"x": fxm, "y": fym},
                      duration_sec=taxi_out_min*60,
                      start_point={"x": gxm, "y": gym}))

    # ── (B) 도착 FATO 보정 + K 세그먼트 추가 ──────────────────
    if arr_ground:
        fxm = arr_ground["fato_xy_km"][0]*1_000
        fym = arr_ground["fato_xy_km"][1]*1_000
        for s in segs:
            if s.segment_id in ("I","J"):
                s.end_point["x"] = fxm; s.end_point["y"] = fym
        # K 추가(FATO→Gate)
        gxm = arr_ground["gate_xy_km"][0]*1_000
        gym = arr_ground["gate_xy_km"][1]*1_000
        segs.append(MissionSegment("K",
                    end_point={"x": gxm, "y": gym},
                    duration_sec=taxi_in_min*60,
                    start_point={"x": fxm, "y": fym}))

    return MissionProfile(segs)


# ----------------------------------------------------------------------
# 1. Path → MissionProfile
# ----------------------------------------------------------------------
def generate_profile(vp_file, wp_file, start_port, end_port):
    planner = PathPlanner(vp_file, wp_file)
    dist, prev = planner.dijkstra(start_port, end_port)
    if math.isinf(dist[end_port]):
        raise RuntimeError("경로를 찾을 수 없습니다.")
    raw   = planner.reconstruct(prev, start_port, end_port)
    full  = rebuild_route(planner, raw)
    prof  = path_to_profile(full, planner.nodes)
    return prof, full, planner

# ----------------------------------------------------------------------
# 2. 시뮬레이션
# ----------------------------------------------------------------------
def plot_trajectory(trajectory):
    """정적 3-D 궤적 한 눈에 보기 (창 닫으면 코드 계속 진행)."""
    xs = [pt["x"] for pt in trajectory]
    ys = [pt["y"] for pt in trajectory]
    zs = [pt["z"] for pt in trajectory]

    fig = plt.figure()
    ax  = fig.add_subplot(111, projection="3d")
    ax.plot(xs, ys, zs, lw=2, color="magenta", label="Path")
    ax.scatter(xs[0], ys[0], zs[0], c="green", s=50, label="Start")
    ax.scatter(xs[-1], ys[-1], zs[-1], c="red",   s=50, label="End")
    ax.set_title("Full 3-D Flight Path")
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
    ax.legend(); plt.tight_layout(); plt.show()

def simulate_agent(agent, total_time, dt):
    steps = int(total_time / dt)
    traj  = []; t = 0.0
    for _ in range(steps):
        agent.step(); t += dt
        p = agent.position.copy(); p["t"] = t
        traj.append(p)
    return traj

def get_sector_polygon(cx,cy,cz,heading,r,half=5,n=20):
    angles = np.linspace(heading - math.radians(half),
                         heading + math.radians(half), n)
    arc = [(cx + r*math.cos(a), cy + r*math.sin(a), cz) for a in angles]
    return [(cx,cy,cz)] + arc + [(cx,cy,cz)]

def animate(traj, dt):
    xs,ys,zs = [ [pt[k] for pt in traj] for k in ('x','y','z')]
    ts       = [pt['t'] for pt in traj]

    hspd, vspd = [0], [0]
    for i in range(1,len(traj)):
        dx,dy,dz = xs[i]-xs[i-1], ys[i]-ys[i-1], zs[i]-zs[i-1]
        hspd.append(math.hypot(dx,dy)/dt)
        vspd.append(dz/dt)

    fig = plt.figure(figsize=(10,12))
    gs  = fig.add_gridspec(3,1)
    ax3d= fig.add_subplot(gs[0,0], projection='3d')
    ax3d.set_title("UAM Mission 3-D")
    point, = ax3d.plot([],[],[],'o',color='blue'); line, = ax3d.plot([],[],[],'-',color='magenta')
    warn = Poly3DCollection([],facecolor='red',alpha=0.3);  ax3d.add_collection3d(warn)
    caut = Poly3DCollection([],facecolor='orange',alpha=0.3);ax3d.add_collection3d(caut)
    margin=100; ax3d.set_xlim(min(xs)-margin,max(xs)+margin); ax3d.set_ylim(min(ys)-margin,max(ys)+margin)
    ax3d.set_zlim(min(zs)-margin,max(zs)+margin); ax3d.set_xlabel("X"); ax3d.set_ylabel("Y"); ax3d.set_zlabel("Z")

    ax_h=fig.add_subplot(gs[1,0]); ax_v=fig.add_subplot(gs[2,0])
    ax_h.set_title("Horizontal Speed"); ax_v.set_title("Vertical Speed")
    lh,=ax_h.plot([],[],'g-'); lv,=ax_v.plot([],[],'m-')
    ax_h.set_xlim(ts[0],ts[-1]); ax_v.set_xlim(ts[0],ts[-1])
    ax_h.set_ylim(0,max(hspd)*1.2 or 1); ax_v.set_ylim(-max(map(abs,vspd))*1.2 or 1,
                                                       max(map(abs,vspd))*1.2 or 1)

    def upd(i):
        point.set_data([xs[i]], [ys[i]])
        point.set_3d_properties([zs[i]])
        line.set_data(xs[:i+1],ys[:i+1]); line.set_3d_properties(zs[:i+1])
        lh.set_data(ts[:i+1],hspd[:i+1]); lv.set_data(ts[:i+1],vspd[:i+1])

        if i>0:
            hd=math.atan2(ys[i]-ys[i-1], xs[i]-xs[i-1])
        else: hd=0
        warn.set_verts([get_sector_polygon(xs[i],ys[i],zs[i],hd,THRESHOLD_HOLD)])
        caut.set_verts([get_sector_polygon(xs[i],ys[i],zs[i],hd,THRESHOLD_DECEL)])
        return point,line,lh,lv,warn,caut

    anim = FuncAnimation(fig, upd, frames=len(traj), interval=50, blit=False)
    plt.tight_layout(); plt.show()
    return anim  # ←★ 반환

# ----------------------------------------------------------------------
# ───────────────────────────────────────────────────────────
#  main : 프로그램 시작점 (요약 → Enter → 시뮬레이션)
# ───────────────────────────────────────────────────────────
# ───────────────────────────────────────────────────────────
#  main : ① 정적 3-D 전체 경로 → 창을 닫으면 ② 애니메이션
# ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    from pathlib import Path
    SIM_ROOT   = Path(__file__).resolve().parents[1]   # …/Simulation
    DATA_DIR   = SIM_ROOT / "Sources"
    vp_xlsx    = DATA_DIR / "vertiport.csv"
    wp_xlsx    = DATA_DIR / "waypoint.csv"

    START_PORT = "가산·대림"          # ← 필요하면 수정
    END_PORT   = "잠실"

    # 0) 경로·MissionProfile 생성
    profile, full_path, planner = generate_profile(vp_xlsx, wp_xlsx,
                                                   START_PORT, END_PORT)

    # 1) MissionProfile·Waypoints 요약 콘솔 출력
    print("\n=== Mission Profile ===")
    for seg in profile.get_segments():
        print(seg)

    # 2) 시뮬레이션(trajectory) 미리 계산
    DT, TOTAL_T, FTYPE = 0.5, 5000, "tiltrotor"
    agent = AircraftAgent(profile.get_segments(), DT, FTYPE, initial_progress=0.0)
    traj  = simulate_agent(agent, TOTAL_T, DT)

    # 3) 전체 궤적(정적 3-D) 먼저 보여주기
    plot_trajectory(traj)       # 창 닫을 때까지 블로킹

    # 4) 창을 닫은 뒤 애니메이션 실행
    anim = animate(traj, DT)    # 실시간 3-D + 속도 그래프

