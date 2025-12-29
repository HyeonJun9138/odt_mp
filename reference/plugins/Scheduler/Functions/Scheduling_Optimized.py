# Scheduling_Optimized.py
# ============================================================================
#  ❱ 한 Origin(버티포트)에 대해
#      1) 이상적 스케줄(wait=15)
#      2) 빈좌석 최소 wait* 탐색
#      3) FATO=RUNWAY_COUNT 지상 시뮬
#      4) Gantt + 혼잡 곡선 비교
# ----------------------------------------------------------------------------
#  ※ 수정점
#     • PREP_TIME_MIN / TAKEOFF_MIN 상수화
#     • sort 키 버그(빈 timedelta) 수정
#     • 코드·주석 정리
# ============================================================================

from __future__ import annotations
import datetime as dt
from typing import List, Dict, Optional
import matplotlib.pyplot as plt
import matplotlib.dates  as mdates
import mplcursors
from matplotlib.patches import Rectangle
from pathlib import Path
# ------------------------------------------------------------------
# vertiport.csv  순서 → "01", "02", … 코드 부여
import csv, pathlib

_SCHED_DIR = Path(__file__).resolve().parents[1]
_SRC_DIR   = _SCHED_DIR / "Sources"

VP_CSV = _SRC_DIR / "vertiport.csv"

# ---------------- 좌석수 설정 한 곳에서 끝! ----------------
MAX_CAP = 6                     # ★ 여기만 9, 11, 12 …로 바꾸세요
BASE_CAPS = [4, 6]               # 고정 캡
SUPPORTED_CAPS = BASE_CAPS + [MAX_CAP]

SEATS = {f"{c}인승": c for c in SUPPORTED_CAPS}
CAP_MAX = MAX_CAP             # alias: 가독성용

def load_port_code(csv_path: pathlib.Path):
    with csv_path.open(encoding="utf-8-sig") as f:
        reader = csv.reader(f); header = next(reader)
        name_idx = next((i for i,h in enumerate(header)
                         if "vertiport" in h.lower()
                         or h.strip().lower() in ("name","vp","vertiport_nm")), 0)
        return {row[name_idx].strip(): f"{i+1:02d}"
                for i,row in enumerate(reader)
                if len(row)>name_idx and row[name_idx].strip()}
    
with VP_CSV.open(encoding="utf-8-sig") as f:
    reader = csv.reader(f)
    header = next(reader)

    # ── ★ 헤더 탐색 로직 개선  (부분 일치 'vertiport' 포함 여부) ──
    def is_name_col(h: str) -> bool:
        h_low = h.strip().lower()
        return (
            "vertiport" in h_low or   # Vertiport 명 / vertiport_name …
            h_low in ("name", "vp", "vertiport_nm")
        )

    name_idx = next((i for i, h in enumerate(header) if is_name_col(h)), 0)

    PORT_CODE = load_port_code(VP_CSV)
# ------------------------------------------------------------------


try:
    from .AssignmentPassenger        import AssignmentPassenger
    from .PassengerTimeScheduler     import DemandProfile, PassengerTimeScheduler
    from .LocationDistanceCalculator import haversine, location_coords
except ImportError:        # ← python Scheduling/Functions/Scheduling_Optimized.py
    import sys, pathlib
    THIS = pathlib.Path(__file__).resolve()
    PKG  = THIS.parent          # .../Scheduling/Functions
    ROOT = PKG.parent           # .../Scheduling
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from AssignmentPassenger        import AssignmentPassenger
    from PassengerTimeScheduler     import DemandProfile, PassengerTimeScheduler
    from LocationDistanceCalculator import haversine, location_coords

# ------------------------ 시뮬 상수 -----------------------------------------
DEFAULT_WAIT     = 5                    # 이상적 스케줄 wait 값
GRID_WAITS       = range(30, 361, 5)     # 빈좌석 최소 탐색 범위
GROUND_CUTOFF_KM = 7                     # 짧은 구간 제외

VERTIPORT_NAME   = "잠실"                # 분석 Origin
TOTAL_PEOPLE     = 108_169               # 입력 총 인원


# ------------------------ Queue 용 상수 ---------------------------------
RUNWAY_COUNT     = 1                     # 지상 시뮬 FATO 개수
TAKEOFF_MIN      = 2                     # FATO 이륙 점유 시간
PREP_TIME_MIN    = 5                     # 게이트 준비(승객 탑승) 시간


NUM_ARR_RUNWAYS  = 1     # 착륙-FATO 수 (필요 시 VP별 dict)
LANDING_MIN      = 2    # 착륙 패드 점유시간
APP_BUFFER_MIN   = 0     # Final-Approach 버퍼(전용 패드라 0)
# --------------------------- 1. 스케줄러 ------------------------------------
class RegularFlightScheduler:
    """passenger_info → 정기편 편성 (max_wait 규칙)"""

    def __init__(self, passenger_info: dict):
        self.passenger_info = passenger_info

    def schedule_flights_for_origin(
        self, origin: str, max_wait: int = DEFAULT_WAIT
    ) -> List[Dict]:
        flights: List[Dict] = []

        # 1) 목적지별 그룹
        od_groups: Dict[str, List[Dict]] = {}
        for rec in self.passenger_info.get(origin, []):
            dst = rec["destination"]
            if dst != origin:
                od_groups.setdefault(dst, []).append(rec)

        # 2) 각 OD 그룹 편성
        for dst, recs in od_groups.items():
            recs.sort(key=lambda r: r["arrival_time"])
            waiting: List[dt.datetime] = []
            last: Optional[dt.datetime] = None
            counter = 1

            for r in recs:
                t = r["arrival_time"]
                if not waiting:
                    waiting, last = [t], t
                    continue

                gap = (t - last).total_seconds() / 60
                if gap > max_wait or len(waiting) >= CAP_MAX:      # ★ 수정
                    flights.append(_mk_flight(origin, dst, counter, last, waiting))
                    counter += 1
                    waiting, last = [t], t
                else:
                    waiting.append(t)
                    last = t

            if waiting and last:
                flights.append(_mk_flight(origin, dst, counter, last, waiting))

        return flights


def _mk_flight(o: str, d: str, n: int,
               last: dt.datetime, waiting: List[dt.datetime]) -> Dict:
    dep = last + dt.timedelta(minutes=10)

    # -------- 가변 좌석 규칙 ----------------------------------
    caps_sorted = sorted(SUPPORTED_CAPS)         
    cap = next(c for c in caps_sorted if len(waiting) <= c)

    TYPE_STR = {v: k for k, v in SEATS.items()}   
    # ── NEW: 한글 대신 코드 사용 ─────────────────────────────
    o_code = PORT_CODE.get(o, "XX")
    d_code = PORT_CODE.get(d, "XX")
    flight_id = f"{o_code}{d_code}_{n:03d}"
    # ─────────────────────────────────────────────────────────

    return {
        "flight_number" : flight_id,
        "uam_id"        : None,
        "origin"        : o,
        "destination"   : d,
        "scheduled_time": dep,
        "passengers"    : len(waiting),
        "aircraft_type" : TYPE_STR[cap],      # ★ 변경
    }



# ------------------------ 2. 이륙 시뮬 ---------------------------------------
def simulate_ground_operations(
    flights: List[Dict],
    num_runways : int   = RUNWAY_COUNT,
    cut_km      : float = GROUND_CUTOFF_KM,
    taxi_out_min: int   = 5,
    takeoff_min : int   = TAKEOFF_MIN,
    **kwargs,
) -> List[Dict]:
    """
    • STD + taxi_out_min(게이트→FATO 이동) 후 대기열 진입
    • 활주로 Early-Finish 우선 배정(takeoff_min 분 점유)
    • OD 거리가 cut_km km 미만이면 제외
    • kwargs: 과거 코드의 prep_min 등은 무시(하위호환)
    """
    # 1) 거리컷 적용 -----------------------------------------------------
    valid: List[Dict] = []
    for f in flights:
        p1, p2 = location_coords.get(f["origin"]), location_coords.get(f["destination"])
        if not p1 or not p2:
            continue
        if haversine(p1[1], p1[0], p2[1], p2[0]) >= cut_km:
            valid.append(f)

    if not valid:
        return []

    # 2) 준비 완료 시각순 정렬 (STD + taxi_out)
    valid.sort(key=lambda x: x["scheduled_time"] + dt.timedelta(minutes=taxi_out_min))

    # 3) 활주로 Earliest-Finish 스케줄링 ---------------------------------
    runways = [valid[0]["scheduled_time"] + dt.timedelta(minutes=taxi_out_min)] * num_runways

    for f in valid:
        ready  = f["scheduled_time"] + dt.timedelta(minutes=taxi_out_min)
        f["etot_plan"] = ready + dt.timedelta(minutes=takeoff_min)

        # 규칙: (a) ready에 전 패드 비어있으면 T1부터
        #       (b) ready에 비어있는 패드 중 가장 낮은 번호
        #       (c) 그 외엔 '가장 빨리 비는' + '낮은 번호' 우선
        available = [i for i, r in enumerate(runways) if r <= ready]
        if all(r <= ready for r in runways):
            idx = 0
        elif available:
            idx = min(available)
        else:
            idx = min(range(len(runways)), key=lambda i: (runways[i], i))
        f["takeoff_pad"] = f"T{idx+1}"

        start  = max(ready, runways[idx])               # 늦은 쪽에서 시작
        finish = start + dt.timedelta(minutes=takeoff_min)
        runways[idx] = finish                           # 활주로 점유 갱신
        f["actual_takeoff_start"]  = start
        f["actual_takeoff_finish"] = finish

    return valid


# ------------------------ 착륙 시뮬 ---------------------------------------

def simulate_landing_ops(
    flights: List[Dict],
    num_runways: int = NUM_ARR_RUNWAYS, 
    buffer_min : int = APP_BUFFER_MIN,
    landing_min: int = LANDING_MIN
) -> List[Dict]:
    for f in flights:
        f["landing_ready_s"] = f["landing_ready"] - dt.timedelta(minutes=buffer_min)
    flights.sort(key=lambda x: x["landing_ready_s"])

    runways = [flights[0]["landing_ready_s"]] * num_runways
    for f in flights:
        ready  = f["landing_ready_s"]
        idx    = runways.index(min(runways))
        f["landing_pad"] = f"L{RUNWAY_COUNT + idx + 1}"
        start  = max(ready, runways[idx])
        finish = start + dt.timedelta(minutes=landing_min)
        runways[idx] = finish
        f["actual_touch"]    = start          # 접지 = STA Actual
        f["actual_shutdown"] = finish
    return flights


# ------------------------ 3. Plot 함수 ---------------------------------------
def plot_combined_charts(f_i: List[Dict], f_a: List[Dict],
                         origin: str, best_wait: int) -> None:
    dests = sorted({f["destination"] for f in f_i})
    ymap  = {d: i for i, d in enumerate(dests)}

    def iv(fs, key): return [(f[key],
                              f[key] + dt.timedelta(minutes=PREP_TIME_MIN)) for f in fs]

    def timeline(intervals):
        if not intervals:
            now = dt.datetime.now(); return [now], [0]
        s0 = min(s for s, _ in intervals); e0 = max(e for _, e in intervals)
        tp = [s0 + dt.timedelta(minutes=m)
              for m in range(int((e0-s0).total_seconds()//60)+1)]
        cong = [sum(1 for s, e in intervals if s <= t < e) for t in tp]
        return tp, cong

    iv_i, iv_a = iv(f_i, "scheduled_time"), iv(f_a, "actual_takeoff_start")
    tp_i, cong_i = timeline(iv_i);  tp_a, cong_a = timeline(iv_a)
    iv_q = [(f["scheduled_time"], f["actual_takeoff_start"]) for f in f_a]
    tp_q, cong_q = timeline(iv_q)

    import matplotlib.gridspec as gridspec
    fig = plt.figure(figsize=(20, 20))
    gs  = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 0.8])
    ax_i, ax_ic = fig.add_subplot(gs[0,0]), fig.add_subplot(gs[0,1])
    ax_a, ax_ac = fig.add_subplot(gs[1,0]), fig.add_subplot(gs[1,1])
    ax_q        = fig.add_subplot(gs[2,:])

    locator = mdates.MinuteLocator(byminute=range(0, 60, 30))
    fmt     = mdates.DateFormatter("%H:%M")
    for ax in (ax_i, ax_ic, ax_a, ax_ac, ax_q):
        ax.xaxis.set_major_locator(locator); ax.xaxis.set_major_formatter(fmt)
        ax.tick_params(axis="x", labelrotation=45)

    def gantt(ax, fs, color, title, key, width):
        patches = []
        for f in fs:
            x = mdates.date2num(f[key]); w = width/(24*60)
            r = Rectangle((x, ymap[f['destination']] - 0.3),
                          w, 0.6, facecolor=color, picker=True)
            r.flight_info = f
            ax.add_patch(r); patches.append(r)
        if patches:
            xs = [p.get_x() for p in patches]
            xe = [p.get_x() + p.get_width() for p in patches]
            ax.set_xlim(min(xs) - 0.001, max(xe) + 0.001)
        ax.set_yticks(list(ymap.values())); ax.set_yticklabels(list(ymap.keys()))
        mplcursors.cursor(patches, hover=True).connect(
            "add", lambda sel: sel.annotation.set_text(
                sel.artist.flight_info["flight_number"]))
        ax.set_title(title)

    seat_gap = lambda fs: sum(SEATS[f['aircraft_type']] - f['passengers']
                            for f in fs)

    gantt(ax_i, f_i, "tab:blue",
          f"{origin} 이상적 (wait=15)\n편수 {len(f_i)}, 빈좌석 {seat_gap(f_i)}",
          "scheduled_time", PREP_TIME_MIN)
    ax_ic.plot(tp_i, cong_i); ax_ic.set_title("이상적 혼잡")

    gantt(ax_a, f_a, "tab:green",
          f"{origin} 실제 (wait*={best_wait})\n편수 {len(f_a)}, 빈좌석 {seat_gap(f_a)}",
          "actual_takeoff_start", TAKEOFF_MIN)
    ax_ac.plot(tp_a, cong_a); ax_ac.set_title("실제 혼잡")

    ax_q.plot(tp_q, cong_q)
    ax_q.set_title("지상 대기 편수")
    fig.subplots_adjust(left=0.06, right=0.98, top=0.95,
                        bottom=0.06, wspace=0.25, hspace=0.35)
    plt.show()

# ------------------------ 4. 메인 -------------------------------------------
if __name__ == "__main__":
    # 1) 승객 생성 ----------------------------------------------------------
    planner  = AssignmentPassenger()
    plan     = planner.plan_traffic(TOTAL_PEOPLE)
    profile  = DemandProfile()
    pts      = PassengerTimeScheduler(plan, planner.locations, profile)
    p_info   = pts.assign_arrival_times(dt.date.today())

    # 2) 스케줄 ------------------------------------------------------------
    sched = RegularFlightScheduler(p_info)
    flights_ideal = sched.schedule_flights_for_origin(VERTIPORT_NAME, DEFAULT_WAIT)

    best_wait, best_empty, best_fl = None, float("inf"), []
    for w in GRID_WAITS:
        fl = sched.schedule_flights_for_origin(VERTIPORT_NAME, w)
        empty = sum(SEATS[f['aircraft_type']] - f['passengers'] for f in fl)
        if empty < best_empty:
            best_wait, best_empty, best_fl = w, empty, fl

    flights_actual = simulate_ground_operations(best_fl, RUNWAY_COUNT)

    # 3) 콘솔 요약 ---------------------------------------------------------
    seat_gap = lambda fs: sum(SEATS[f['aircraft_type']] - f['passengers']
                            for f in fs)
    print(f"\n[이상적] wait=15  → 편수 {len(flights_ideal):3d}, 빈좌석 {seat_gap(flights_ideal):4d}")
    print(f"[최적 ] wait*={best_wait:<3d} → 편수 {len(best_fl):3d}, 빈좌석 {best_empty:4d}")
    print(f"[실제 ] FATO {RUNWAY_COUNT}개 → 편수 {len(flights_actual):3d}, 빈좌석 {seat_gap(flights_actual):4d}\n")

    # 4) 시각화 -----------------------------------------------------------
    plot_combined_charts(flights_ideal, flights_actual, VERTIPORT_NAME, best_wait)
