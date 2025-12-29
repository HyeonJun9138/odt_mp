# sitl_sim.py
# ────────────────────────────────────────────────────────────
# • FPL 폴더 내 모든 CSV → FlightPlan → MissionProfile → AircraftAgent
# • Start/Stop, 배속(sim_speed)·가상시간(sim_time) 관리
# • step(real_elapsed_sec) : 실제 경과시간 → 가상시간(dt * sim_speed) 로 환산
# • snapshot() : {id:{x,y,z,phase,…}} 딕셔너리 반환 (GUI 연동용)
#   └─ 동일 ID가 여러 구간(leg)을 순차 수행할 수 있도록 구현
# ────────────────────────────────────────────────────────────


import glob, math, datetime as dt, sys
from pathlib import Path
from typing import List, Dict
import csv
import pandas as pd

# ── 프로젝트 루트(…/TrafficSim_System)를 import 경로에 추가 ─────────────
ROOT_DIR = Path(__file__).resolve().parents[1]        # SITL/.. → TrafficSim_System
sys.path.append(str(ROOT_DIR))

LOG_FILE = Path(__file__).with_name("track_log.csv")
SCHEDULE_LOG_FILE = Path(__file__).with_name("schedule_checking.csv")

_log_header_written = False
_schedule_log_header_written = False

from Monitoring.Functions.MissionProfile import MissionSegment, MissionProfile
from Monitoring.Functions.AircraftAgent  import AircraftAgent
from Monitoring.Functions.geo_utils import lonlat_to_xy, xy_to_lonlat

# ────────────────────────────────────────────────────────────
# 헬퍼 : 시각 파싱 / 좌표 변환
# ────────────────────────────────────────────────────────────
def _parse_hms(timestr: str) -> dt.time:
    """
    '06:30' 또는 '06:30:28' → datetime.time
    """
    timestr = timestr.strip()
    for fmt in ("%H:%M:%S", "%H:%M"):
        try:
            return dt.datetime.strptime(timestr, fmt).time()
        except ValueError:
            continue
    raise ValueError(f"Bad time string: {timestr}")

lonlat_to_xy_m = lonlat_to_xy

# ────────────────────────────────────────────────────────────
# FlightPlan – CSV 한 행 ↔ MissionProfile·AircraftAgent 래퍼
# ────────────────────────────────────────────────────────────
class FlightPlan:
    def __init__(self, row: pd.Series, dt_sec: float):
        self.id: str   = str(row["ID"])
        self.ftype     = str(row.get("Type", "tiltrotor")).lower()
        self.pax: int  = int(row.get("Pax", 0))
        self.local_id  = str(row.get("LocalID", ""))
        self.origin    = str(row["From"])
        self.dest      = str(row["To"])
        self.std_str   = str(row["STD"])
        self.sta_str   = str(row["STA"])
        self.std: dt.datetime | None = None   # 로딩 후 설정
        self.atd: dt.datetime | None = None
        
        # ▼▼▼ [ADD] CSV 출발 게이트 정보 보관 (없으면 None) ▼▼▼
        try:
            self.dep_fato_no = int(row.get("DepFATO_No")) if str(row.get("DepFATO_No")).strip() not in ("", "nan", "None") else None
        except Exception:
            self.dep_fato_no = None
        try:
            self.dep_gate_no = int(row.get("DepGate_No")) if str(row.get("DepGate_No")).strip() not in ("", "nan", "None") else None
        except Exception:
            self.dep_gate_no = None
            
        try:
            self.arr_fato_no = int(row.get("ArrFATO_No")) if str(row.get("ArrFATO_No")).strip() not in ("", "nan", "None") else None
        except Exception:
            self.arr_fato_no = None
        try:
            self.arr_gate_no = int(row.get("ArrGate_No")) if str(row.get("ArrGate_No")).strip() not in ("", "nan", "None") else None
        except Exception:
            self.arr_gate_no = None
        # ▲▲▲ [ADD END] ▲▲▲

        # ── 세그먼트 파싱 ───────────────────────────────────────
        segments: List[MissionSegment] = []
        last_point: Dict[str, float] | None = None
        idx = 1
        while f"Seg{idx}" in row:
            cell = row[f"Seg{idx}"]
            if pd.isna(cell):
                break
            for token in (t.strip() for t in str(cell).split(";") if t.strip()):
                seg_id, lon, lat, lane = self._token_to_parts(token)   # ★ lane 받기
                x, y = lonlat_to_xy_m(float(lon), float(lat))
                start_pt = {"x": last_point["x"], "y": last_point["y"]} if last_point else None
                segment = MissionSegment(
                    seg_id,
                    end_point={"x": x, "y": y},
                    lane_type=lane,
                    start_point=start_pt
                )
                segments.append(segment)
                last_point = {"x": x, "y": y}
            idx += 1

        self.profile = MissionProfile(segments)
        self.agent   = AircraftAgent(segments, dt_sec,
                                     flight_type=self._infer_ftype())
        self.active  = False

        # 누적 세그먼트 길이와 로깅 인덱스 (schedule_checking.csv용)
        self._segment_boundaries: list[float] = []
        acc = 0.0
        for seg_info in self.agent.segments_info:
            acc += seg_info["length"]
            self._segment_boundaries.append(acc)
        self._next_segment_log_idx: int = 0

        def _pick_cell(*names):
            for n in names:
                if n in row and not pd.isna(row[n]):
                    s = str(row[n]).strip()
                    if s != "":
                        return s
            return None

        self.dep_gate_no = _pick_cell("DepGate_No", "DepGate", "Gate_No", "Gate")
        self.dep_fato_no = _pick_cell("DepFATO_No", "DepFATO", "FATO_No", "FATO")

    # .........................................................
    def _infer_ftype(self) -> str:
        if "lift"  in self.ftype: return "lift_and_cruise"
        if "multi" in self.ftype: return "multirotor"
        return "tiltrotor"

    # ------------------------------------------------------------------
    # FlightPlan._token_to_parts
    # ------------------------------------------------------------------
    @staticmethod
    def _token_to_parts(token: str):
        """
        토큰 형식
            "F : 126.931483 37.492144 L-1000"
            "E : 126.931483 37.492144"
        반환
            (seg_id,  lon, lat, lane_type or None)
        """
        seg_part, xy = token.split(":")
        parts = xy.strip().split()

        if len(parts) < 2:
            raise ValueError(f"Bad segment token: {token!r}")

        lon, lat = parts[0], parts[1]
        lane     = parts[2] if len(parts) >= 3 else None
        return seg_part.strip(), lon, lat, lane


    # .........................................................
    def maybe_activate(self, now: dt.datetime):
        if (not self.active) and now >= self.std:
            self.active = True
            self.atd    = now

    # .........................................................
    def step(self):
        if self.active:
            self.agent.step()

    # .........................................................
    def is_arrived(self) -> bool:
        return self.agent.phase == "ARRIVED"

    # .........................................................
    def snapshot(self, now: dt.datetime) -> dict:
        ag   = self.agent
        pos  = ag.position
        lon, lat = xy_to_lonlat(pos["x"], pos["y"])

        total   = ag.get_total_route_length()
        rem_m   = max(total - ag.progress, 0)
        rem_sec = ag.estimate_remaining_time()
        eta_str = (now + dt.timedelta(seconds=rem_sec)
                   ).strftime("%H:%M:%S") if rem_sec > 0 else "-"

        return {
            "id": self.id,
            "lat": lat, "lon": lon, "alt_m": pos["z"],
            "x": pos["x"], "y": pos["y"], "z": pos["z"],
            "heading_deg": ag.get_heading_deg(),
            "progress_m": ag.progress,
            "remain_m":   rem_m,
            "phase":      ag.phase,
            "lane":       getattr(ag.cur_segment, "lane_type", None),
            "pax":        self.pax,
            "local_id":   self.local_id,
            "atd": self.atd.strftime("%H:%M:%S") if self.atd else "-",
            "eta": eta_str,

            # ▼▼▼ [ADD] 출발 버티포트/게이트 정보 ▼▼▼
            "From":        self.origin,
            "To":          self.dest,
            "DepFATO_No":  self.dep_fato_no,
            "DepGate_No":  self.dep_gate_no,
            "ArrFATO_No":  self.arr_fato_no,
            "ArrGate_No":  self.arr_gate_no,
            # ▲▲▲ [ADD END] ▲▲▲
        }

    # ----------------------------------------------------------


# ────────────────────────────────────────────────────────────
# SitlSim – 가상시간·플릿 관리 (멀티-레그 지원)
# ────────────────────────────────────────────────────────────
class SitlSim:
    def __init__(self, fpl_folder: str | Path, dt_sim: float = 1.0):
        """
        fpl_folder 안의 *.csv 모두 로드.
        동일 ID가 여러 번 등장해도 STD 순으로 차례대로 수행.
        """
        self.dt_sim     = dt_sim
        self.sim_speed  = 1.0
        self.sim_time: dt.datetime | None = None
        self.running    = False

        self._fleet:   Dict[str, FlightPlan] = {}   # 현재 비행 중
        self._pending: List[FlightPlan]      = []   # STD 전 대기 (모든 leg)
        self._load_fpl_folder(fpl_folder, dt_sim)
        self._last_logged_time: dict[str, tuple[str, str]] = {}

    # ----------------------------------------------------------
    def _load_fpl_folder(self, folder: str | Path, dt_sec: float):
        csv_files = glob.glob(str(Path(folder) / "*.csv"))
        dfs = [pd.read_csv(f) for f in csv_files]
        df_all = pd.concat(dfs, ignore_index=True)

        today = dt.date.today()
        for _, row in df_all.iterrows():
            fp = FlightPlan(row, dt_sec)
            fp.std = dt.datetime.combine(today, _parse_hms(fp.std_str))
            self._pending.append(fp)

        # STD 오름차순 정렬(활성·큐잉 시 효율 ↑)
        self._pending.sort(key=lambda x: x.std)

    # ----------------------------------------------------------
    # GUI에서 호출
    # ----------------------------------------------------------
    def start(self, operation_start_str: str, sim_speed: float = 1.0):
        """operation_start_str 예: '06:30'"""
        
        # ▼ ① 이전 로그 삭제 + 헤더 플래그 리셋
        global _log_header_written
        _log_header_written = False
        global _schedule_log_header_written
        _schedule_log_header_written = False
        if LOG_FILE.exists():
            LOG_FILE.unlink()
        if SCHEDULE_LOG_FILE.exists():
            SCHEDULE_LOG_FILE.unlink()
        t0 = dt.datetime.combine(dt.date.today(),
              dt.datetime.strptime(operation_start_str, "%H:%M").time())
        self.sim_time  = t0
        self.sim_speed = sim_speed
        self.running   = True
        self._last_logged_time.clear()
    def stop(self):
        self.running = False

    # ----------------------------------------------------------
    def _activate_ready_plans(self):
        """
        • STD 도달 & 해당 ID가 비행 중이 아닐 때만 활성화
        • 동일 ID 중복 투입 방지
        """
        ready: List[FlightPlan] = []
        for fp in self._pending:
            if self.sim_time >= fp.std and fp.id not in self._fleet:
                ready.append(fp)

        for fp in ready:
            fp.maybe_activate(self.sim_time)
            self._fleet[fp.id] = fp
            self._pending.remove(fp)

    # ----------------------------------------------------------
    def _remove_arrived_aircraft(self):
        """ARRIVED 기체를 플릿에서 제거 (다음 leg는 pending에 이미 존재)"""
        done_ids = [fid for fid, fp in self._fleet.items() if fp.is_arrived()]
        for fid in done_ids:
            del self._fleet[fid]

    # ----------------------------------------------------------
    # ────────────────────────────────────────────────────────────
    # NEW — F-segment “front-distance” 계산
    # ────────────────────────────────────────────────────────────
    def _update_sensor_distances(self):
        """
        순항(F) 기체끼리 ‘같은 항로(lane_type·고도)’ & 앞쪽 관계를 찾고,
        거리[m] + Time-to-Collision[s] 를 에이전트에 주입한다.
        """
        if len(self._fleet) < 2:
            for fp in self._fleet.values():
                fp.agent.update_sensor(None, None)
            return

        # 1) 모든 위치·속도 캐시
        data = {}
        for fp in self._fleet.values():
            ag = fp.agent
            if ag.cur_segment.segment_id.upper() != "F":
                continue
            hdg_rad = math.radians(ag.get_heading_deg())
            vx = math.cos(hdg_rad) * ag.current_speed
            vy = math.sin(hdg_rad) * ag.current_speed
            data[fp.id] = {
                "agent": ag,
                "pos":   ag.position,
                "vel":   (vx, vy),
                "lane":  getattr(ag.cur_segment, "lane_type", None),
                "alt":   getattr(ag.cur_segment, "ending_altitude", None)
            }

        # 2) 각 기체별 가장 위험한 선행기 탐색
        for id_i, info_i in data.items():
            ag_i   = info_i["agent"]
            pos_i  = info_i["pos"]
            vx_i, vy_i = info_i["vel"]
            lane_i = info_i["lane"]
            alt_i  = info_i["alt"]

            best_d   = None
            best_ttc = None

            # 헤딩 단위벡터
            speed_i = math.hypot(vx_i, vy_i)
            if speed_i < 1e-3:
                hx, hy = 0.0, 0.0
            else:
                hx, hy = vx_i / speed_i, vy_i / speed_i

            for id_j, info_j in data.items():
                if id_j == id_i:
                    continue
                if info_j["lane"] != lane_i:
                    continue
                alt_j = info_j["alt"]
                if alt_i is not None and alt_j is not None \
                   and abs(alt_i - alt_j) > 1000:          # 1 000 ft
                    continue

                pos_j = info_j["pos"]
                dx = pos_j["x"] - pos_i["x"]
                dy = pos_j["y"] - pos_i["y"]

                # 내 앞쪽인가?
                if dx * hx + dy * hy <= 0:
                    continue
                d_horiz = math.hypot(dx, dy)

                # 상대 속도
                vx_rel = info_j["vel"][0] - vx_i
                vy_rel = info_j["vel"][1] - vy_i
                closing_rate = -(dx * vx_rel + dy * vy_rel) / max(d_horiz, 1e-3)
                if closing_rate <= 0:
                    ttc = None   # 멀어지거나 평행
                else:
                    ttc = d_horiz / closing_rate

                # 가장 ‘위험’(거리 최소) 선행기 채택
                if (best_d is None) or (d_horiz < best_d):
                    best_d   = d_horiz
                    best_ttc = ttc

            ag_i.update_sensor(best_d, best_ttc)

    # ────────────────────────────────────────────────────────────
    # UPDATED — 가상시간 한 tick 수행
    # ────────────────────────────────────────────────────────────
    def step(self, real_elapsed_sec: float):
        if (not self.running) or (self.sim_time is None):
            return

        # ① 가상 시계 advance
        sim_elapsed = real_elapsed_sec * self.sim_speed
        self.sim_time += dt.timedelta(seconds=sim_elapsed)

        # ② STD past → activate (ID 중복 체크 포함)
        self._activate_ready_plans()

        # ③ dt_sim 단위 tick 수행 -----------------------------------
        ticks    = int(sim_elapsed // self.dt_sim)
        leftover = sim_elapsed - ticks * self.dt_sim

        for _ in range(ticks):
            self._update_sensor_distances()     # ←★ 센서 갱신
            for fp in self._fleet.values():
                fp.step()

        if leftover > 1e-6:
            # 남은 시간도 같은 논리로 처리
            self._update_sensor_distances()
            ratio = leftover / self.dt_sim
            for fp in self._fleet.values():
                fp.agent.step(speed_factor=ratio)

        # 세그먼트 도착 시각 기록
        self._log_segment_arrivals()

        # ④ ARRIVED 기체 정리
        self._remove_arrived_aircraft()


    # ----------------------------------------------------------
    def snapshot(self) -> Dict[str, Dict]:
        data = {fid: fp.snapshot(self.sim_time) for fid, fp in self._fleet.items()}

        # ▼ CSV 로깅 (sim_time, id, lat, lon, alt_m, x, y, z, lane)
        global _log_header_written
        with LOG_FILE.open("a", newline="") as f:
            w = csv.writer(f)
            if not _log_header_written:
                w.writerow([
                    "sim_time","reg","phase","lat","lon","alt_m",
                    "x","y","z","lane"
                ])
                _log_header_written = True

            sim_str = (self.sim_time.strftime("%H:%M:%S.%f")[:-3]
                        if self.sim_time else "-")
            for pkt in data.values():
                reg = pkt.get("reg") or pkt.get("local_id") or pkt.get("id")
                phase = pkt.get("phase") or "-"
                key = (sim_str, phase)
                if self._last_logged_time.get(reg) == key:
                    continue

                w.writerow([
                    sim_str,
                    reg,
                    phase,
                    pkt["lat"], pkt["lon"], pkt["alt_m"],
                    pkt["x"], pkt["y"], pkt["z"],
                    pkt.get("lane") or "-"
                ])
                self._last_logged_time[reg] = key

        return data

    # ----------------------------------------------------------
    def get_spacing_info(self) -> dict[str, tuple[float | None, float | None]]:
        """
        {ACID: (distance_m, ttc_s)} 반환
          • distance_m : 앞 기체까지 수평 거리 (None = 정보 없음)
          • ttc_s      : Time-to-Collision (None = 충돌 경로 아님)
        """
        out = {}
        for fid, fp in self._fleet.items():
            ag = fp.agent
            out[fid] = (ag.sensor_distance, ag.sensor_ttc)
        return out

    # ----------------------------------------------------------
    def _log_segment_arrivals(self):
        """세그먼트 종료 시각을 schedule_checking.csv 에 저장."""
        if self.sim_time is None or not self._fleet:
            return

        rows: list[list] = []
        sim_str = self.sim_time.strftime("%H:%M:%S")

        for fp in self._fleet.values():
            boundaries = getattr(fp, "_segment_boundaries", [])
            seg_idx = getattr(fp, "_next_segment_log_idx", 0)
            while seg_idx < len(boundaries) and \
                  fp.agent.progress >= boundaries[seg_idx] - 1e-3:
                # 도달한 세그먼트 정보
                segments = fp.profile.get_segments()
                if seg_idx >= len(segments):
                    break
                segment = segments[seg_idx]
                lon, lat = xy_to_lonlat(segment.end_point["x"], segment.end_point["y"])
                lane = segment.lane_type or "-"

                rows.append([
                    sim_str,
                    fp.id,
                    fp.local_id or fp.id,
                    seg_idx + 1,
                    segment.segment_id,
                    lane,
                    lon,
                    lat,
                ])

                seg_idx += 1

            fp._next_segment_log_idx = seg_idx

        if not rows:
            return

        global _schedule_log_header_written
        with SCHEDULE_LOG_FILE.open("a", newline="") as f:
            w = csv.writer(f)
            if not _schedule_log_header_written:
                w.writerow([
                    "sim_time",
                    "flight_id",
                    "local_id",
                    "segment_order",
                    "segment_id",
                    "lane",
                    "lon",
                    "lat",
                ])
                _schedule_log_header_written = True

            w.writerows(rows)

# ────────────────────────────────────────────────────────────
# 단순 로컬 테스트
# ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import time
    scheduler_root = Path(__file__).resolve().parents[1] / "Scheduler" / "FPL_Result"
    if scheduler_root.exists():
        folder_candidates = sorted((p for p in scheduler_root.iterdir() if p.is_dir()), reverse=True)
        default_folder = folder_candidates[0] if folder_candidates else scheduler_root
    else:
        default_folder = scheduler_root
    FOLDER = default_folder
    sim = SitlSim(FOLDER, dt_sim=1.0)
    sim.start("06:30", sim_speed=1.0)     # 10× 배속

    while True:                    # Ctrl-C 로 종료
        sim.step(0.1)

        # ── 간격 & TTC 로그 ───────────────────────────────────
        info = sim.get_spacing_info()
        spacing_str = ", ".join(
            (
                f"{aid}: d={d:6.1f}m"
                + (f" t={t:4.1f}s" if t is not None else " t=-")
            ) if d is not None else f"{aid}: -"
            for aid, (d, t) in info.items()
        )

        print(sim.sim_time.time(), "| active:", len(sim._fleet),
              "|", spacing_str)
        time.sleep(0.1)