# Scheduling/Functions/PassengerTimeScheduler.py
# ============================================================================
# rev-5  (2025-05-08)
#   • 기본 excel_dir = Scheduling/Sources
#   • 중복 import 정리, mplcursors 선택적 로드
#   • 코드 스타일 통일
# ============================================================================

from __future__ import annotations
import datetime as _dt
from pathlib import Path as _Path
from typing import Dict, List

import numpy as _np
import pandas as _pd
import matplotlib.pyplot as _plt
import matplotlib.dates  as _mdates

try:
    import mplcursors
    _HAS_MPLCURSORS = True
except ModuleNotFoundError:
    _HAS_MPLCURSORS = False

# 한글 폰트 및 음수 기호 (Windows)
_plt.rcParams["font.family"]      = "Malgun Gothic"
_plt.rcParams["axes.unicode_minus"] = False

__all__ = ["DemandProfile", "PassengerTimeScheduler"]

# ----------------------------------------------------------------------------
# 1. DemandProfile  (월‧요일‧시간 가중치 로더)
# ----------------------------------------------------------------------------
class DemandProfile:
    """월-요일-시간대 가중치 테이블을 읽어 확률분포를 생성."""

    _FILE_DAY = "TrafficDemand_DayofWork.csv"
    _FILE_MON = "TrafficDemand_Monthly.csv"
    _FILE_TIM = "TrafficDemand_Timeline.csv"

    def __init__(self, excel_dir: str | _Path | None = None):
        # 기본 경로 = Scheduling/Sources
        base = (
            _Path(excel_dir).expanduser().resolve()
            if excel_dir
            else _Path(__file__).resolve().parents[1] / "Sources"
        )
        self._assert_exists(base)

        df_day = _pd.read_csv(base / self._FILE_DAY, encoding="utf-8-sig")
        df_mon = _pd.read_csv(base / self._FILE_MON, encoding="utf-8-sig")
        df_tim = _pd.read_csv(base / self._FILE_TIM, encoding="utf-8-sig")

        # → 가중치 시리즈/배열로 변환 (전체 합 1)
        self.day_weights = (
            df_day.set_index("시점수")["유입비율"].div(100).rename(lambda x: x.lower())
        )
        self.month_weights = (
            df_mon.set_index("지점 시작")["유입비율"].div(100).rename(lambda x: x[:3].lower())
        )
        self.timeline_weights = df_tim["유입비율"].div(100).to_numpy()  # len == 24

    # -----------------------------------------------------------------
    def _assert_exists(self, base: _Path) -> None:
        missing = [fn for fn in (self._FILE_DAY, self._FILE_MON, self._FILE_TIM)
                   if not (base / fn).is_file()]
        if missing:
            raise FileNotFoundError(
                f"[DemandProfile] 다음 파일을 찾지 못했습니다 → {base}\n - "
                + "\n - ".join(missing)
            )

    # -----------------------------------------------------------------
    def scale_count(self, raw: int, date: _dt.date) -> int:
        """월·요일 가중치 적용 후 반올림"""
        m_key = date.strftime("%b").lower()     # jan …
        d_key = date.strftime("%A").lower()     # monday …
        m_norm = self.month_weights[m_key] / self.month_weights.mean()
        d_norm = self.day_weights[d_key] / self.day_weights.mean()
        return max(1, round(raw * m_norm * d_norm))

    # -----------------------------------------------------------------
    def sample_times(self, n: int, date: _dt.date,
                     seed: int | None = None) -> List[_dt.datetime]:
        """24-시 블록 가중치대로 무작위 n개 시각 반환"""
        rng = _np.random.default_rng(seed)
        hours   = rng.choice(24, size=n, p=self.timeline_weights)
        minutes = rng.integers(0, 60, size=n)
        seconds = rng.integers(0, 60, size=n)
        day0 = _dt.datetime.combine(date, _dt.time())
        return [day0 + _dt.timedelta(hours=int(h), minutes=int(m), seconds=int(s))
                for h, m, s in zip(hours, minutes, seconds)]

# ----------------------------------------------------------------------------
# 2. PassengerTimeScheduler
# ----------------------------------------------------------------------------
class PassengerTimeScheduler:
    """AssignmentPassenger 결과 + DemandProfile 로 도착 시각 부여."""

    def __init__(self,
                 assignment_plan: Dict[str, Dict[str, int]],
                 locations: List[str],
                 profile: DemandProfile):
        self.assignment_plan = assignment_plan
        self.locations       = locations
        self.profile         = profile
        self.passenger_info: Dict[str, List[dict]] = {o: [] for o in locations}

    # -----------------------------------------------------------------
    def assign_arrival_times(self, date: _dt.date,
                             seed: int | None = None) -> Dict[str, List[dict]]:
        """OD-수요 전체에 시간 할당 → self.passenger_info 반환"""
        for origin, dest_map in self.assignment_plan.items():
            records: List[dict] = []
            for dest, cnt_raw in dest_map.items():
                if origin == dest or cnt_raw == 0:
                    continue
                cnt_scaled = self.profile.scale_count(cnt_raw, date)
                times = self.profile.sample_times(cnt_scaled, date, seed)
                records += [
                    {"id": "", "origin": origin, "destination": dest, "arrival_time": t}
                    for t in times
                ]
            # 정렬 + ID 부여
            records.sort(key=lambda r: r["arrival_time"])
            for idx, rec in enumerate(records, 1):
                rec["id"] = f"{origin}_{idx}"
            self.passenger_info[origin] = records
        return self.passenger_info

    # -----------------------------------------------------------------
    def plot_gantt_for_origin(self, origin: str) -> None:
        """하루치 승객 도착 분포를 Gantt(Scatter)로 시각화."""
        data = self.passenger_info.get(origin) or []
        if not data:
            print(f"[plot_gantt_for_origin] '{origin}' 데이터가 없습니다.")
            return

        times = [r["arrival_time"] for r in data]
        yvals = list(range(1, len(times) + 1))

        fig, ax = _plt.subplots(figsize=(10, 6))
        sc = ax.scatter(times, yvals, marker="|", linewidths=1)
        ax.set_xlabel("도착 시간"); ax.set_ylabel("승객 순번")
        ax.set_title(f"{origin} 버티포트 1일 도착 분포 (n={len(times)})")
        ax.xaxis.set_major_formatter(_mdates.DateFormatter("%H:%M"))
        _plt.tight_layout()

        if _HAS_MPLCURSORS:
            cur = mplcursors.cursor(sc, hover=True)

            @cur.connect("add")
            def _on_add(sel):
                rec = data[sel.index]
                tstr = rec["arrival_time"].strftime("%H:%M:%S")
                sel.annotation.set_text(
                    f"ID: {rec['id']}\n시간: {tstr}\n목적지: {rec['destination']}"
                )

        _plt.show()

# ----------------------------------------------------------------------------
# 3. Quick test
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    from AssignmentPassenger import AssignmentPassenger

    total_people = 108000
    today        = _dt.date.today()

    planner = AssignmentPassenger()
    plan    = planner.plan_traffic(total_people)

    profile   = DemandProfile()   # ← 기본경로 자동
    scheduler = PassengerTimeScheduler(plan, planner.locations, profile)
    scheduler.assign_arrival_times(today, seed=42)

    origin = planner.locations[0]
    print(f"{origin} sample:", scheduler.passenger_info[origin][:5])
    scheduler.plot_gantt_for_origin(origin)
