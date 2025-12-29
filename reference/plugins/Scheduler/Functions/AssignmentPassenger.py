# Scheduling/Functions/AssignmentPassenger.py
# ============================================================================
# • Vertiport 목록 : Monitoring/Sources/vertiport.csv  (두 번째 컬럼)
# • 출·도착 비율   : GUI가 넘겨준 departure / arrival CSV **그대로 사용**
#   └ 둘 중 하나라도 없으면 locations 수만큼 1/n 균등 비율 CSV 자동 생성
# • 하드코딩 테이블 완전 제거
# ============================================================================

from __future__ import annotations
import csv
from pathlib import Path
from typing import List, Dict

import pandas as pd


class AssignmentPassenger:
    """Vertiport-filtered UAM OD 수요 배분 클래스 (CSV 버전)."""

    # ------------------------------------------------------------------
    def __init__(self,
                 dep_csv: str | Path,
                 arr_csv: str | Path,
                 vert_csv: str | Path) -> None:

        # ── 경로 정리 ----------------------------------------------------
        self.dep_csv  = Path(dep_csv).expanduser().resolve()
        self.arr_csv  = Path(arr_csv).expanduser().resolve()
        self.vert_csv = Path(vert_csv).expanduser().resolve()

        if not self.vert_csv.is_file():
            raise FileNotFoundError(f"Vertiport CSV 없음 → {self.vert_csv}")

        # ── Vertiport 목록 ----------------------------------------------
        self.locations: List[str] = self._read_vertiport_names(self.vert_csv)

        # ── Ratio CSV 없으면 균등 CSV 생성 -------------------------------
        if not self.dep_csv.is_file() or not self.arr_csv.is_file():
            self._create_uniform_ratio_csvs()

        # ── Ratio CSV 로드 ----------------------------------------------
        self.departure_ratios = self._load_ratios(self.dep_csv)
        self.arrival_ratios   = self._load_ratios(self.arr_csv)
        self._validate_lengths()

        # print("▶ Vertiport names:", self.locations)          # __init__ 끝
        # print("▶ Departure ratios:", self.departure_ratios)  # _load_ratios 끝

    # ==================================================================
    # public
    # ==================================================================
    def plan_traffic(self, total_people: int) -> Dict[str, Dict[str, int]]:
        """total_people 명을 OD 테이블로 분배."""
        plan: Dict[str, Dict[str, int]] = {}
        n = len(self.locations)

        for i, ori in enumerate(self.locations):
            ori_total = total_people * self.departure_ratios[i]
            denom = sum(self.arrival_ratios[j] for j in range(n) if j != i)

            plan[ori] = {}
            for j, dest in enumerate(self.locations):
                if i == j:
                    plan[ori][dest] = 0
                else:
                    share = self.arrival_ratios[j] / denom
                    plan[ori][dest] = round(ori_total * share)

        return plan

    # ==================================================================
    # helpers
    # ==================================================================
    @staticmethod
    def _read_vertiport_names(csv_file: Path) -> List[str]:
        """vertiport.csv → 두 번째 컬럼에 포트 이름이 있다고 가정."""
        df = pd.read_csv(csv_file, encoding="utf-8-sig", usecols=[1])
        names = (
            df.iloc[:, 0]
              .dropna()
              .astype(str)
              .str.replace(r"\s*·\s*", "·", regex=True)
              .str.strip()
              .tolist()
        )
        if not names:
            raise ValueError(f"Vertiport CSV에 이름이 없습니다 → {csv_file}")
        return names

    # ------------------------------------------------------------------
    def _create_uniform_ratio_csvs(self) -> None:
        """departure/arrival CSV 가 없을 때 1/n 균등 비율로 자동 생성."""
        n = len(self.locations)
        if n == 0:
            raise RuntimeError("Vertiport 목록이 비어 있어 Ratio CSV를 만들 수 없습니다.")

        self.dep_csv.parent.mkdir(parents=True, exist_ok=True)

        with self.dep_csv.open("w", newline="", encoding="utf-8-sig") as f_dep, \
             self.arr_csv.open("w", newline="", encoding="utf-8-sig") as f_arr:
            w_dep, w_arr = csv.writer(f_dep), csv.writer(f_arr)
            w_dep.writerow(["Location", "Ratio"])
            w_arr.writerow(["Location", "Ratio"])
            ratio = 1.0 / n
            for loc in self.locations:
                w_dep.writerow([loc, ratio])
                w_arr.writerow([loc, ratio])

    # ------------------------------------------------------------------
    def _load_ratios(self, csv_path: Path) -> List[float]:
        """
        CSV → Vertiport 이름↔비율 매핑 후 self.locations 순서대로 리스트 반환.
        허용 헤더(대/소문자 무시):
          • 이름 컬럼 : origin / destination / location / vertiport
          • 비율 컬럼 : ratio
        """

        df = pd.read_csv(csv_path, encoding="utf-8-sig").dropna(how="all")

        # ratio 컬럼 찾기
        ratio_col = next((c for c in df.columns if c.strip().lower() == "ratio"), None)
        if ratio_col is None:
            raise ValueError(f"{csv_path} 에 'ratio' 컬럼이 없습니다.")

        # 이름 컬럼 찾기
        key_col = next(
            (c for c in df.columns
             if c.strip().lower() in {"origin", "destination", "location", "vertiport"}),
            df.columns[0],   # 못 찾으면 첫 컬럼 사용
        )

        # mapping : 이름 → float(ratio)
        mapping: Dict[str, float] = {
            str(r[key_col]).strip(): float(r[ratio_col])
            for _, r in df.iterrows()
            if pd.notna(r[key_col]) and pd.notna(r[ratio_col])
        }

        return [mapping.get(loc, 0.0) for loc in self.locations]

    # ------------------------------------------------------------------
    def _validate_lengths(self) -> None:
        if not (
            len(self.locations)
            and len(self.locations) == len(self.departure_ratios) == len(self.arrival_ratios)
        ):
            raise ValueError("Vertiport 수와 ratio 행 수가 일치하지 않습니다.")
