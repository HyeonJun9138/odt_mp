# ── 패키지 환경 보정 ────────────────────────────────────────────────
if __package__ is None:
    import os, sys
    pkg_root = os.path.dirname(os.path.dirname(__file__))  # …/TrafficSim
    sys.path.insert(0, pkg_root)
    __package__ = "Monitoring"

# ── 표준 / PyQt 모듈 ───────────────────────────────────────────────
import os
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QTabWidget

# ── 프로젝트 모듈 ─────────────────────────────────────────────────
from Functions.PathPlanning import PathPlanner
from Tabs import MonitoringTab
from pathlib import Path

# CSV 경로 (프로젝트 구조에 맞게 필요 시 수정)
_THIS_DIR = Path(__file__).resolve().parent
_SRC_DIR = (_THIS_DIR / "Sources").resolve()
_VP_CSV  = str(_SRC_DIR / "vertiport.csv")
_WP_CSV  = str(_SRC_DIR / "waypoint.csv")

# ── 메인 윈도우 ───────────────────────────────────────────────────
class MainWindow(QMainWindow):
    def __init__(self, fpl_dir=None):
        super().__init__()
        self.setWindowTitle("UAM Traffic GUI Prototype")
        self.resize(1800, 1000)

        # 0) 공통 PathPlanner 하나만 생성해 모든 탭이 공유
        planner = PathPlanner(_VP_CSV, _WP_CSV)

        # 1) 탭 인스턴스
        self.monitoring_tab = MonitoringTab(fpl_dir=fpl_dir)
        # self.analytic_tab   = AnalyticTab()
        # self.congestion_tab = CongestionTab(planner)

        # # 2) 시그널 연결
        # # Monitoring → Analytic (패킷 수신 / 삭제)
        # self.monitoring_tab.packetReceived.connect(
        #     self.analytic_tab.process_new_data_packet
        # )
        # self.monitoring_tab.vehicleRemoved.connect(
        #     self.analytic_tab.remove_vehicle
        # )

        # Monitoring → Congestion (실시간 혼잡 텍스트)
        # self.congestion_tab.bind_traffic_source(
        #     self.monitoring_tab.trafficUpdated
        # )

        # 3) QTabWidget 구성
        tabs = QTabWidget(movable=True, tabsClosable=False)
        tabs.addTab(self.monitoring_tab, "Monitoring")
        # tabs.addTab(self.analytic_tab,   "Analytic")
        # tabs.addTab(self.congestion_tab, "Congestion")
        self.setCentralWidget(tabs)

# ── entrypoint ───────────────────────────────────────────────────
def main():
    app = QApplication(sys.argv)
    win = MainWindow(fpl_dir=os.environ.get("FLEETOPS_FPL_DIR"))
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
