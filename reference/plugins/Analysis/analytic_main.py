#!/usr/bin/env python
from __future__ import annotations
import sys, socket, json, traceback
from typing import Dict
import os

os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = "--disable-gpu"
os.environ["QTWEBENGINE_DISABLE_SANDBOX"] = "1"

from PyQt5.QtCore    import QTimer, QThread, pyqtSignal
from PyQt5.QtWidgets import QApplication, QMainWindow, QTabWidget

from Tabs.congestion_tab  import CongestionTab
from Tabs.noise_tab       import NoiseTab
from Tabs.main_tab import MainTab

# ──────────────────────────────────────────────────────────
# UDP 수신 스레드
# ──────────────────────────────────────────────────────────
class UdpRxThread(QThread):
    pkt = pyqtSignal(dict)            # 수신한 JSON 객체

    def __init__(self, port: int = 50052, parent=None):
        super().__init__(parent)
        self._running = True
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind(("127.0.0.1", port))
        self._sock.setblocking(False)

    # .....................................................
    def run(self):
        while self._running:
            try:
                data, addr = self._sock.recvfrom(65_535)

                if data.strip().upper() == b"PING":
                    self._sock.sendto(b"PONG", addr)
                    continue

                self.pkt.emit(json.loads(data.decode()))
            except BlockingIOError:
                pass
            except json.JSONDecodeError:
                pass
            except Exception:
                traceback.print_exc()
            self.msleep(20)                           # 50 fps

    def stop(self):
        self._running = False
        self.wait()

# ──────────────────────────────────────────────────────────
# 메인 윈도우
# ──────────────────────────────────────────────────────────
class MainWindow(QMainWindow):
    def __init__(self, listen_port: int = 50052):
        super().__init__()
        self.setWindowTitle("UAM Analytic Dashboard – 4 Tabs")
        self.resize(1600, 800)
        self._paused = False

        # ── 탭 위젯 ─────────────────────────────────────────
        self.tabs = QTabWidget(movable=True, tabsClosable=False, parent=self)

        self.noise_tab      = NoiseTab()
        self.congestion_tab = CongestionTab()
        self.main_tab       = MainTab()        # ★ 테스트 탭

        self.tabs.addTab(self.main_tab,       "Main")
        self.tabs.addTab(self.noise_tab,      "Noise")
        self.tabs.addTab(self.congestion_tab, "Congestion")

        self.setCentralWidget(self.tabs)

        # ── UDP 수신 스레드 ─────────────────────────────────
        self._rx = UdpRxThread(listen_port, self)
        self._rx.pkt.connect(self._on_packet)
        self._rx.start()

        # ── 5 s 타임아웃 관리 ──────────────────────────────
        self._last_seen: Dict[str, int] = {}
        self._gc_timer = QTimer(self)
        self._gc_timer.timeout.connect(self._gc)
        self._gc_timer.start(1000)            # 1 s

    # ------------------------------------------------------
    def _feed_all_tabs(self, vid: str, data: dict):
        for tab in (self.main_tab,
                    self.noise_tab,
                    self.congestion_tab):

            tab.process_new_data_packet(vid, data)
        self._last_seen[vid] = 0

    # ------------------------------------------------------
    def _gc(self):
        if self._paused:          # ◀ Pause 중이면 타이머·삭제 모두 건너뜀
            return
    
        for vid in list(self._last_seen.keys()):
            self._last_seen[vid] += 1
            if self._last_seen[vid] > 5:
                for tab in (self.main_tab,
                            self.noise_tab,
                            self.congestion_tab):
                    tab.remove_vehicle(vid)
                self._last_seen.pop(vid, None)

    # ------------------------------------------------------
    def _on_packet(self, pkt: dict):
        if "fleet" in pkt:
            # ① 시뮬레이션 시각을 꺼내 둔다
            sim_time = pkt.get("time")

            # 1) 시뮬 전체 Pause 신호 (bool)
        if "pause" in pkt:                 # ex) {"pause": true}
            self._paused = bool(pkt["pause"])
            return                         # 상태 전송 패킷이므로 데이터 없음

        # 2) 다중 기체
        if "fleet" in pkt:
            for vid, data in pkt["fleet"].items():
                # ② 각 서브패킷에 time 필드로 주입
                if sim_time is not None:
                    data["time"] = sim_time
                self._feed_all_tabs(vid, data)
                self._last_seen[vid] = 0   # 타임스탬프 초기화

        # 3) 단일 기체
        elif "id" in pkt:
            vid = pkt["id"]
            self._feed_all_tabs(vid, pkt)
            self._last_seen[vid] = 0

    # ------------------------------------------------------
    def closeEvent(self, ev):
        if self._gc_timer.isActive():
            self._gc_timer.stop()
        self._rx.stop()
        super().closeEvent(ev)

# ──────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=50052,
                    help="UDP listen port")
    args = ap.parse_args()

    app = QApplication(sys.argv)
    win = MainWindow(args.port)
    win.show()
    sys.exit(app.exec_())
    
    # ★ ① flush 함수 정의
    from Analysis.Functions.terrain_noise import NOISE_LOGGER
    def _flush_noise():
        # 1) 남아 있던 1분 버킷까지 큐에 밀어 넣기
        NOISE_LOGGER._q.put((NOISE_LOGGER._ref_time, NOISE_LOGGER._bucket))
        # 2) 종료 신호
        NOISE_LOGGER._q.put(None)
        # 3) 백그라운드 쓰레드가 파일 쓰기 마칠 때까지 기다림
        NOISE_LOGGER.join()
        
    # ★ ② 앱 종료 직전에 flush 호출하도록 연결
    app.aboutToQuit.connect(_flush_noise)

    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
