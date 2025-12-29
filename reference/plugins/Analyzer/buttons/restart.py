#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys
from PyQt5.QtWidgets import QMessageBox, QApplication

def request_restart(parent):
    resp = QMessageBox.question(
        parent, "프로그램 재시작", "정말 재시작할까요?",
        QMessageBox.Yes | QMessageBox.No, QMessageBox.No
    )
    if resp != QMessageBox.Yes:
        return
    try:
        python = sys.executable
        os.execl(python, python, *sys.argv)  # 현재 프로세스를 새 프로세스로 교체
    except Exception:
        # execl 실패 시 새 프로세스 띄우고 종료
        import subprocess
        subprocess.Popen([sys.executable] + sys.argv)
        QApplication.quit()
