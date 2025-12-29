# -*- coding: utf-8 -*-
import sys
from PyQt5.QtWidgets import QApplication
from window import AppWindow

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = AppWindow()
    w.showFullScreen()
    sys.exit(app.exec_())
