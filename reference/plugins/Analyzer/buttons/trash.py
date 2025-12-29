#!/usr/bin/env python
# -*- coding: utf-8 -*-

from PyQt5.QtWidgets import QMessageBox

def request_delete_all(parent, count: int) -> bool:
    """
    등록(올라와있는)된 파일을 '모두' 삭제할지 확인하고, Yes면 True 반환.
    실제 파일 삭제는 하지 않음(필요 시 호출부에서 처리).
    """
    if count <= 0:
        QMessageBox.information(parent, "삭제", "삭제할 업로드 데이터가 없습니다.")
        return False

    resp = QMessageBox.question(
        parent,
        "전체 삭제",
        f"등록된 {count}개 업로드 데이터를 모두 삭제할까요?",
        QMessageBox.Yes | QMessageBox.No,
        QMessageBox.No
    )
    return resp == QMessageBox.Yes
