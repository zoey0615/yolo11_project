
# -*- coding: utf-8 -*-
"""
yolo11：細菌顯微影像 YOLO11 偵測流程套件

使用方式：
    from yolo11 import DetectBacteria
    db = DetectBacteria('cocci')      # 或 'bacilli' / 'all'

使用說明請見 README.md。
"""
# metrics_patch 必須最先載入:環境設定和適應性含數猴子補丁
#忽略 F401（未使用的導入）與 I001（導入順序）警告，防止被自動排版工具打亂順序
from . import metrics_patch  # noqa: F401, I001
from .detector import DetectBacteria #從內部的 detector 模組中，把細菌偵測類別 DetectBacteria 匯入到套件層級
__all__ = ['DetectBacteria'] # 定義對外公開的介面清單（Public API）