# -*- coding: utf-8 -*-
"""
metrics_patch：環境設定與 Ultralytics 自訂適應度（fitness）猴子補丁

本模組只靠「被 import 時的副作用」生效，沒有對外提供任何名稱。
由 yolo11/__init__.py 最先載入，順序不可更動。
"""
# 1. 環境變數：必須在 numpy / torch 載入之前設定
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"   #解決多執行緒函式庫衝突omp error
# 2. matplotlib 後端：必須在 ultralytics 匯入 pyplot 之前設定
import matplotlib
matplotlib.use('Agg')   # Anti-Grain Geometry：非互動式後端，避免無顯示器環境報錯
import numpy as np
from ultralytics.utils.metrics import Metric #從Ultralytics YOLO 套件內部的資料夾與檔案結構，使用Metric 類別

class CustomMetric(Metric): #先取一個新名字（CustomMetric），繼承官方的 Metric，自訂適應度
    """基因演算法會組合參數，用yolo model跑出[Precision, Recall, mAP50, mAP50-95]用適應性函數當裁判"""
    def fitness(self) -> float:
        # mean_results() 回傳所有類別平均後的 [Precision, Recall, mAP50, mAP50-95]
        w = [0.0, 0.3, 0.0, 0.7]
        # nan_to_num：將 NaN 轉成 0.0，避免某類別缺資料時整體分數變 NaN
        return float((np.nan_to_num(np.array(self.mean_results())) * w).sum())
Metric.fitness = CustomMetric.fitness # 動態替換類別方法（Monkey Patch）:透過猴子補丁，直接把官方原本的 fitness 函數強行抽換掉