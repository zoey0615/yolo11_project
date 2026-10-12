# -*- coding: utf-8 -*-
"""
detector：DetectBacteria 類別（細菌顯微影像 YOLO11 偵測流程）

本類別只負責「設定與路徑」，功能由三個 Mixin 提供：
  DatasetBuilderMixin   標籤轉換、混合資料集、K-Fold 切分   （dataset_builder.py）
  TrainerMixin          調參、單折訓練、交叉驗證             （trainer.py）
  EvaluatorMixin        測試集評估、CSV、劃框、單張推論      （evaluator.py）

使用說明請見 README.md。
"""
from pathlib import Path

from .config import MODES, DEFAULT_MODE, DEFAULT_TRAINING, SPECIES_RAW_IDS
from .dataset_builder import DatasetBuilderMixin
from .trainer import TrainerMixin
from .evaluator import EvaluatorMixin
from .mixer import MixerMixin

class DetectBacteria(MixerMixin, DatasetBuilderMixin, TrainerMixin, EvaluatorMixin):
    # 各物種在原始資料集中的標籤 ID：(Single, Colony)；來源為 config，保留類別屬性以相容舊用法
    SPECIES_RAW_IDS = SPECIES_RAW_IDS

    # ------------------------------------------------------------------ 設定
    def __init__(self, *args):
        """不帶參數 → 預設 bacilli；不支援的字串 → 印出警告並切換為 bacilli"""
        choice = args[0] if args else DEFAULT_MODE
        if not isinstance(choice, str) or choice not in MODES:
            print(f"不支援的細菌種類: {choice}，預設切換至 {DEFAULT_MODE}")
            choice = DEFAULT_MODE
        self._apply_mode(choice)

    def _apply_mode(self, mode):
        """依 MODES 設定表套用模式（所有後續方法都讀取這些屬性）"""
        cfg = MODES[mode]
        print(cfg.banner)
        self.img_size = DEFAULT_TRAINING['img_size']        # 統一影像大小
        self.epoch = DEFAULT_TRAINING['epoch']              # 訓練輪數
        self.batch_size = DEFAULT_TRAINING['batch_size']    # 批次大小
        self.class_mapping = dict(cfg.class_mapping)        # 原始標籤 ID → 最終類別 ID
        self.class_names = list(cfg.class_names)
        self.num_classes = len(self.class_names)
        self.bacteria_names = [cfg.name]
        self.is_mixed = cfg.is_mixed
        self.custom_dataset_dir = None      # 建立混合資料集後會指向 dataset_mix/{tag}
    # 為了相容舊程式碼，保留原本的三個初始化方法名稱
    def init_bacilli(self):
        self._apply_mode('bacilli')

    def init_cocci(self):
        self._apply_mode('cocci')

    def init_all(self):
        self._apply_mode('all')

    # ------------------------------------------------------------------ 路徑
    @property
    def name(self):
        """目前模式的名稱（cocci / bacilli / all），用於所有輸出資料夾與檔名"""
        return self.bacteria_names[0]

    @property
    def dataset_dir(self):
        return self.custom_dataset_dir or Path(f'dataset_{self.name}')

    @property
    def image_dir(self):
        return self.dataset_dir / 'images'

    @property
    def label_dir(self):
        return self.dataset_dir / 'labels'