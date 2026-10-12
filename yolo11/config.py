# -*- coding: utf-8 -*-
"""
config：常數與模式設定
只依賴標準函式庫，因此是整個套件的最底層（其他模組可以 import 它，它不 import 任何本套件模組）。
"""
from dataclasses import dataclass #純資料容器裝飾器
from pathlib import Path #路徑操作

# ============================================================================
# 常數
# ============================================================================
IMAGE_EXTS = ('.jpg', '.png', '.jpeg')       # 支援的影像副檔名（比對時一律轉小寫）
BASE_SEED = 42                               # K-Fold 切分種子；第 i 折訓練種子 = BASE_SEED + i
N_SPLITS = 5                                 # 調參時固定使用的 K-Fold 折數（取第 1 折）
TUNE_EPOCHS = 20                             # 調參時每次嘗試的訓練輪數
FIXED_CONF = 0.5                             # CSV 報表中 Precision/Recall/F1@固定門檻 所用的信心門檻
RUNS_DIR = './runs/detect'                     # Ultralytics 輸出根目錄
MERGED_DATASET_DIR = Path('dataset_all')     # 混合模式：訓練+驗證資料
MERGED_TEST_DIR = Path('test_image_all')     # 混合模式：測試集

SEARCH_SPACE = { # 超參數搜尋邊界 ：針對細菌影像關鍵參數
    'lr0': (1e-4, 1e-2),     # 初始學習率
    'mosaic': (0.0, 0.5),    # 降低 Mosaic 避免過度切割微小菌體
    'degrees': (0.0, 180.0), # 顯微鏡全角度旋轉
    'scale': (0.1, 0.5),     # 縮放模擬 400x/1000x 放大倍率
    'cls': (0.5, 1.5),       # 加強 Single 與 Colony 分類權重
    'iou': (0.4, 0.7),       # 調整 NMS 區分重疊菌體
}

# 調參結果帶入訓練時，這些鍵不可被覆蓋
NON_OVERRIDABLE_HYP_KEYS = {'data', 'epochs', 'task', 'mode', 'model', 'single_cls', 'imgsz', 'pretrained'}
# 各物種在原始資料集中的標籤 ID 對照表：(Single, Colony)
SPECIES_RAW_IDS = {'cocci': (10, 11), 'bacilli': (20, 21)}

@dataclass(frozen=True) #模式設定類別上方的裝飾器:自動在背景生成好 def __init__ 
class ModeConfig:
    """單一模式（cocci / bacilli / all）的設定。新增模式只需在 MODES 加一筆。"""
    name: str            # 輸出資料夾與檔名使用的名稱
    banner: str          # 初始化時印出的訊息
    class_mapping: dict  # 原始標籤 ID → 最終類別 ID
    class_names: tuple   # 最終類別名稱（索引即最終類別 ID）
    is_mixed: bool       # 是否為球菌+桿菌混合模式


MODES = { #字典與ModeConfig 物件實例
    'bacilli': ModeConfig('bacilli', "Setup bacteria: bacilli",
                          {20: 0, 21: 1}, ('Single', 'Colony'), False),
    'cocci':   ModeConfig('cocci', "Setup bacteria: cocci",
                          {10: 0, 11: 1}, ('Single', 'Colony'), False),
    # 原始標籤 ID 本來就不重疊 (10/11 vs 20/21)，用一個聯集字典即可
    'all':     ModeConfig('all', "Setup bacteria: all (cocci + bacilli, 4 classes)",
                          {10: 0, 11: 1, 20: 2, 21: 3},
                          ('Cocci_Single', 'Cocci_Colony', 'Bacilli_Single', 'Bacilli_Colony'), True)
    }
DEFAULT_MODE = 'bacilli'
DEFAULT_TRAINING = {'img_size': 512, 'epoch': 50, 'batch_size': 16}
SOURCE_NAMES = ('dibas', 'matlab', 'diffusion')   # 三種訓練資料來源
MIX_ROOT = Path('dataset_mix')                    # 混合資料集輸出根目錄