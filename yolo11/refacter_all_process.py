# -*- coding: utf-8 -*-
"""
Created on Sat Oct  3 18:21:18 2026

@author: Admin
"""

# -*- coding: utf-8 -*-
"""
DetectBacteria：細菌顯微影像 YOLO11 偵測流程（訓練、調參、K-Fold、測試集評估、單張推論）

支援三種模式：
  'cocci'   球菌單獨訓練  → 2 類 (Single, Colony)
  'bacilli' 桿菌單獨訓練  → 2 類 (Single, Colony)
  'all'     球菌+桿菌混合訓練 → 4 類 (Cocci_Single, Cocci_Colony, Bacilli_Single, Bacilli_Colony)

【資料夾結構】
  單物種模式（例如 cocci）：
    dataset_cocci/images/  dataset_cocci/labels/   訓練+驗證資料（影像與標籤分開存放）
    test_image_cocci/                              獨立測試集（影像與同名 .txt 標籤放同一資料夾）
  混合模式：需先由 build_merged_dataset() 產生
    dataset_all/images/  dataset_all/labels/  test_image_all/
    檔名加物種前綴，例如 cocci_001.jpg、bacilli_001.jpg

【原始標籤 ID 與最終類別 ID】
  cocci  : 10/11 → 0/1        （模式 'cocci'）  ；  10/11 → 0/1 （模式 'all'）
  bacilli: 20/21 → 0/1        （模式 'bacilli'）；  20/21 → 2/3 （模式 'all'）
  順序固定為 (Single, Colony)

------------------------------------------------------------------------
0. 程式結構總覽（快速理解用）
------------------------------------------------------------------------
  檔案由上而下分為六區：

  A. 環境設定與全域猴子補丁   CustomMetric：把 Ultralytics 的 Metric.fitness 換成
                              0.3×Recall + 0.7×mAP50-95（調參與訓練挑 best.pt 都會用到，勿改）
  B. 常數與模式設定           MODES（三種模式的設定表）、SEARCH_SPACE、FIXED_CONF 等
  C. 共用工具函式（模組層級） 影像灰階轉換、標籤 ID 轉換、暫存資料集建立、yaml/路徑清單輸出
  D. DetectBacteria 類別
       設定      __init__ / _apply_mode / 路徑屬性
       資料準備  build_merged_dataset（僅 'all'）、get_kfold_splits
       調參      tune_hyperparameters
       訓練      train_yolo_single_iteration、evaluate_model、run_cross_validation
       評估      detect_bacteria（= 預處理 → val → 寫 CSV → 劃框，各步驟拆成 _ 開頭的小方法）
       推論      predict_image
  E. main()                   完整流程範例（調參 → 5 折訓練 → 測試集評估）
  F. if __name__ == '__main__'

  資料流（原始資料不動，一律複製到暫存資料夾轉換，用完刪除）：
    dataset_{name}/images+labels ─(灰階3通道＋標籤映射)→ dataset_{name}/temp_seed_{seed}/ → 訓練 → 刪除
    test_image_{name}/           ─(灰階3通道＋標籤轉換)→ temp_test_eval_{run_id}/ → val + predict → 刪除

------------------------------------------------------------------------
1. 匯入類別並實例化
------------------------------------------------------------------------
   from yolo11_model import DetectBacteria
   db = DetectBacteria('cocci')     # 或 'bacilli' / 'all'
   # 不帶參數 → 預設 'bacilli'；不支援的字串 → 印出警告並切換為 'bacilli'

   可視需求覆寫預設參數：
   db.epoch = 50        # 訓練輪數
   db.batch_size = 16   # 批次大小
   db.img_size = 512    # 統一影像尺寸

------------------------------------------------------------------------
1b. 混合模式資料準備（只有 'all' 需要，單物種模式請略過）
------------------------------------------------------------------------
   db = DetectBacteria('all')
   db.build_merged_dataset(force=True)     # 第一次或來源資料有變動時用 force=True

   參數：
   species_list = ('cocci', 'bacilli')   要合併的物種
   force  = False   已存在 dataset_all / test_image_all 時預設略過；True 則刪除重建
   strict = True    發現異常標籤就中止並清除半成品；False 則略過異常行並警告

   內部行為：
   - 合併 dataset_{sp}/ 與 test_image_{sp}/ → dataset_all/、test_image_all/
   - 檔名加上 '{sp}_' 前綴避免重複，也是 get_kfold_splits() 分層的依據
   - 標籤依「資料夾所屬物種」轉成最終 ID，同時接受原始 ID (10/11、20/21)
     與已正規化的 0/1（0=Single、1=Colony），例如桿菌資料夾的 0/1 會轉成 2/3
   - 轉換後掃描輸出，確認 ID 只剩 0~3，並印出各類別實例數
   - 印出各物種含「0/1 標籤」的檔案，請抽查影像確認 0=Single、1=Colony 順序無誤
   - 影像為複製，dataset_all 會額外佔用一份磁碟空間

------------------------------------------------------------------------
2. 超參數自動調優（選用，建議先跑一次）
------------------------------------------------------------------------
   db.tune_hyperparameters(iterations=50, base_model='yolo11m.pt')

   內部行為：
   - 取 get_kfold_splits() 產生的 5 折中的「第 1 折」當作調參用的
     train/val（並非用全部 5 折跑調參，只用一折以節省時間）
     混合模式的第 1 折同樣是依物種分層的切分
   - 對 lr0 / mosaic / degrees / scale / cls / iou 六個超參數，
     以基因演算法（epochs=20/iteration）搜尋最佳組合
   - 適應度函數為自訂：0.7 × mAP50-95 + 0.3 × Recall
   - 結果自動存於：
       runs/detect/tune_{bacteria_name}/best_hyperparameters.yaml
     bacteria_name 為 cocci / bacilli / all，不同模式的結果互不覆蓋

------------------------------------------------------------------------
3. 讀取調參結果（若步驟 2 有執行）
------------------------------------------------------------------------
   import yaml
   from pathlib import Path
   bacteria_name = db.bacteria_names[0]
   best_hyp_yaml = Path(f'runs/detect/tune_{bacteria_name}/best_hyperparameters.yaml')
   best_hyp = yaml.safe_load(open(best_hyp_yaml)) if best_hyp_yaml.exists() else None
   # 若找不到，best_hyp 為 None，K-Fold 會使用 Ultralytics 預設超參數

------------------------------------------------------------------------
4. 執行 K-Fold 交叉驗證訓練，取得最佳模型
------------------------------------------------------------------------
   best_result = db.run_cross_validation(
       num_loops=5,             # K-Fold 的 K 值（5 折）
       base_model='yolo11m.pt',
       custom_hyp=best_hyp      # 步驟 3 讀到的超參數，可為 None
   )

   內部行為：
   - get_kfold_splits() 產生 5 組互斥的 train/val 切分
     （每筆資料恰好被驗證一次，非隨機重疊抽樣）
       單物種模式：sklearn KFold(shuffle=True)
       混合模式  ：sklearn StratifiedKFold(shuffle=True)，依檔名前綴（物種）分層，
                   每折驗證集的物種比例與整體相近
   - 每一折都完整訓練一次（train_yolo_single_iteration），
     並用 evaluate_model() 算出該折驗證集 mAP50
   - 5 折中 mAP50 最高者，視為最終最佳模型
   - 回傳 dict，包含：
       best_result['model_path']      # 最佳權重 .pt 路徑
       best_result['iteration']       # 第幾折
       best_result['seed']            # 該折使用的 seed
       best_result['yaml_path']       # 該折使用的資料設定檔
       best_result['eval_report_dir'] # 該折混淆矩陣/PR曲線存放位置
   - 注意：驗證折與訓練資料同為合成影像來源，最終結果請以獨立測試集（步驟 5a）為準

------------------------------------------------------------------------
5a. 對「有標籤」的獨立測試集做完整評估
------------------------------------------------------------------------
   單物種模式：
   db.detect_bacteria(
       model_name=best_result['model_path'],
       image_source='./test_image_cocci',   # 資料夾或單一檔案皆可
       confidence=0.5
   )

   混合模式（同一個模型，分別評估合併測試集與各物種測試集）：
   for folder, tag in [('./test_image_all', 'merged'),
                       ('./test_image_cocci', 'cocci'),
                       ('./test_image_bacilli', 'bacilli')]:
       db.detect_bacteria(model_name=best_result['model_path'],
                          image_source=folder, confidence=0.5, tag=tag)

   參數：
   tag     輸出名稱後綴。run_id = '{bacteria_name}_{tag}'；tag=None 時 run_id = bacteria_name
   species 僅混合模式使用，測試集標籤所屬物種 ('cocci' / 'bacilli')
           None 時由資料夾名稱推測（名稱需含 cocci 或 bacilli）；
           合併後的 test_image_all 標籤已是最終 ID，不需指定；
           若測試集標籤是 0/1 且資料夾名稱無法辨識物種，必須手動指定

   內部行為：
   - 自動將測試影像轉灰階(3通道)，與訓練時前處理一致
   - 標籤依測試集所屬物種轉成最終 ID；轉換後若出現超出範圍的 ID 會印出警告
   - 呼叫 model.val() 計算 Precision / Recall / mAP50 / mAP50-95
   - 輸出 CSV 報表（整體 + 各類別）：
       runs/detect/test_evaluation_{run_id}/test_evaluation_{run_id}.csv
     測試集中沒有標註的類別，CSV 內以 nan 表示（例如球菌測試集上的 Bacilli_* 兩類）
   - 輸出劃框圖片：
       runs/detect/predict_{run_id}/
   - 輸出資料夾以 run_id 區分；同一個 run_id 重跑會直接覆蓋，
     若要保留舊結果，請換 tag 或手動備份

   解讀提醒：
   - 混合模型在單物種測試集上，物種混淆會拉低該物種各類別的指標；
     完整的 4×4 物種混淆矩陣請看 test_evaluation_all_merged/
   - 與單物種模型比較時，請比較同物種各類別（如 Cocci_* 對單獨球菌模型），
     不要直接比整體 mAP

------------------------------------------------------------------------
5b. 對「單張、無標籤」的新影像做純推論
------------------------------------------------------------------------
   db.predict_image(
       model_name=best_result['model_path'],
       image_path='./test_image/your_image.jpg',
       confidence=0.5
   )

   內部行為：
   - 自動轉灰階(3通道)，前處理與 detect_bacteria() 一致
   - 純推論（不跑 model.val()，不需要標籤檔，不計算 mAP/P/R）
   - 印出每個偵測框的類別、信心值、座標
     混合模型輸出 4 類（物種+型態），單物種模型輸出 2 類
   - 劃框結果存於：runs/detect/predict_single/（固定名稱，重跑會覆蓋）
   適用情境：全新、未標註的單張影像快速檢視偵測結果。
   若該圖片本身有對應標籤、需要精確指標，仍應使用 detect_bacteria()。

------------------------------------------------------------------------
6. 混合訓練完整流程範例
------------------------------------------------------------------------
   db = DetectBacteria('all')
   db.build_merged_dataset(force=True)
   db.tune_hyperparameters(iterations=50, base_model='yolo11m.pt')
   best_hyp = yaml.safe_load(open('runs/detect/tune_all/best_hyperparameters.yaml'))
   best_result = db.run_cross_validation(num_loops=5, base_model='yolo11m.pt', custom_hyp=best_hyp)
   # 再依 5a 的混合模式範例評估三組測試集
"""
# ============================================================================
# A. 環境設定與全域猴子補丁
# ============================================================================
import matplotlib #幫 YOLO 框架內部自動繪圖的功能做好環境設定，確保程式在伺服器上能順利跑完
matplotlib.use('Agg')  # Anti-Grain Geometry（反走樣幾何繪圖庫）:強制切換為非互動式繪圖後端，避免無顯示器環境報錯
from ultralytics import YOLO  #載入 Ultralytics YOLO 物件偵檢核心類別
from ultralytics.utils.metrics import Metric, smooth  # Metric：(適應性分數)類別 /home/zoey/.conda/envs/yolo/lib/python3.10/site-packages/ultralytics/utils/metrics.py；smooth：F1 曲線平滑
import os                   # 處理檔案路徑與環境變數
import yaml                 # 生成與讀取 YOLO 所需的設定檔 (.yaml)
import torch #PyTorch 深度學習框架，用來偵測 GPU 資源及清空顯存。
import cv2 #OpenCV 影像處理函式庫，用來讀取、轉換灰階及寫入影像
import shutil               # 用於清理評估暫存資料夾 
import csv                  # 將模型驗證數據匯出成結構化表格報表 
import json                 # 另存最佳模型資訊 (best_info.json)
import numpy as np          # 用於計算陣列與處理缺失值 NaN（Not a Number）
from collections import Counter, defaultdict  # 標籤統計
from dataclasses import dataclass             # 模式設定表
from pathlib import Path # 處理檔案物件導向路徑
from sklearn.model_selection import KFold, StratifiedKFold # 匯入一般與分層 K 折交叉驗證切分工具
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" #強行通過套件載入時的CPU 的多執行緒衝突

# ====(基因演算法)====== 自訂適應度 (Fitness) 類別 ====================
class CustomMetric(Metric): #存取父類別中與模型評估相關的屬性和方法
    """Custom Metric class for Ultralytics YOLO models with 70% mAP50-95 and 30% Recall fitness."""
    def fitness(self) -> float: #型態提示:須回傳一個浮點數
        w = [0.0, 0.3, 0.0, 0.7]  #適應性函數權重
        #呼叫物件內部的方法，取得當前模型驗證後各項效能指標的所有物件類別的平均表現:[Precision, Recall, mAP50, mAP50-95]
        return float((np.nan_to_num(np.array(self.mean_results())) * w).sum()) #適應性函數:NaN to Number將 NaN 自動轉為 0.0（預設值）。
Metric.fitness = CustomMetric.fitness #動態替換類別方法（Monkey Patch）:Ultralytics 在訓練（model.train()）與超參數調校（model.tune()）評估模型好壞時，都會呼叫 Metric 的 fitness() 方法


# ============================================================================
# B. 常數與模式設定
# ============================================================================
IMAGE_EXTS = ('.jpg', '.png', '.jpeg')       # 支援的影像副檔名（比對時一律轉小寫）
BASE_SEED = 42                               # K-Fold 切分種子；第 i 折訓練種子 = BASE_SEED + i
N_SPLITS = 5                                 # 調參時固定使用的 K-Fold 折數（取第 1 折）
TUNE_EPOCHS = 20                             # 調參時每次嘗試的訓練輪數
FIXED_CONF = 0.5                             # CSV 報表中 Precision/Recall/F1@固定門檻 所用的信心門檻
RUNS_DIR = 'runs/detect'                     # Ultralytics 輸出根目錄
MERGED_DATASET_DIR = Path('dataset_all')     # 混合模式：訓練+驗證資料
MERGED_TEST_DIR = Path('test_image_all')     # 混合模式：測試集

# 超參數搜尋邊界 (Search Space)：針對細菌影像關鍵參數
SEARCH_SPACE = {
    'lr0': (1e-4, 1e-2),     # 初始學習率
    'mosaic': (0.0, 0.5),    # 降低 Mosaic 避免過度切割微小菌體
    'degrees': (0.0, 180.0), # 顯微鏡全角度旋轉
    'scale': (0.1, 0.5),     # 縮放模擬 400x/1000x 放大倍率
    'cls': (0.5, 1.5),       # 加強 Single 與 Colony 分類權重
    'iou': (0.4, 0.7),       # 調整 NMS 區分重疊菌體
}

# 調參結果帶入訓練時，這些鍵不可被覆蓋（由程式自行決定）
NON_OVERRIDABLE_HYP_KEYS = {'data', 'epochs', 'task', 'mode', 'model', 'single_cls', 'imgsz'}


@dataclass(frozen=True)
class ModeConfig:
    """單一模式（cocci / bacilli / all）的設定。新增模式只需在 MODES 加一筆。"""
    name: str            # 輸出資料夾與檔名使用的名稱（bacteria_name）
    banner: str          # 初始化時印出的訊息
    class_mapping: dict  # 原始標籤 ID → 最終類別 ID
    class_names: tuple   # 最終類別名稱（索引即最終類別 ID）
    is_mixed: bool       # 是否為球菌+桿菌混合模式


MODES = {
    'bacilli': ModeConfig('bacilli', "Setup bacteria: bacilli",
                          {20: 0, 21: 1}, ('Single', 'Colony'), False),
    'cocci':   ModeConfig('cocci', "Setup bacteria: cocci",
                          {10: 0, 11: 1}, ('Single', 'Colony'), False),
    # 原始標籤 ID 本來就不重疊 (10/11 vs 20/21)，用一個聯集字典即可
    'all':     ModeConfig('all', "Setup bacteria: all (cocci + bacilli, 4 classes)",
                          {10: 0, 11: 1, 20: 2, 21: 3},
                          ('Cocci_Single', 'Cocci_Colony', 'Bacilli_Single', 'Bacilli_Colony'), True),
}
DEFAULT_MODE = 'bacilli'
DEFAULT_TRAINING = {'img_size': 512, 'epoch': 50, 'batch_size': 16}


# ============================================================================
# C. 共用工具函式（模組層級，不依賴 DetectBacteria 的狀態）
# ============================================================================
def is_image_name(name):
    """副檔名是否為支援的影像格式"""
    return Path(name).suffix.lower() in IMAGE_EXTS


def list_image_names(folder):
    """回傳資料夾內排序後的影像檔名清單"""
    return sorted(f.name for f in Path(folder).iterdir() if f.is_file() and is_image_name(f.name))


def label_name_of(img_name):
    """影像檔名 → 標籤檔名（以最後一個點切分副檔名）"""
    return img_name.rsplit('.', 1)[0] + '.txt'


def species_of(filename):
    """檔名前綴即物種（cocci_001.jpg → 'cocci'）"""
    return filename.split('_', 1)[0]


def to_gray_3ch(img):
    """BGR 影像 → 灰階後複製成 3 通道（訓練、調參、評估、推論一律使用，確保前處理一致）"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.merge([gray, gray, gray])


def read_gray_3ch(path):
    """讀取影像並轉灰階 3 通道；讀取失敗回傳 None"""
    img = cv2.imread(str(path))
    return None if img is None else to_gray_3ch(img)


def convert_label_lines(lines, label_map):
    """
    轉換 YOLO 標籤行的類別 ID，回傳 (新標籤行, 輸出 ID 集合)。
    - 不在轉換表內的 ID 原樣保留（交給事後檢查）；
    - 無法解析類別 ID 的行略過；空行略過。
    """
    new_lines, out_ids = [], set()
    for line in lines:
        parts = line.split()
        if not parts:
            continue
        try:
            old_id = int(parts[0])
        except ValueError:
            continue
        new_id = label_map.get(old_id, old_id)
        parts[0] = str(new_id)
        out_ids.add(new_id)
        new_lines.append(' '.join(parts) + '\n')
    return new_lines, out_ids


def stage_image(src, dst):
    """把影像轉灰階 3 通道後存到 dst；已存在則略過；讀取失敗時退回原檔複製，避免整批中斷"""
    if src.exists() and not dst.exists():
        gray = read_gray_3ch(src)
        if gray is not None:
            cv2.imwrite(str(dst), gray)
        else:
            shutil.copy(src, dst)


def stage_label(src, dst, label_map):
    """若來源標籤存在，轉換類別 ID 後寫到 dst"""
    if src.exists():
        with open(src, 'r') as f:
            lines = f.readlines()
        new_lines, _ = convert_label_lines(lines, label_map)
        with open(dst, 'w') as f:
            f.writelines(new_lines)


def stage_split(names, src_img_dir, src_lbl_dir, dst_img_dir, dst_lbl_dir, label_map):
    """
    把一組影像（檔名清單）處理後放進暫存資料集：影像轉灰階、標籤轉換類別 ID。
    回傳處理後影像的絕對路徑清單（供寫入 train/val 清單檔）。
    """
    src_img_dir, src_lbl_dir = Path(src_img_dir), Path(src_lbl_dir)
    dst_img_dir, dst_lbl_dir = Path(dst_img_dir), Path(dst_lbl_dir)
    if not src_img_dir.exists():
        return []
    paths = []
    for name in names:
        if not is_image_name(name):
            continue
        dst_img = dst_img_dir / name
        stage_image(src_img_dir / name, dst_img)
        paths.append(str(dst_img.absolute()))
        lbl = label_name_of(name)
        stage_label(src_lbl_dir / lbl, dst_lbl_dir / lbl, label_map)
    return paths


def write_path_list(path, paths):
    """寫入影像路徑清單檔（每行一個路徑）"""
    with open(path, 'w') as f:
        f.write('\n'.join(paths) + '\n')


def write_data_yaml(path, train, val, root, nc, names, test=None):
    """寫入 YOLO 資料集設定檔（路徑一律轉成絕對路徑）"""
    data = {'train': os.path.abspath(train), 'val': os.path.abspath(val)}
    if test is not None:
        data['test'] = os.path.abspath(test)
    data.update({'path': os.path.abspath(root), 'nc': nc, 'names': list(names)})
    with open(path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)


def print_detections(model, result):
    """印出單張影像的每個偵測框（類別、信心值、座標）"""
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        class_id = int(box.cls[0].item())
        conf_val = float(box.conf[0].item())
        class_name = model.names[class_id]
        print(f"  -> Detected: {class_name} ({conf_val:.2f}) at [{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}]")


# ============================================================================
# D. DetectBacteria 類別
# ============================================================================
class DetectBacteria:
    #定義不同細菌種類在原始資料集中的標籤 ID（Raw Label IDs）對照表：(Single, Colony)
    SPECIES_RAW_IDS = {'cocci': (10, 11), 'bacilli': (20, 21)}

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
        return Path(f'dataset_{self.name}')

    @property
    def image_dir(self):
        return self.dataset_dir / 'images'

    @property
    def label_dir(self):
        return self.dataset_dir / 'labels'

    # ------------------------------------------------------------------ 標籤轉換（物種相關）
    def _species_label_map(self, species):
        """ 回傳該物種的「標籤 ID → 最終 ID」轉換表（同時接受原始 ID 與 0/1） """
        single_raw, colony_raw = self.SPECIES_RAW_IDS[species] #字典查詢各個物種（Species）對應的原始標籤 ID。
        single_out = self.class_mapping[single_raw]
        colony_out = self.class_mapping[colony_raw]
        return {single_raw: single_out, colony_raw: colony_out,
                0: single_out, 1: colony_out}

    def _infer_species(self, folder_name):
        """ 由資料夾名稱推測物種（test_image_cocci → 'cocci'）；無法唯一判斷回傳 None """
        hits = [sp for sp in self.SPECIES_RAW_IDS if sp in folder_name.lower()]
        return hits[0] if len(hits) == 1 else None

    def _resolve_label_map(self, species):
        """ 混合模式且已知物種 → 該物種的轉換表；其餘 → class_mapping """
        if self.is_mixed and species is not None:
            return self._species_label_map(species)
        return self.class_mapping

    def _convert_label_lines(self, lines, label_map):
        """ 轉換標籤行的類別 ID（實作見模組層級的 convert_label_lines，保留此方法以相容舊呼叫） """
        return convert_label_lines(lines, label_map)

    def _convert_label_file(self, src, dst, species, stats):
        """ 讀取單一 YOLO 標籤檔，依該物種的轉換表轉成最終類別 ID 後寫出（嚴格版，供 build_merged_dataset 使用）。
            空檔案（無目標的負樣本）會保留為空檔。 """
        label_map = self._species_label_map(species)
        raw_ids = set(self.SPECIES_RAW_IDS[species]) # 該物種的「原始標籤 ID 集合」
        new_lines = []
        for ln, line in enumerate(src.read_text().splitlines(), 1):
            parts = line.split()
            if not parts:
                continue
            try:
                old_id = int(parts[0])
            except ValueError:
                stats['bad_format'].append(f'{src}:{ln}')
                continue
            if old_id not in label_map:
                stats['unexpected'].append(f'{src}:{ln} (id={old_id}, 物種={species})')
                continue
            # 記錄這行是「原始 ID」還是「已正規化的 0/1」，供最後報告
            kind = 'raw' if old_id in raw_ids else 'normalized'
            stats['source'][(species, kind)] += 1
            if kind == 'normalized':
                stats['normalized_files'][species].add(src.name)
            parts[0] = str(label_map[old_id])
            new_lines.append(' '.join(parts) + '\n')
        dst.write_text(''.join(new_lines))
        stats['converted'] += len(new_lines)

    # ------------------------------------------------------------------ 資料準備：混合資料集
    def build_merged_dataset(self, species_list=('cocci', 'bacilli'), force=False, strict=True):
        """ 合併 dataset_{sp}/ 與 test_image_{sp}/ → dataset_all/、test_image_all/
            1. 檔名加 '{sp}_' 前綴避免重複，並供 get_kfold_splits() 依前綴分層
            2. 標籤依「所屬物種」轉成最終 0~3；同時接受原始 ID (10/11, 20/21) 與已正規化的 0/1
            strict=True：發現異常標籤就中止並清除半成品；False：略過異常行並警告 """
        if not self.is_mixed:
            raise RuntimeError("請以 DetectBacteria('all') 呼叫此方法（需要 4 類別的 class_mapping）")
        for sp in species_list:
            missing = set(self.SPECIES_RAW_IDS[sp]) - set(self.class_mapping)
            if missing:
                raise ValueError(f"class_mapping 缺少 {sp} 的原始 ID: {sorted(missing)}")

        if MERGED_DATASET_DIR.exists() and MERGED_TEST_DIR.exists() and not force:
            print("dataset_all / test_image_all 已存在，略過合併（要重建請 force=True）")
            return
        if force:
            self._remove_merged_dirs()

        dst_img, dst_lbl = MERGED_DATASET_DIR / 'images', MERGED_DATASET_DIR / 'labels'
        dst_test = MERGED_TEST_DIR
        for d in (dst_img, dst_lbl, dst_test):
            d.mkdir(parents=True, exist_ok=True)

        jobs = []
        for sp in species_list:
            jobs += [(f'dataset_{sp}/images', dst_img, sp),
                     (f'dataset_{sp}/labels', dst_lbl, sp),
                     (f'test_image_{sp}',     dst_test, sp)]

        stats = {'converted': 0, 'bad_format': [], 'unexpected': [],
                 'source': Counter(), 'normalized_files': defaultdict(set)}
        for src, dst, sp in jobs:
            self._merge_one_folder(Path(src), dst, sp, stats)

        self._check_label_problems(stats, strict)
        self._report_normalized_labels(stats, species_list)
        counter = self._count_label_ids(dst_lbl, dst_test)

        bad_out = set(counter) - set(self.class_mapping.values())
        if bad_out:
            raise RuntimeError(f"輸出標籤出現非預期 ID: {sorted(bad_out)}")
        print(f"\n[完成] 共轉換 {stats['converted']} 個標註。各類別實例數（訓練+測試）：")
        for cid, name in enumerate(self.class_names):
            print(f"  {cid} {name}: {counter.get(cid, 0)}")

    @staticmethod
    def _remove_merged_dirs():
        shutil.rmtree(MERGED_DATASET_DIR, ignore_errors=True)
        shutil.rmtree(MERGED_TEST_DIR, ignore_errors=True)

    def _merge_one_folder(self, src, dst, species, stats):
        """把單一來源資料夾的影像（複製）與標籤（轉換）加上 '{species}_' 前綴後放進 dst"""
        if not src.exists():
            print(f"警告：找不到 {src}，略過")
            return
        n = 0
        for f in src.iterdir():
            if not f.is_file():
                continue
            if f.suffix.lower() == '.txt':
                self._convert_label_file(f, dst / f'{species}_{f.name}', species, stats)
                n += 1
            elif is_image_name(f.name):
                shutil.copy(f, dst / f'{species}_{f.name}')
                n += 1
        print(f"{src} → {dst}: {n} 個檔案")

    def _check_label_problems(self, stats, strict):
        """ 異常標籤處理：strict 時中止並清除半成品 """
        problems = stats['bad_format'] + stats['unexpected']
        if not problems:
            return
        print(f"\n警告：共 {len(problems)} 行標籤異常（前 10 筆）：")
        for p in problems[:10]:
            print(f"  - {p}")
        if strict:
            self._remove_merged_dirs()
            raise ValueError("標籤轉換發現異常，已中止並清除 dataset_all / test_image_all。"
                             "請修正來源資料，或以 strict=False 略過異常行。")

    @staticmethod
    def _report_normalized_labels(stats, species_list):
        """ 回報：哪些物種出現了「已正規化的 0/1」，請使用者抽查順序 """
        for sp in species_list:
            n_raw = stats['source'][(sp, 'raw')]
            n_norm = stats['source'][(sp, 'normalized')]
            print(f"[{sp}] 原始 ID 標註: {n_raw}；已正規化 0/1 標註: {n_norm}")
            if n_norm:
                files = sorted(stats['normalized_files'][sp])
                print(f"  ↳ 含 0/1 的標籤檔 {len(files)} 個，例如: {files[:3]}"
                      f"（已依 {sp} 視為 Single/Colony 轉換，請確認順序無誤）")

    @staticmethod
    def _count_label_ids(*folders):
        """ 掃描輸出資料夾內所有標籤檔，統計各類別 ID 的實例數 """
        counter = Counter()
        for folder in folders:
            for lf in folder.glob('*.txt'):
                for line in lf.read_text().splitlines():
                    parts = line.split()
                    if parts:
                        counter[int(parts[0])] += 1
        return counter

    # ------------------------------------------------------------------ 資料準備：K-Fold
    def get_kfold_splits(self, image_dir, n_splits=5, seed=BASE_SEED):
        """ 取得 K-Fold 的所有摺數分割清單：[(影像訓練清單, 影像驗證清單, 標籤訓練清單, 標籤驗證清單), ...]
            單物種模式：KFold(shuffle=True)；混合模式：StratifiedKFold，依檔名前綴（物種）分層 """
        image_list = list_image_names(image_dir)
        image_array = np.array(image_list)
        label_array = np.array([label_name_of(n) for n in image_list])

        if self.is_mixed:
            species = np.array([species_of(n) for n in image_list])
            unknown = set(species) - set(self.SPECIES_RAW_IDS)
            if unknown:
                raise ValueError(f"檔名缺少物種前綴 (cocci_ / bacilli_): {sorted(unknown)[:5]}")
            kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            split_iter = kf.split(image_array, species)   # 依物種分層
        else:
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
            split_iter = kf.split(image_array)

        return [(image_array[tr].tolist(), image_array[va].tolist(),
                 label_array[tr].tolist(), label_array[va].tolist())
                for tr, va in split_iter]

    def save_txt(self, filename, data_list, base_dir):
        """ 將訓練與驗證資料檔名絕對路徑存入文字檔（保留以相容舊呼叫；
            目前流程由 train_yolo_single_iteration 直接寫入處理後的路徑清單） """
        with open(filename, 'w') as f:
            for item in data_list:
                f.write(str(Path(base_dir) / item) + '\n')

    # ------------------------------------------------------------------ 訓練
    def train_yolo_single_iteration(self, image_train, image_val, train_txt_path, val_txt_path,
                                    seed, model_name='yolo11m.pt', custom_hyp=None):
        """ 訓練單一折：建暫存資料集（灰階＋標籤映射）→ 寫清單與 yaml → model.train()。
            回傳 (權重資料夾, yaml 路徑, mAP50, 暫存根目錄) """
        name = self.name
        temp_root = self.dataset_dir / f'temp_seed_{seed}'    # 依 seed 隔離的暫存根目錄
        temp_img, temp_lbl = temp_root / 'images', temp_root / 'labels'
        temp_img.mkdir(parents=True, exist_ok=True)
        temp_lbl.mkdir(parents=True, exist_ok=True)

        train_paths = stage_split(image_train, self.image_dir, self.label_dir, temp_img, temp_lbl, self.class_mapping)
        val_paths = stage_split(image_val, self.image_dir, self.label_dir, temp_img, temp_lbl, self.class_mapping)

        write_path_list(train_txt_path, train_paths)
        write_path_list(val_txt_path, val_paths)
        yaml_path = f'bacteria_{name}_seed_{seed}.yaml'
        write_data_yaml(yaml_path, train_txt_path, val_txt_path, temp_root, self.num_classes, self.class_names)

        model = YOLO(model_name)
        train_args = {
            'data': yaml_path,
            'epochs': self.epoch,
            'imgsz': self.img_size,
            'batch': self.batch_size,
            'name': f'bacteria_model_{name}_seed_{seed}',
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'val': True,
            'seed': seed,
        }
        if custom_hyp:
            train_args.update({k: v for k, v in custom_hyp.items() if k not in NON_OVERRIDABLE_HYP_KEYS})

        print(f"\n--- 開始訓練 {name} 模型 (Seed: {seed}) ---")
        results = model.train(**train_args)

        try:
            current_map50 = results.results_dict['metrics/mAP50(B)']
        except KeyError:
            current_map50 = results.box.map50
        return results.save_dir, yaml_path, current_map50, temp_root

    def evaluate_model(self, model_path, yaml_path, experiment_name):
        """ 載入模型並對驗證集進行單獨評估，自動生成混淆矩陣與 PR 曲線；回傳 mAP50 """
        model = YOLO(model_path)
        metrics = model.val(data=yaml_path, name=experiment_name)
        return metrics.box.map50

    def run_cross_validation(self, num_loops=5, base_model='yolo11m.pt', custom_hyp=None):
        """ 核心整合：執行真正的 K-Fold 交叉驗證，並以各折驗證集 mAP50 最高者作為最佳模型 """
        name = self.name
        best_map50 = -1.0
        best_model_info = {}

        print(f"\n==================== 開始執行 {num_loops} 折真實 K-Fold 交叉驗證 ====================")
        splits = self.get_kfold_splits(self.image_dir, n_splits=num_loops, seed=BASE_SEED)

        for i, (image_train, image_val, _, _) in enumerate(splits):
            fold = i + 1
            seed = BASE_SEED + i  # 給每一摺一個標記用的 seed
            print(f"\n>>> 【第 {fold} / {num_loops} 折交叉驗證】 <<<")

            train_txt = str(self.dataset_dir / f'image_train_fold_{fold}.txt')
            val_txt = str(self.dataset_dir / f'image_val_fold_{fold}.txt')
            save_dir, yaml_path, _, temp_root = self.train_yolo_single_iteration(
                image_train, image_val, train_txt, val_txt, seed, base_model, custom_hyp=custom_hyp)

            exp_eval_name = f'eval_report_{name}_fold_{fold}'
            best_pt = os.path.join(save_dir, 'weights', 'best.pt')
            print(f"正在為第 {fold} 折生成獨立評估圖表...")
            eval_map50 = self.evaluate_model(best_pt, yaml_path, exp_eval_name)
            print(f"-> 本摺驗證結果 mAP50: {eval_map50:.4f}")
            shutil.rmtree(temp_root, ignore_errors=True)   # 驗證完才能刪暫存資料集

            if eval_map50 > best_map50:   # 比較並篩選出表現最優異的模型
                best_map50 = eval_map50
                best_model_info = {
                    'iteration': fold,
                    'seed': seed,
                    'model_path': best_pt,
                    'yaml_path': yaml_path,
                    'eval_report_dir': f'{RUNS_DIR}/{exp_eval_name}',
                }
                print(f"★ 發現更優模型！當前最佳 mAP50 提升至: {best_map50:.4f}")

        self._print_cv_report(num_loops, best_map50, best_model_info)
        self._save_best_model(best_model_info)
        return best_model_info

    def _print_cv_report(self, num_loops, best_map50, info):
        print("\n" + "=" * 60)
        print("【K-Fold 交叉驗證訓練完成報告】")
        print(f" 細菌種類: {self.name}")
        print(f" 總計折數 (K): {num_loops} 折")
        print(f" 表現最優異的摺數: 第 {info['iteration']} 折")
        print(f" 最高精準度 mAP50: {best_map50:.4f}")
        print(f" 最佳模型權重路徑: {info['model_path']}")
        print(f" 該最優模型的評估圖表儲存於:\n   ↳ {info['eval_report_dir']}/")
        print("=" * 60)

    def _save_best_model(self, info):
        """ 把最佳權重另存到 runs/detect/best_{name}/best.pt，並寫 best_info.json """
        out_dir = Path(f'{RUNS_DIR}/best_{self.name}')
        out_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(info['model_path'], out_dir / 'best.pt')
        with open(out_dir / 'best_info.json', 'w', encoding='utf-8') as f:
            json.dump(info, f, ensure_ascii=False, indent=2)
        print(f" 最佳權重已另存: {out_dir / 'best.pt'}")

    # ------------------------------------------------------------------ 調參
    def tune_hyperparameters(self, iterations=30, base_model='yolo11m.pt'):
        """ 針對細菌微觀影像特徵進行自動超參數搜尋 (Genetic Algorithm)，只使用 K-Fold 的第 1 折 """
        name = self.name
        print(f"\n==================== 開始超參數自動調參搜尋 (Iterations: {iterations}) ====================")

        # 1. 專屬該次調參的暫存資料集根目錄（若有殘留先清除重建）
        tune_root = Path(f'./dataset_{name}_tune_temp')
        shutil.rmtree(tune_root, ignore_errors=True)
        tune_img, tune_lbl = tune_root / 'images', tune_root / 'labels'
        tune_img.mkdir(parents=True, exist_ok=True)
        tune_lbl.mkdir(parents=True, exist_ok=True)

        # 2. 取 K-Fold 的第一摺，處理影像與標籤
        image_train, image_val, _, _ = self.get_kfold_splits(self.image_dir, n_splits=N_SPLITS, seed=BASE_SEED)[0]
        train_paths = stage_split(image_train, self.image_dir, self.label_dir, tune_img, tune_lbl, self.class_mapping)
        val_paths = stage_split(image_val, self.image_dir, self.label_dir, tune_img, tune_lbl, self.class_mapping)

        # 3. 清單檔與調參專用 data.yaml
        train_txt, val_txt = tune_root / 'tune_train.txt', tune_root / 'tune_val.txt'
        write_path_list(train_txt, train_paths)
        write_path_list(val_txt, val_paths)
        tune_yaml_path = f'tune_{name}.yaml'
        write_data_yaml(tune_yaml_path, train_txt, val_txt, tune_root, self.num_classes, self.class_names)

        # 4. 執行基因演算法調參
        model = YOLO(base_model)
        model.tune(
            data=tune_yaml_path,
            model=base_model,
            epochs=TUNE_EPOCHS,
            iterations=iterations,
            optimizer='AdamW',
            plots=True,
            save=True,
            val=True,
            space=SEARCH_SPACE,
            name=f'tune_{name}'
        )

        # 5. 清理調參專用暫存資料集根目錄與 yaml 檔
        shutil.rmtree(tune_root, ignore_errors=True)
        if os.path.exists(tune_yaml_path):
            os.remove(tune_yaml_path)
        print(f"\n[完成] 超參數最佳化搜尋結束，結果已寫入 {RUNS_DIR}/tune_{name}/best_hyperparameters.yaml")

    # ------------------------------------------------------------------ 評估：測試集
    def _prf_at_conf(self, metrics, thr):
        """取固定信心門檻 thr 下，每個『測試集有標註的類別』的 P / R / F1"""
        px = np.asarray(metrics.box.px)                     # 1000 個信心值 (0~1)
        idx = int(np.argmin(np.abs(px - thr)))
        p = np.asarray(metrics.box.p_curve)[:, idx]
        r = np.asarray(metrics.box.r_curve)[:, idx]
        f1 = 2 * p * r / (p + r + 1e-16)
        return p, r, f1

    def detect_bacteria(self, model_name='yolo11m.pt', image_source=None, confidence=0.5,
                        tag=None, species=None):
        """ 
        測試集綜合評估與預測：
        1. 自動將測試圖片轉灰階(3通道)
        2. 自動對齊 .txt 標籤（依測試集所屬物種轉成最終類別 ID）並呼叫 model.val()
        3. 自動匯出數據為 CSV 檔案
        4. 自動繪製並儲存帶框圖片
        tag    : 輸出名稱後綴。tag='cocci' → test_evaluation_all_cocci / predict_all_cocci
                 None 時沿用舊名稱 test_evaluation_{bacteria_name}
        species: 僅混合模式使用，測試集標籤所屬物種 ('cocci'/'bacilli')。
                 None 時由資料夾名稱推測；已合併的 test_image_all 不需指定
        """
        run_id = f'{self.name}_{tag}' if tag else self.name    # 所有輸出名稱都用它
        if image_source is None:
            image_source = f'./test_image_{self.name}'

        collected = self._collect_test_images(image_source)
        if collected is None:
            return None
        test_path, image_list = collected
        label_map = self._resolve_eval_label_map(test_path, species)

        # 1. 預處理：影像轉灰階、標籤轉最終 ID，放進暫存資料夾
        eval_temp_dir = Path(f'./temp_test_eval_{run_id}')
        eval_img_dir, processed_img_paths = self._prepare_eval_dataset(test_path, image_list, label_map, eval_temp_dir)
        test_txt_path, test_yaml_path = self._write_eval_files(run_id, processed_img_paths, eval_temp_dir)

        # 2. 評估（mAP / Precision / Recall）
        model = YOLO(model_name)
        print(f"\n==================== 開始評估測試集指標 (mAP / Precision / Recall) ====================")
        eval_run_name = f'test_evaluation_{run_id}'
        metrics = model.val(
            data=test_yaml_path,
            imgsz=self.img_size,
            name=eval_run_name,
            split='val',
            exist_ok=True          # 同名資料夾直接覆蓋，避免 Ultralytics 自動改名成 xxx2，導致圖表與 CSV 分家
        )
        overall = self._print_overall_report(run_id, metrics)

        # 3. CSV 報表
        csv_save_path = Path(f'{RUNS_DIR}/{eval_run_name}') / f'test_evaluation_{run_id}.csv'
        self._write_metrics_csv(metrics, csv_save_path, overall)
        print(f"\n[完成] 數據指標 CSV 檔已儲存至: {csv_save_path}")

        # 4. 預測並輸出劃框影像
        self._predict_and_save(model, eval_img_dir, confidence, run_id)
        print(f"\n[完成] 帶框照片已儲存至 {RUNS_DIR}/predict_{run_id}/")
        print(f"[完成] 評估圖表與 CSV 檔已儲存至 {RUNS_DIR}/{eval_run_name}/")

        # 5. 清理暫存檔與 GPU 顯存
        for f in (test_txt_path, test_yaml_path):
            if os.path.exists(f):
                os.remove(f)
        shutil.rmtree(eval_temp_dir, ignore_errors=True)
        torch.cuda.empty_cache()
        return metrics

    @staticmethod
    def _collect_test_images(image_source):
        """ 收集測試影像。回傳 (測試資料夾, 影像檔名清單)；找不到或沒有影像時印出訊息並回傳 None """
        test_path = Path(image_source)
        if not test_path.exists():
            print(f"錯誤：找不到測試集目錄或檔案: {image_source}")
            return None
        if test_path.is_dir():
            image_list = list_image_names(test_path)
        else:
            image_list = [test_path.name]
            test_path = test_path.parent
        if not image_list:
            print("未找到有效的測試影像。")
            return None
        return test_path, image_list

    def _resolve_eval_label_map(self, test_path, species):
        """ 決定這個測試集的標籤轉換表（混合模式會由資料夾名稱推測物種） """
        if self.is_mixed and species is None:
            species = self._infer_species(test_path.name)
        if species is not None and species not in self.SPECIES_RAW_IDS:
            raise ValueError(f"不支援的 species: {species}")
        label_map = self._resolve_label_map(species)
        if self.is_mixed:
            print(f"[標籤轉換] 測試集物種: {species if species else '未指定（視為已是最終 ID 或原始 10/11/20/21）'}")
        return label_map

    def _prepare_eval_dataset(self, test_path, image_list, label_map, eval_temp_dir):
        """ 建立評估暫存資料夾：影像轉灰階、標籤轉最終 ID。回傳 (暫存影像資料夾, 處理後影像的絕對路徑清單) """
        shutil.rmtree(eval_temp_dir, ignore_errors=True)   # 清掉上次中斷留下的殘留
        eval_img_dir = eval_temp_dir / 'images'
        eval_label_dir = eval_temp_dir / 'labels'
        eval_img_dir.mkdir(parents=True, exist_ok=True)
        eval_label_dir.mkdir(parents=True, exist_ok=True)

        processed_img_paths = []
        all_out_ids = set()
        print(f"\n==================== 正在預處理測試集影像 (自動轉灰階) ====================")
        for img_name in image_list:
            gray = read_gray_3ch(test_path / img_name)
            if gray is None:
                continue
            save_img_path = eval_img_dir / img_name
            cv2.imwrite(str(save_img_path), gray)
            processed_img_paths.append(str(save_img_path.resolve()))

            label_path = test_path / (Path(img_name).stem + '.txt')
            if label_path.exists():
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                new_lines, out_ids = convert_label_lines(lines, label_map)
                all_out_ids |= out_ids
                with open(eval_label_dir / label_path.name, 'w') as f:
                    f.writelines(new_lines)

        # 轉換後 ID 必須落在 0 ~ num_classes-1，否則 YOLO 會悄悄忽略或報錯
        bad_ids = all_out_ids - set(range(self.num_classes))
        if bad_ids:
            print(f"警告：轉換後仍有超出範圍的類別 ID: {sorted(bad_ids)}，請檢查測試集標籤或 species 設定")
        return eval_img_dir, processed_img_paths

    def _write_eval_files(self, run_id, processed_img_paths, eval_temp_dir):
        """ 寫評估用的影像清單檔與 data yaml；回傳 (清單檔路徑, yaml 路徑) """
        test_txt_path = f'./test_eval_list_{run_id}.txt'
        with open(test_txt_path, 'w') as f:
            for p in processed_img_paths:
                f.write(f"{p}\n")
        test_yaml_path = f'test_eval_{run_id}.yaml'
        write_data_yaml(test_yaml_path, test_txt_path, test_txt_path, eval_temp_dir,
                        self.num_classes, self.class_names, test=test_txt_path)
        return test_txt_path, test_yaml_path

    @staticmethod
    def _print_overall_report(run_id, metrics):
        """ 印出整體指標並回傳 (P, R, mAP50, mAP50-95) """
        p = float(metrics.box.mp)
        r = float(metrics.box.mr)
        map50 = float(metrics.box.map50)
        map95 = float(metrics.box.map)
        print("\n" + "=" * 50)
        print(f"【獨立測試集評估報告 - {run_id}】")
        print(f" Precision  : {p:.4f}")
        print(f" Recall     : {r:.4f}")
        print(f" mAP50      : {map50:.4f}")
        print(f" mAP50-95   : {map95:.4f}")
        print("=" * 50)
        return p, r, map50, map95

    def _write_metrics_csv(self, metrics, csv_save_path, overall):
        """ 寫 CSV：整體一列 + 各類別一列。測試集沒有標註的類別以 nan 表示。
            Precision/Recall/F1 為各類別在 peak F1 門檻下的值；@FIXED_CONF 欄為固定門檻下的值 """
        p, r, map50, map95 = overall
        csv_save_path.parent.mkdir(parents=True, exist_ok=True)
        p_fx, r_fx, f1_fx = self._prf_at_conf(metrics, FIXED_CONF)

        # 峰值 F1 所在的門檻（與 F1_curve.png 圖例一致）
        f1_mean_curve = np.asarray(metrics.box.f1_curve).mean(0)
        peak_conf = float(np.asarray(metrics.box.px)[smooth(f1_mean_curve, 0.1).argmax()])
        f1_all = float(np.mean(metrics.box.f1))   # 各類別 F1 的平均 (macro)

        with open(csv_save_path, mode='w', newline='', encoding='utf-8-sig') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['Class', 'Precision', 'Recall', 'F1', 'mAP50', 'mAP50-95',
                             f'Precision@{FIXED_CONF}', f'Recall@{FIXED_CONF}', f'F1@{FIXED_CONF}',
                             'peak_conf'])
            writer.writerow(['all (Overall)', f"{p:.4f}", f"{r:.4f}", f"{f1_all:.4f}",
                             f"{map50:.4f}", f"{map95:.4f}",
                             f"{p_fx.mean():.4f}", f"{r_fx.mean():.4f}", f"{f1_fx.mean():.4f}",
                             f"{peak_conf:.3f}"])

            cls_pos = {int(c): k for k, c in enumerate(metrics.box.ap_class_index)}
            for i, class_name in enumerate(self.class_names):
                if i in cls_pos:
                    k = cls_pos[i]
                    class_p, class_r = float(metrics.box.p[k]), float(metrics.box.r[k])
                    class_f1 = float(metrics.box.f1[k])
                    class_map50, class_map95 = float(metrics.box.ap50[k]), float(metrics.box.ap[k])
                    fx = (float(p_fx[k]), float(r_fx[k]), float(f1_fx[k]))
                else:   # 測試集沒有此類別的標註
                    class_p = class_r = class_f1 = class_map50 = class_map95 = float('nan')
                    fx = (float('nan'),) * 3

                print(f" 類別 [{class_name}] -> P: {class_p:.4f}, R: {class_r:.4f}, F1: {class_f1:.4f} | "
                      f"@{FIXED_CONF} P: {fx[0]:.4f}, R: {fx[1]:.4f}, F1: {fx[2]:.4f}")
                writer.writerow([class_name, f"{class_p:.4f}", f"{class_r:.4f}", f"{class_f1:.4f}",
                                 f"{class_map50:.4f}", f"{class_map95:.4f}",
                                 f"{fx[0]:.4f}", f"{fx[1]:.4f}", f"{fx[2]:.4f}", ""])

    def _predict_and_save(self, model, eval_img_dir, confidence, run_id):
        """ 對暫存測試影像做預測，輸出劃有目標框的影像，並逐張印出偵測結果 """
        print(f"\n==================== 開始繪製與輸出劃框影像 ====================")
        results = model.predict(
            source=str(eval_img_dir),
            imgsz=self.img_size,
            save=True,
            conf=confidence,
            project=RUNS_DIR,
            name=f'predict_{run_id}',
            exist_ok=True,
            batch=4,
            stream=True
        )
        for result in results:
            print(f"\n影像: {Path(result.path).name}")
            print_detections(model, result)

    # ------------------------------------------------------------------ 推論：單張
    def predict_image(self, model_name, image_path, confidence=0.5, save_result=True):
        """
        純推論單張影像（不需要標籤檔、不計算 mAP/Precision/Recall）：
        1. 自動轉灰階(3通道)，與訓練/評估時的前處理保持一致
        2. 回傳偵測結果，並印出每個偵測框的類別、信心值、座標
        3. 可選擇是否將劃框後的圖片存檔
        """
        img_path = Path(image_path)
        if not img_path.exists():
            print(f"錯誤：找不到影像檔案: {image_path}")
            return None

        gray_3ch = read_gray_3ch(img_path)   # 與訓練/測試時前處理一致
        if gray_3ch is None:
            print(f"錯誤：無法讀取影像: {image_path}")
            return None

        model = YOLO(model_name)
        results = model.predict(
            source=gray_3ch,
            imgsz=self.img_size,
            conf=confidence,
            save=save_result,
            project=RUNS_DIR,
            name='predict_single',
            exist_ok=True
        )

        print(f"\n影像: {img_path.name}")
        result = results[0]
        if len(result.boxes) == 0:
            print("  -> 未偵測到任何物件")
        print_detections(model, result)

        if save_result:
            print(f"[完成] 劃框結果已儲存至 {RUNS_DIR}/predict_single/")
        return results


# ============================================================================
# E. 完整流程範例
# ============================================================================
def main(mode='all', tune_iterations=50, base_model_name='yolo11m.pt'):
    """ 調參 → 帶入最佳超參數做 5 折交叉驗證 → 獨立測試集評估 """
    db = DetectBacteria(mode)   # 'cocci' / 'bacilli'：2 類；'all'：4 類
    if db.is_mixed:
        db.build_merged_dataset(force=True)       # 只在第一次需要，之後會自動略過

    # 基本訓練設定
    db.epoch = 50
    db.batch_size = 16  # 16 張圖片同時進行運算。
    db.img_size = 512
    bacteria_name = db.name

    # ---------- 步驟 1/3：超參數調優，結果自動存入 runs/detect/tune_{bacteria_name}/ ----------
    print(f"\n[步驟 1/3] 開始執行 YOLO11 自動超參數調優...")
    db.tune_hyperparameters(iterations=tune_iterations, base_model=base_model_name)

    # ---------- 步驟 2/3：自動讀取調參產生的最佳超參數 ----------
    best_hyp_yaml = Path(f'{RUNS_DIR}/tune_{bacteria_name}/best_hyperparameters.yaml')
    best_hyp = None
    if best_hyp_yaml.exists():
        print(f"\n[步驟 2/3] 成功找到調參結果！正在載入最佳超參數: {best_hyp_yaml}")
        with open(best_hyp_yaml, 'r', encoding='utf-8') as f:
            best_hyp = yaml.safe_load(f)
        print("載入的最佳超參數組合：")
        if isinstance(best_hyp, dict):
            for k, v in best_hyp.items():
                print(f"  - {k}: {v}")

    # ---------- 步驟 3/3：帶入最佳超參數，執行 5 折交叉驗證 ----------
    print(f"\n[步驟 3/3] 開始帶入最佳超參數進行 5 折交叉驗證訓練...")
    best_result = db.run_cross_validation(num_loops=5, base_model=base_model_name, custom_hyp=best_hyp)

    # ---------- 獨立測試集評估 ----------
    # 混合模式：合併測試集 + 各物種單獨評估（species 由資料夾名稱自動推測）；單物種模式：tag=None
    if db.is_mixed:
        test_sets = [('./test_image_all',     'merged'),
                     ('./test_image_cocci',   'cocci'),
                     ('./test_image_bacilli', 'bacilli')]
    else:
        test_sets = [(f'./test_image_{bacteria_name}', None)]

    for folder, tag in test_sets:
        print(f"\n開始執行測試集評估與推論 (模型: {best_result['model_path']}, 測試集: {folder})...")
        db.detect_bacteria(
            model_name=best_result['model_path'],
            image_source=folder,
            confidence=0.5,
            tag=tag
        )


# ============================================================================
# F. 進入點
# ============================================================================
if __name__ == '__main__':
    main(mode='all')