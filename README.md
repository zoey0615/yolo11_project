# yolo11_project：細菌顯微影像 YOLO11 偵測流程

以 [Ultralytics YOLO11](https://docs.ultralytics.com/) 為基礎，針對**球菌（cocci）**與**桿菌（bacilli）**顯微影像，偵測 **Single（單一菌體）** 與 **Colony（菌落）** 兩種目標。
套件 `yolo11` 把「資料準備 → 調參 → K-Fold 交叉驗證訓練 → 測試集評估 → 推論」整合成一個 `DetectBacteria` 類別。

## 功能特色

- **三種模式**：`cocci`（2 類）、`bacilli`（2 類）、`all`（球菌＋桿菌，4 類）
- **標籤自動轉換**：原始 ID（10/11、20/21）自動轉成最終類別 ID，也相容已正規化的 0/1
- **K-Fold 交叉驗證**：單物種用 `KFold`，混合模式用 `StratifiedKFold`（依物種分層）
- **三來源混合資料集**：DIBAS（真實）、MATLAB、Diffusion（合成）可自訂比例，每折每類別訓練總量固定為 N，驗證集永遠只用真實影像
- **超參數調優**：針對顯微影像設計搜尋空間，使用 YOLO 內建基因演算法
- **自訂適應度（fitness）**：以 `0.3 × Recall + 0.7 × mAP50-95` 作為調參評分
- **測試集評估**：輸出 mAP / Precision / Recall / F1 的 CSV，以及劃框影像
- **統一前處理**：所有影像一律轉灰階並複製成 3 通道

## 專案結構

```
yolo11_project/
├── main.py                     # 主程式（流程進入點）
└── yolo11/                     # 核心套件
    ├── __init__.py             # 套件入口，匯出 DetectBacteria
    ├── metrics_patch.py        # 環境設定 + 自訂 fitness 猴子補丁
    ├── config.py               # 常數與模式設定
    ├── labels.py               # 標籤／檔名純字串處理
    ├── staging.py              # 影像灰階化、暫存資料集、YAML 產生
    ├── detector.py             # DetectBacteria 主類別
    ├── dataset_builder.py      # DatasetBuilderMixin：標籤轉換、合併、K-Fold
    ├── mixer.py                # MixerMixin：三來源固定總量混合資料集
    ├── trainer.py              # TrainerMixin：調參、單折訓練、交叉驗證
    ├── evaluator.py            # EvaluatorMixin：測試集評估、CSV、單張推論
    └── refacter_all_process.py # 重構流程腳本（請自行補充說明）
```

### 模組依賴關係

```
config ← labels ← staging ← dataset_builder / mixer / trainer / evaluator ← detector ← __init__
```

`config.py` 是最底層（只依賴標準函式庫）；`metrics_patch.py` 必須在 `__init__.py` 最先載入。

## 各 .py 檔說明

| 檔案 | 說明 |
|---|---|
| `__init__.py` | 套件入口。先載入 `metrics_patch`，再匯出 `DetectBacteria`，使用 `from yolo11 import DetectBacteria` |
| `metrics_patch.py` | 設定 `KMP_DUPLICATE_LIB_OK`（避免 OpenMP 衝突）、matplotlib 使用 `Agg` 後端；並以猴子補丁替換 Ultralytics `Metric.fitness`，權重為 `[P, R, mAP50, mAP50-95] = [0, 0.3, 0, 0.7]`，NaN 轉 0 |
| `config.py` | 所有常數：影像副檔名、種子、K-Fold 折數、調參 epoch、固定信心門檻、輸出路徑、`SEARCH_SPACE`、`MODES`、預設訓練參數、三來源名稱 |
| `labels.py` | 純字串處理函式（易於單元測試）：`label_name_of`、`species_of`、`convert_label_lines` |
| `staging.py` | 影像與暫存資料集處理：副檔名判斷、影像列舉、灰階轉換、暫存資料集建立、路徑清單檔與 YOLO `data.yaml` 輸出 |
| `detector.py` | `DetectBacteria`：只負責設定與路徑，功能由四個 Mixin 提供 |
| `dataset_builder.py` | `DatasetBuilderMixin`：標籤轉換、`dataset_all` / `test_image_all` 合併、K-Fold 切分 |
| `mixer.py` | `MixerMixin`：三來源來源池、比例設定、固定總量 N 的每折抽樣；覆蓋 `get_kfold_splits` |
| `trainer.py` | `TrainerMixin`：超參數調優、單折訓練、K-Fold 交叉驗證、最佳模型另存 |
| `evaluator.py` | `EvaluatorMixin`：獨立測試集評估、CSV 報表、劃框影像、單張推論 |

## 物件與函式說明

### `config.py`

| 名稱 | 說明 |
|---|---|
| `IMAGE_EXTS` | 支援的副檔名 `.jpg` `.png` `.jpeg` |
| `BASE_SEED` / `N_SPLITS` | K-Fold 切分種子（42）／調參時的折數（5，取第 1 折）；第 i 折訓練種子 = `BASE_SEED + i` |
| `TUNE_EPOCHS` | 調參時每次嘗試的訓練輪數（20） |
| `FIXED_CONF` | CSV 中 `@固定門檻` 欄位使用的信心門檻（0.5） |
| `RUNS_DIR` | Ultralytics 輸出根目錄 `./runs/detect` |
| `SEARCH_SPACE` | 調參搜尋範圍：`lr0`、`mosaic`、`degrees`、`scale`、`cls`、`iou` |
| `NON_OVERRIDABLE_HYP_KEYS` | 調參結果套用到訓練時不可被覆蓋的鍵（`data`、`epochs`、`imgsz` 等） |
| `SPECIES_RAW_IDS` | 各物種原始標籤 ID：`cocci=(10,11)`、`bacilli=(20,21)`（Single, Colony） |
| `ModeConfig` | 凍結的 dataclass，描述單一模式（名稱、banner、類別對應、類別名稱、是否混合） |
| `MODES` | `bacilli` / `cocci`（2 類）、`all`（4 類：Cocci_Single、Cocci_Colony、Bacilli_Single、Bacilli_Colony） |
| `DEFAULT_TRAINING` | `img_size=512`、`epoch=50`、`batch_size=16` |
| `SOURCE_NAMES` / `MIX_ROOT` | 三來源名稱 `dibas`、`matlab`、`diffusion`／混合資料集根目錄 `dataset_mix` |

### `labels.py`

| 函式 | 說明 |
|---|---|
| `label_name_of(img_name)` | 影像檔名 → 標籤檔名（`cocci_001.jpg` → `cocci_001.txt`） |
| `species_of(filename)` | 由檔名前綴取得物種（`cocci_001.jpg` → `cocci`） |
| `convert_label_lines(lines, label_map)` | 轉換 YOLO 標籤第一欄的類別 ID；回傳（新標籤行, 輸出 ID 集合）。不在表內的 ID 原樣保留，無法解析或空白行略過 |

### `staging.py`

| 函式 | 說明 |
|---|---|
| `is_image_name(name)` | 副檔名是否為支援格式 |
| `list_image_names(folder)` | 回傳資料夾內排序後的影像檔名（確保每次順序一致） |
| `to_gray_3ch(img)` / `read_gray_3ch(path)` | BGR → 灰階 → 複製成 3 通道；後者讀取失敗回傳 `None` |
| `stage_image(src, dst)` | 影像轉灰階後存到暫存位置 |
| `stage_label(src, dst, label_map)` | 轉換標籤 ID 後寫到暫存位置 |
| `stage_split(...)` | 把一組檔名的影像與標籤處理後放進暫存資料集，回傳影像絕對路徑清單 |
| `write_path_list(path, paths)` | 寫入影像路徑清單檔（每行一個） |
| `write_data_yaml(...)` | 寫入 YOLO 資料集設定檔 |

### `detector.py`：`DetectBacteria`

```python
class DetectBacteria(MixerMixin, DatasetBuilderMixin, TrainerMixin, EvaluatorMixin)
```

`MixerMixin` 排在最前面，才能覆蓋 `DatasetBuilderMixin.get_kfold_splits`；未呼叫 `set_mix()` 時會自動退回原版行為。

| 成員 | 說明 |
|---|---|
| `__init__(mode)` | 傳入 `'cocci'` / `'bacilli'` / `'all'`；不帶參數或不支援的字串預設為 `bacilli` |
| `_apply_mode(mode)` | 依 `MODES` 套用設定（影像大小、epoch、batch、類別對應與名稱） |
| `init_bacilli()` / `init_cocci()` / `init_all()` | 舊版相容的模式切換方法 |
| `name` | 目前模式名稱，用於所有輸出資料夾與檔名 |
| `dataset_dir` / `image_dir` / `label_dir` | 資料集、影像、標籤路徑；`custom_dataset_dir` 有設定時優先使用 |

### `dataset_builder.py`：`DatasetBuilderMixin`

| 方法 | 說明 |
|---|---|
| `build_merged_dataset(species_list, force, strict)` | 合併 `dataset_{sp}/` 與 `test_image_{sp}/` 成 `dataset_all/`、`test_image_all/`；檔名加物種前綴，標籤轉成 0~3。`strict=True` 遇到異常標籤會中止並清除半成品 |
| `get_kfold_splits(image_dir, n_splits, seed)` | 回傳 `[(訓練影像, 驗證影像, 訓練標籤, 驗證標籤), ...]`；單物種用 `KFold`，混合模式用 `StratifiedKFold` |
| `save_txt(...)` | 把檔名絕對路徑存成文字檔（舊版相容） |
| `_species_label_map` / `_infer_species` / `_resolve_label_map` | 取得物種標籤轉換表、由資料夾名稱推測物種 |
| `_convert_label_file` / `_merge_one_folder` | 單一標籤檔轉換、單一資料夾合併 |
| `_check_label_problems` / `_report_normalized_labels` / `_count_label_ids` | 異常標籤檢查、已正規化 0/1 標籤回報、各類別實例數統計 |

### `mixer.py`：`MixerMixin`

| 名稱 | 說明 |
|---|---|
| `source_of(name)` | `cocci_dibas_001.jpg` → `dibas`；舊格式回傳 `None` |
| `greedy_select(items, targets, seed)` | 貪心抽圖：固定種子隨機排序，加入後任何類別超標就跳過，直到各類別達標 |
| `build_source_pool(sources, force)` | 把三來源完整轉標籤後複製到 `dataset_mix/pool_{name}/`（只需做一次，所有比例共用） |
| `set_mix(ratios, tag, seed)` | 設定實驗比例，三個比例總和須為 1 |
| `build_merged_test(...)` | 只合併測試集成 `test_image_all/` |
| `get_kfold_splits(...)` | 僅對真實 DIBAS 切折；N = 所有折訓練集中各類別實例數的最小值；每折從 DIBAS 訓練折抽 `N×r_dibas`、從 MATLAB / Diffusion 抽 `N×r`；切分結果存成 `dataset_mix/splits_{tag}.json` |

### `trainer.py`：`TrainerMixin`

| 方法 | 說明 |
|---|---|
| `tune_hyperparameters(iterations, base_model)` | 取第 1 折，用基因演算法搜尋超參數；結果寫入 `runs/detect/tune_{name}/best_hyperparameters.yaml` |
| `train_yolo_single_iteration(...)` | 訓練單一折（建立暫存資料集 → 寫 YAML → 訓練），可帶入 `custom_hyp` |
| `evaluate_model(model_path, yaml_path, name)` | 載入權重對驗證集評估，產生混淆矩陣與 PR 曲線，回傳 mAP50 |
| `run_cross_validation(num_loops, base_model, custom_hyp)` | 執行 K-Fold，以各折驗證 mAP50 最高者為最佳模型 |
| `_save_best_model(info)` | 最佳權重另存到 `runs/detect/best_{name}/best.pt` 並寫 `best_info.json` |

### `evaluator.py`：`EvaluatorMixin`

| 方法 | 說明 |
|---|---|
| `detect_bacteria(model_name, image_source, confidence, tag, species)` | 測試集完整評估：灰階預處理 → 標籤轉換 → `model.val()` → 輸出 CSV → 輸出劃框影像 → 清理暫存 |
| `predict_image(model_name, image_path, confidence, save_result)` | 單張推論（不需標籤，不算 mAP），印出類別、信心值與座標 |
| `print_detections(model, result)` | 模組層級函式，印出每個偵測框 |
| `_write_metrics_csv` / `_prf_at_conf` | 輸出整體與各類別指標；Precision/Recall/F1 為 peak-F1 門檻下的值，`@0.5` 欄為固定門檻下的值，測試集無標註的類別以 `nan` 表示 |

## 環境安裝

建議 Python 3.9 以上，並依自己的 CUDA 版本先安裝 PyTorch。

```bash
pip install ultralytics opencv-python numpy scikit-learn pyyaml matplotlib
```

## 資料夾準備

專案根目錄（執行 `main.py` 的位置）需準備：

```
dataset_cocci/
├── images/        # *.jpg / *.png
└── labels/        # YOLO 格式 *.txt（類別 ID 為 10/11 或 0/1）
dataset_bacilli/
├── images/
└── labels/        # 類別 ID 為 20/21 或 0/1
test_image_cocci/    # 測試影像 + 同名 .txt
test_image_bacilli/
```

標籤格式為 YOLO：`class_id x_center y_center width height`（皆為 0~1 正規化）。

## 使用方式

### 1. 單一物種（cocci 或 bacilli）

```python
from yolo11 import DetectBacteria

db = DetectBacteria('cocci')   # 或 'bacilli'

# (選用) 超參數調優
db.tune_hyperparameters(iterations=30, base_model='yolo11m.pt')

# 5 折交叉驗證訓練
info = db.run_cross_validation(num_loops=5, base_model='yolo11m.pt')

# 測試集評估
db.detect_bacteria(model_name=info['model_path'], confidence=0.5)
```

### 2. 混合模式（球菌＋桿菌，4 類）

```python
db = DetectBacteria('all')
db.build_merged_dataset()                      # 合併 dataset_all / test_image_all
info = db.run_cross_validation(num_loops=5)

# 分別評估兩個物種的測試集
db.detect_bacteria(info['model_path'], image_source='./test_image_cocci',   tag='cocci',   species='cocci')
db.detect_bacteria(info['model_path'], image_source='./test_image_bacilli', tag='bacilli', species='bacilli')
```

### 3. 三來源混合比例實驗（DIBAS / MATLAB / Diffusion）

```python
db = DetectBacteria('cocci')

SOURCES = {
    'dibas':     [{'images': 'raw/dibas/images',     'labels': 'raw/dibas/labels'}],
    'matlab':    [{'images': 'raw/matlab/images',    'labels': 'raw/matlab/labels'}],
    'diffusion': [{'images': 'raw/diffusion/images', 'labels': 'raw/diffusion/labels'}],
}
# 混合模式（DetectBacteria('all')）時，每個資料夾需額外指定 'species': 'cocci' 或 'bacilli'

db.build_source_pool(SOURCES)                                   # 只需建一次
db.set_mix({'dibas': 0.5, 'matlab': 0.25, 'diffusion': 0.25}, tag='mix_50_25_25')
info = db.run_cross_validation(num_loops=5)
```

### 4. 單張推論

```python
db.predict_image('runs/detect/best_cocci/best.pt', 'sample.jpg', confidence=0.5)
```

### 5. 使用調參結果訓練

把 `best_hyperparameters.yaml` 讀成字典後帶入：

```python
import yaml
hyp = yaml.safe_load(open('runs/detect/tune_cocci/best_hyperparameters.yaml'))
db.run_cross_validation(num_loops=5, custom_hyp=hyp)
```

## 輸出結果

| 路徑 | 內容 |
|---|---|
| `runs/detect/bacteria_model_{name}_seed_{seed}/` | 各折訓練結果與權重 |
| `runs/detect/eval_report_{name}_fold_{k}/` | 各折評估圖表（混淆矩陣、PR 曲線） |
| `runs/detect/best_{name}/best.pt`、`best_info.json` | 交叉驗證中表現最佳的模型與資訊 |
| `runs/detect/tune_{name}/` | 調參結果 |
| `runs/detect/test_evaluation_{run_id}/` | 測試集評估圖表與 CSV |
| `runs/detect/predict_{run_id}/` | 劃框後的測試影像 |
| `dataset_mix/splits_{tag}.json` | 混合實驗每折的訓練／驗證檔名紀錄 |

## 注意事項

- 影像一律以**灰階 3 通道**輸入，推論時前處理需與訓練一致（`predict_image` 已自動處理）。
- 混合模式的檔名前綴（`cocci_` / `bacilli_`）是分層切折的依據，請勿隨意改名。
- 調參與訓練過程會在資料夾內產生暫存檔，正常結束會自動清除；若中途中斷，可手動刪除 `temp_seed_*`、`temp_test_eval_*`、`*_tune_temp` 等殘留資料夾。
- 來源路徑或內容改變時，`build_source_pool` 與 `build_merged_dataset` 需加上 `force=True` 才會重建。

