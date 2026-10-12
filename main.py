# -*- coding: utf-8 -*-
"""
main：三來源比例實驗（來源池 → [每模式一次]調參 → 5 折交叉驗證 → 獨立測試集評估）

放在 yolo11/ 套件的「外面」，請在 yolo11_project/ 根目錄執行。
cocci / bacilli / all 三個模式請分開執行（改 MODE 與 SOURCES 即可）。
"""
import shutil
from pathlib import Path

import yaml

from yolo11 import DetectBacteria
from yolo11.config import RUNS_DIR

if __name__ == '__main__':
    # ===================== 要改的參數都在這裡 =====================
    MODE = 'all'                    # 'cocci' / 'bacilli'：2 類；'all'：4 類
    PRETRAINED = False                    # False → 對照組：不使用預訓練權重（從頭訓練）
    BASE_MODEL = 'yolo11m.pt' if PRETRAINED else 'yolo11m.yaml'
    SUFFIX = '' if PRETRAINED else '_scratch'

    # 三個來源比例（總和必須 = 1）。每組實驗每類別總量都是 N，只改變來源組成
    RATIOS = {'dibas': 1.0, 'matlab': 0.0, 'diffusion': 0.0}

    # 三個來源的輸入路徑（images 與 labels 子資料夾）。
    # 'all'：每個資料夾都要有 species；'cocci' / 'bacilli'：只放該物種的資料夾，species 可省略。
    SOURCES = {
        'dibas': [
            {'images': 'dataset_cocci/images',   'labels': 'dataset_cocci/labels',   'species': 'cocci'},
            {'images': 'dataset_bacilli/images', 'labels': 'dataset_bacilli/labels', 'species': 'bacilli'},
        ],
        'matlab': [
            {'images': 'matlab_cocci/images',    'labels': 'matlab_cocci/labels',    'species': 'cocci'},
            {'images': 'matlab_bacilli/images',  'labels': 'matlab_bacilli/labels',  'species': 'bacilli'},
        ],
        'diffusion': [
            {'images': 'diffusion_cocci/images',   'labels': 'diffusion_cocci/labels',   'species': 'cocci'},
            {'images': 'diffusion_bacilli/images', 'labels': 'diffusion_bacilli/labels', 'species': 'bacilli'},
        ],
    }

    SAMPLE_SEED = 42                # 抽圖種子，固定才能重現

    # ---- 調參設定 ----
    TUNE_ITERATIONS = 50
    # True ：調參資料組成 = 上面的 RATIOS，每個比例組各自調參 50 輪，結果存 tune_{MODE}_{tag}/
    # False：整個模式共用一次調參（組成見 SHARED_TUNE_RATIOS），結果存 tune_{MODE}/
    TUNE_SAME_AS_RATIOS = True
    SHARED_TUNE_RATIOS = {'dibas': 1.0, 'matlab': 0.0, 'diffusion': 0.0}   # 僅上面為 False 時使用
    FORCE_RETUNE = False            # True → 刪掉舊的調參結果並重新跑 50 輪
    # ==============================================================

    tag = f"d{round(RATIOS['dibas']*100)}_m{round(RATIOS['matlab']*100)}_f{round(RATIOS['diffusion']*100)}{SUFFIX}"

    db = DetectBacteria(MODE)
    db.epoch, db.batch_size, db.img_size = 50, 16, 512

    # ---------- 步驟 0：來源池（完整複製三來源，每個模式只需一次）與測試集 ----------
    db.build_source_pool(SOURCES, force=False)     # 來源路徑/內容有改時請改 True
    if db.is_mixed:
        db.build_merged_test(force=False)

    # ---------- 步驟 1：超參數調優 ----------
    tune_ratios = RATIOS if TUNE_SAME_AS_RATIOS else SHARED_TUNE_RATIOS
    tune_name = f'{MODE}_{tag}' if TUNE_SAME_AS_RATIOS else f'{MODE}{SUFFIX}'
    tune_dir = Path(f'{RUNS_DIR}/tune_{tune_name}')
    best_hyp_yaml = tune_dir / 'best_hyperparameters.yaml'
    if FORCE_RETUNE:
        shutil.rmtree(tune_dir, ignore_errors=True)   # 否則 Ultralytics 會自動改名成 tune_xxx2

    if best_hyp_yaml.exists():
        print(f"\n[步驟 1/3] 已有調參結果，略過調參: {best_hyp_yaml}")
    else:
        print(f"\n[步驟 1/3] 開始超參數調優（{TUNE_ITERATIONS} 輪），資料組成: {tune_ratios}")
        db.set_mix(tune_ratios, f'tune_{tune_name}', seed=SAMPLE_SEED)
        db.bacteria_names = [tune_name]            # 調參輸出到 tune_{tune_name}
        db.tune_hyperparameters(iterations=TUNE_ITERATIONS, base_model=BASE_MODEL)

    # ---------- 步驟 2：讀取最佳超參數 ----------
    best_hyp = None
    if best_hyp_yaml.exists():
        print(f"\n[步驟 2/3] 載入最佳超參數: {best_hyp_yaml}")
        with open(best_hyp_yaml, 'r', encoding='utf-8') as f:
            best_hyp = yaml.safe_load(f)
        if isinstance(best_hyp, dict):
            for k, v in best_hyp.items():
                print(f"  - {k}: {v}")
    else:
        print(f"\n[步驟 2/3] 找不到 {best_hyp_yaml}，使用預設超參數")

    # ---------- 步驟 3：本次比例組別的 5 折交叉驗證 ----------
    db.set_mix(RATIOS, tag, seed=SAMPLE_SEED)
    db.bacteria_names = [f'{MODE}_{tag}']          # 讓 runs/ 輸出依實驗組別分開
    print(f"\n[步驟 3/3] 開始 5 折交叉驗證訓練（{tag}）...")
    best_result = db.run_cross_validation(num_loops=5, base_model=BASE_MODEL, custom_hyp=best_hyp)

    # ---------- 獨立測試集評估（一律為真實 DIBAS 測試集） ----------
    if db.is_mixed:
        test_sets = [('./test_image_all', 'merged'),
                     ('./test_image_cocci', 'cocci'),
                     ('./test_image_bacilli', 'bacilli')]
    else:
        test_sets = [(f'./test_image_{MODE}', None)]

    for folder, t in test_sets:
        print(f"\n開始測試集評估 (模型: {best_result['model_path']}, 測試集: {folder})...")
        db.detect_bacteria(model_name=best_result['model_path'],
                           image_source=folder, confidence=0.5, tag=t)