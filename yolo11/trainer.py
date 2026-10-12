# -*- coding: utf-8 -*-
"""
trainer：TrainerMixin（超參數調優、單折訓練、K-Fold 交叉驗證）

這是 Mixin，不能單獨使用，必須由 DetectBacteria 繼承。
方法內的 self.xxx 由宿主類別DetectBacteria提供：
  self.name 細菌種類字串
  / dataset_dir (訓練資料集根目錄)/ image_dir(原始影像來源資料夾) / label_dir (原始標籤來源資料夾)  路徑屬性（detector.py）
  self.class_mapping     原始標籤 ID → 最終類別 ID(字典)
  / class_names / num_classes    模式設定（detector.py）
  self.epoch / img_size / batch_size                訓練設定（detector.py）
  self.get_kfold_splits                             K-Fold 切分（DatasetBuilderMixin，來自mixer.py）
"""
import json
import os
import shutil
from pathlib import Path
import torch #判斷當前的硬體環境是 CPU 還是 GPU
from ultralytics import YOLO #載入模型與預訓練權重
from .config import ( #K-Fold 切分種子、折數、訓練輪數、分析結果根目錄(相對路徑).../run/detect
    BASE_SEED, N_SPLITS, TUNE_EPOCHS, RUNS_DIR,
    SEARCH_SPACE, NON_OVERRIDABLE_HYP_KEYS,   # 超參數搜尋邊界、訓練時不可被覆蓋鍵
)
#把一組影像（檔名清單）/與標籤檔處理後放進暫存資料集、寫入影像路徑清單檔、寫入 YOLO 資料集設定檔
from .staging import stage_split, write_path_list, write_data_yaml


class TrainerMixin:

    # ----------- 訓練
    #輸入:這折的訓練集/驗證集影像檔名清單、訓練集/驗證集清單文字檔的儲存路徑
    def train_yolo_single_iteration(self, image_train, image_val, train_txt_path, val_txt_path,
                                    seed, model_name='yolo11m.pt', custom_hyp=None): #隨機種子、YOLO 模型權重名稱、透過基因演算法調校好的字典
        """訓練單一折"""
        name = self.name
        temp_root = self.dataset_dir / f'temp_seed_{seed}'    # 依 seed 隔離的暫存根目錄
        temp_img, temp_lbl = temp_root / 'images', temp_root / 'labels'
        temp_img.mkdir(parents=True, exist_ok=True)
        temp_lbl.mkdir(parents=True, exist_ok=True)
#  回傳(暫存)灰階後影像的絕對路徑清單
        train_paths = stage_split(image_train, self.image_dir, self.label_dir, temp_img, temp_lbl, self.class_mapping)
        val_paths = stage_split(image_val, self.image_dir, self.label_dir, temp_img, temp_lbl, self.class_mapping)
#輸入:包含影像路徑的文字檔、(暫存)處理後影像的絕對路徑清單
        write_path_list(train_txt_path, train_paths)
        write_path_list(val_txt_path, val_paths)
        yaml_path = f'bacteria_{name}_seed_{seed}.yaml'
        #輸入:YAML檔案路徑、訓練/驗證/測試圖片清單絕對路徑文字檔、資料集根目錄、類別總數、類別名稱清單
        write_data_yaml(yaml_path, train_txt_path, val_txt_path, temp_root, self.num_classes, self.class_names)

        model = YOLO(model_name)
        train_args = { #args.yaml
            'data': yaml_path, #每折交叉驗證的資料設定檔(專案根目錄下)
            'epochs': self.epoch, #（訓練輪數 / 週期）
            'imgsz': self.img_size,
            'batch': self.batch_size, #批次大小:更新一次權重的單位
            'name': f'bacteria_model_{name}_seed_{seed}', #每一折交叉驗證結果子資料夾
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'val': True,
            'seed': seed,
        }
        # .yaml = 只有架構（從頭訓練）；.pt = 載入預訓練權重
        train_args['pretrained'] = not str(model_name).endswith(('.yaml', '.yml')) #反轉布林值
        if custom_hyp: #透過基因演算法調校好的字典
        #.items():把一個字典裡面的鍵和值提取出來，變成一個個的 (key, value) 小組
            train_args.update({k: v for k, v in custom_hyp.items() if k not in NON_OVERRIDABLE_HYP_KEYS})

        print(f"\n--- 開始訓練 {name} 模型 (Seed: {seed}) ---")
        results = model.train(**train_args)  # 啟動訓練；Ultralytics 內部會自動建立驗證器，每個 epoch 結束與訓練結束時驗證

        try: #.results_dict[]:把這場訓練的所有成績數據，全部整理並存放在字典裡
            current_map50 = results.results_dict['metrics/mAP50(B)'] #value(模型優劣的判段標準」)
        except KeyError: #last.pt
            current_map50 = results.box.map50 #用物件屬性找
        return results.save_dir, yaml_path, current_map50, temp_root #回傳:每一折交叉驗證結果完整資料夾路徑、 每折交叉驗證的資料設定檔(專案根目錄下)、模型優劣的判段標準、影像資料夾下依 seed 隔離的暫存根目錄

    def evaluate_model(self, model_path, yaml_path, experiment_name):
        """載入模型並對驗證集進行單獨評估，自動生成混淆矩陣與 PR 曲線；回傳 mAP50"""
        model = YOLO(model_path) #best.pt
        metrics = model.val(data=yaml_path, name=experiment_name) #模型獨立驗證與評估(新建立的驗證器)
        return metrics.box.map50 

    def run_cross_validation(self, num_loops=5, base_model='yolo11m.pt', custom_hyp=None):
        """執行K-Fold 交叉驗證，並以各折驗證集 mAP50 最高者作為最佳模型"""
        name = self.name
        best_map50 = -1.0 #確保第一折必定能被記錄至最佳模型
        best_model_info = {} #空字典

        print(f"\n==================== 開始執行 {num_loops} 折真實 K-Fold 交叉驗證 ====================")
        splits = self.get_kfold_splits(self.image_dir, n_splits=num_loops, seed=BASE_SEED) #回傳一個包含 5 個元素的 Python 清單
#enumerate() 是 Python 的內置函數。它會把一個可疊代物件（如清單）包裝起來，在迴圈進行時，同時提供「索引編號（Index）」與「當前元素」。
        for i, (image_train, image_val, _, _) in enumerate(splits):
            fold = i + 1
            seed = BASE_SEED + i  # 給每一摺一個標記用的 seed
            print(f"\n>>> 【第 {fold} / {num_loops} 折交叉驗證】 <<<")
            train_txt = str(self.dataset_dir / f'image_train_fold_{fold}.txt')
            val_txt = str(self.dataset_dir / f'image_val_fold_{fold}.txt')
            #回傳:每一折交叉驗證資料夾路徑、 每折交叉驗證的yaml設定檔(專案根目錄下)、模型優劣的判段標準、影像資料夾下依 seed 隔離的暫存根目錄
            save_dir, yaml_path, _, temp_root = self.train_yolo_single_iteration( #訓練單一折
                image_train, image_val, train_txt, val_txt, seed, base_model, custom_hyp=custom_hyp)

            exp_eval_name = f'eval_report_{name}_fold_{fold}' #runs/detect
            best_pt = os.path.join(save_dir, 'weights', 'best.pt') #\runs\detect\bacteria_model_all_seed_42\weights...
            print(f"正在為第 {fold} 折生成獨立評估圖表...")
            eval_map50 = self.evaluate_model(best_pt, yaml_path, exp_eval_name) #載入模型並對驗證集進行單獨評估，自動生成混淆矩陣與 PR 曲線；回傳 mAP50
            print(f"-> 本摺驗證結果 mAP50: {eval_map50:.4f}")
            shutil.rmtree(temp_root, ignore_errors=True)   # 驗證完才能刪暫存資料集

            if eval_map50 > best_map50:   # 比較並篩選出表現最優異的模型
                best_map50 = eval_map50
                best_model_info = { #..\runs\detect\best_all
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
#內部私有方法
    def _print_cv_report(self, num_loops, best_map50, info): #印出交叉驗證報告
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
        """把最佳權重另存到 runs/detect/best_{name}/best.pt，並寫 best_info.json"""
        out_dir = Path(f'{RUNS_DIR}/best_{self.name}')
        out_dir.mkdir(parents=True, exist_ok=True) #parents=True自動建立上層資料夾
        shutil.copy(info['model_path'], out_dir / 'best.pt') #.copy(src, dst)：它會把來源檔案（Source）完整複製到目標位置（Destination）。
        with open(out_dir / 'best_info.json', 'w', encoding='utf-8') as f:
            json.dump(info, f, ensure_ascii=False, indent=2) #indent=2（排版縮排
        print(f" 最佳權重已另存: {out_dir / 'best.pt'}")

    #------------ 調參
    def tune_hyperparameters(self, iterations=30, base_model='yolo11m.pt'):
        """針對細菌影像特徵進行自動超參數搜尋 (Genetic Algorithm)，只使用 K-Fold 的第 1 折"""
        name = self.name
        print(f"\n==================== 開始超參數自動調參搜尋 (Iterations: {iterations}) ====================")

        # 1. 專屬該次調參的暫存資料集根目錄（若有殘留先清除重建）
        tune_root = Path(f'./dataset_{name}_tune_temp')
        shutil.rmtree(tune_root, ignore_errors=True)
        tune_img, tune_lbl = tune_root / 'images', tune_root / 'labels'
        tune_img.mkdir(parents=True, exist_ok=True)
        tune_lbl.mkdir(parents=True, exist_ok=True)

        # 2. 取 K-Fold 的第一摺，處理影像與標籤
        image_train, image_val, _, _ = self.get_kfold_splits(self.image_dir, n_splits=N_SPLITS, seed=BASE_SEED)[0] #回傳一個包含 5 個元素的 Python 清單
        #  回傳(暫存)灰階後影像的絕對路徑清單
        train_paths = stage_split(image_train, self.image_dir, self.label_dir, tune_img, tune_lbl, self.class_mapping)
        val_paths = stage_split(image_val, self.image_dir, self.label_dir, tune_img, tune_lbl, self.class_mapping)

        # 3. 清單檔與調參專用 data.yaml
        train_txt, val_txt = tune_root / 'tune_train.txt', tune_root / 'tune_val.txt' #多重變數指派<-Tuple
        #輸入:包含影像路徑的文字檔、(暫存)處理後影像的絕對路徑清單
        write_path_list(train_txt, train_paths) #寫入影像路徑清單檔
        write_path_list(val_txt, val_paths)
        tune_yaml_path = f'tune_{name}.yaml'
        #輸入:YAML檔案路徑、訓練/驗證圖片清單絕對路徑文字檔、資料集根目錄、類別總數、標籤類別名稱清單
        write_data_yaml(tune_yaml_path, train_txt, val_txt, tune_root, self.num_classes, self.class_names) #寫入 YOLO 資料集設定檔

        # 4. 執行基因演算法調參
        model = YOLO(base_model)
        model.tune( #執行基因演算法超參數調參
            data=tune_yaml_path, # 指定資料集設定檔（YAML）路徑，告訴模型訓練與驗證資料在哪裡 
            model=base_model,
            epochs=TUNE_EPOCHS,
            iterations=iterations, 
            optimizer='AdamW',
            plots=True,# 是否在調參結束後自動產生並儲存搜尋分析圖表（如各參數表現分佈圖）
            save=True, #best.pt/last.pt
            val=True, # 每次訓練完是否要在驗證集上進行評估，藉此計算該組超參數的分數（Fitness）
            space=SEARCH_SPACE,
            name=f'tune_{name}', #調參結果儲存的子資料夾名稱
            pretrained=not str(base_model).endswith(('.yaml', '.yml')),
        ) #.endswith() 括號內可以放一個元組（Tuple），只要字串字尾符合其中任何一個副檔名，就會回傳 True，否則回傳 False。

        # 5. 清理調參專用暫存資料集根目錄與 yaml 檔
        shutil.rmtree(tune_root, ignore_errors=True)
        if os.path.exists(tune_yaml_path):
            os.remove(tune_yaml_path) # 刪除專案根目錄下暫存的 yaml 檔
        print(f"\n[完成] 超參數最佳化搜尋結束，結果已寫入 {RUNS_DIR}/tune_{name}/best_hyperparameters.yaml")