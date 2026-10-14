# -*- coding: utf-8 -*-
"""
evaluator：EvaluatorMixin（獨立測試集評估、CSV 報表、劃框輸出、單張推論）

這是 Mixin，不能單獨使用，必須由 DetectBacteria 繼承。
方法內的 self.xxx 由宿主類別(detector.py)提供：

  self.name (菌種/資料夾名稱)/ img_size / num_classes / class_names(標籤類別名稱值組)   設定與路徑 源自config.py、detector.py
  self.is_mixed /   模式設定 源自config.py、detector.py
  SPECIES_RAW_IDS    # 各物種在原始資料集中的標籤 ID 對照表：(Single, Colony) SPECIES_RAW_IDS = {'cocci': (10, 11), 'bacilli': (20, 21)}              
  用 self. 的（都是 DetectBacteria子類別 物件上的）屬性與方法
  self._infer_species(由資料夾名稱推測物種) / self._resolve_label_map (混合模式且已知物種 → 該物種的轉換字典表)     標籤轉換（dataset_builder.py)

"""
import csv
import os
import shutil
from pathlib import Path

import cv2 #將記憶體中的影像矩陣（NumPy 陣列）儲存到硬碟中，變成一個實體的圖片檔案
import numpy as np
import torch #清除 PyTorch 在 GPU 顯存中未被使用但被快取佔用的記憶體（釋放顯存快取）。
from ultralytics import YOLO #實例化與載入模型權重
from ultralytics.utils.metrics import smooth #對整條 F1 曲線進行平滑處理（過濾掉高頻雜訊）
#用 import 的（都是獨立的函式或常數，不屬於任何物件）
from .config import FIXED_CONF, RUNS_DIR   # CSV 報表中 Precision/Recall/F1@固定門檻 所用的信心門檻、Ultralytics 輸出根目錄
from .labels import convert_label_lines #通用:轉換第一欄位後的YOLO 標籤檔
#回傳影像資料夾內排序後的影像檔名清單、讀取影像並轉灰階 3 通道、寫入 YOLO 資料集設定檔
from .staging import list_image_names, read_gray_3ch, write_data_yaml


# ============================================================================
# 預測結果輸出（只處理 YOLO 預測結果，所以留在本檔）:模組層級的共用工具函式
# ============================================================================
def print_detections(model, result): #輸入:YOLO 模型物件、單張影像的預測結果物件(由 Ultralytics YOLO 套件提供)
    """印出單張影像的每個偵測框（類別、信心值、座標）"""
    for box in result.boxes: #逐個處理畫面中被抓出來的每一個邊界框
        x1, y1, x2, y2 = box.xyxy[0].tolist() #邊界框座標(張量大小[1,4]:二維矩陣)
        class_id = int(box.cls[0].item()) #類別代號。[1]（一維張量)
        conf_val = float(box.conf[0].item()) #預測信心值[1]（一維張量)
        class_name = model.names[class_id] #對應類別標籤的字典
        print(f"  -> Detected: {class_name} ({conf_val:.2f}) at [{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}]") #.2:保留小數點後 2 位


class EvaluatorMixin:

    # ---- 評估：測試集
    def _prf_at_conf(self, metrics, thr): #輸入:驗證結果物件(model.val)、信心門檻值
        """取固定信心門檻下，每個測試集有標註的類別的 P / R / F1"""
        px = np.asarray(metrics.box.px)                     # YOLO 預設產生的 1000 個信心值取樣點（介於 0 到 1 之間）。numpy.ndarray（一維 NumPy 陣列）(1000,)
        idx = int(np.argmin(np.abs(px - thr))) #尋找最小值的位置索引(距離信心門檻最近的點)
        p = np.asarray(metrics.box.p_curve)[:, idx] #每個類別在各個信心點下的 Precision（精確率）曲線
        r = np.asarray(metrics.box.r_curve)[:, idx] #每個類別在各個信心點下的 Recall（召回率）曲線。
        f1 = 2 * p * r / (p + r + 1e-16)
        return p, r, f1
#輸入:權重檔案、測試影像來源、預測信心門檻、輸出名稱後綴、測試集所屬物種
    def detect_bacteria(self, model_name='yolo11m.pt', image_source=None, confidence=0.5,
                        tag=None, species=None):
        """
        測試集綜合評估與預測：
        1. 自動將測試圖片轉灰階(3通道)
        2. 自動對齊 .txt 標籤（依測試集所屬物種轉成最終類別 ID）並呼叫 model.val()
        3. 自動匯出數據為 CSV 檔案
        4. 自動繪製並儲存帶框圖片
        tag    : 輸出名稱後綴。tag='cocci' → test_evaluation_all_cocci / predict_all_cocci
        species: 僅混合模式使用，測試集標籤所屬物種 ('cocci'/'bacilli')。
                 None 時由資料夾名稱推測；已合併的 test_image_all 不需指定
        """
        run_id = f'{self.name}_{tag}' if tag else self.name    # (模式+菌種名稱)所有輸出名稱都用它
        if image_source is None:
            image_source = f'./test_image_{self.name}'

        collected = self._collect_test_images(image_source) #收集測試影像。回傳 (測試資料夾路徑, 影像檔名清單)
        if collected is None:
            return None
        test_path, image_list = collected #序列解包
        label_map = self._resolve_eval_label_map(test_path, species)  #回傳該物種的轉換字典表

        # 1. 預處理：影像轉灰階、標籤轉最終 ID，放進暫存資料夾
        eval_temp_dir = Path(f'./temp_test_eval_{run_id}') #暫存評估資料夾path物件
        #回傳 (暫存影像資料夾, 處理後影像的絕對路徑清單)
        eval_img_dir, processed_img_paths = self._prepare_eval_dataset(test_path, image_list, label_map, eval_temp_dir) #建立暫存資料夾：影像轉灰階、標籤轉最終 ID。
        #回傳:灰階後的圖片絕對路徑清單文字檔、測試集yaml設定檔
        test_txt_path, test_yaml_path = self._write_eval_files(run_id, processed_img_paths, eval_temp_dir)

        # 2. 評估（mAP / Precision / Recall）
        model = YOLO(model_name)
        print(f"\n==================== 開始評估測試集指標 (mAP / Precision / Recall) ====================")
        eval_run_name = f'test_evaluation_{run_id}'
        metrics = model.val( #進行模型推論與真實答案比對
            data=test_yaml_path,
            imgsz=self.img_size,
            name=eval_run_name, #指定輸出結果的資料夾名稱。
            split='val', #YOLO 執行時只會去讀取 YAML 裡的 val 欄位
            exist_ok=True          # 同名資料夾直接覆蓋，避免 Ultralytics 自動改名成 xxx2，導致圖表與 CSV 分家
        )
        overall = self._print_overall_report(run_id, metrics) #印出整體指標並回傳 (P, R, mAP50, mAP50-95)

        # 3. CSV 報表
        csv_save_path = Path(f'{RUNS_DIR}/{eval_run_name}') / f'test_evaluation_{run_id}.csv'
        self._write_metrics_csv(metrics, csv_save_path, overall) #寫 CSV：整體一列 + 各類別一列。
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

    @staticmethod #不能使用同類別的方法或屬性
    def _collect_test_images(image_source): #輸入:測試影像來源
        """收集測試影像。回傳 (測試資料夾路徑, 影像檔名清單)"""
        test_path = Path(image_source)
        if not test_path.exists():
            print(f"錯誤：找不到測試集目錄或檔案: {image_source}")
            return None
        if test_path.is_dir(): #路徑屬性是否為資料夾
            image_list = list_image_names(test_path) #輸入:影像資料夾路徑 回傳排序後檔名字串清單
        else:
            image_list = [test_path.name] #檔名含副檔名
            test_path = test_path.parent #上一層路徑（父目錄）
        if not image_list:
            print("未找到有效的測試影像。")
            return None
        return test_path, image_list #如果一個函數想要回傳多個值Python 會自動把這些值組合成一個「元組」回傳。

    def _resolve_eval_label_map(self, test_path, species): #輸入:測試集路徑、測試集菌種
        """決定這個測試集的標籤轉換表"""
        if self.is_mixed and species is None:
            species = self._infer_species(test_path.name) #由資料夾名稱推測物種 輸入:資料夾的名稱
        if species is not None and species not in self.SPECIES_RAW_IDS:
            raise ValueError(f"不支援的 species: {species}")
        label_map = self._resolve_label_map(species) #混合模式且已知物種 → 該物種的轉換字典表
        if self.is_mixed:
            print(f"[標籤轉換] 測試集物種: {species if species else '未指定（視為已是最終 ID 或原始 10/11/20/21）'}")
        return label_map #回傳該物種的轉換字典表
#輸入:測試集資料夾PATH物件、要處理的影像檔名清單(包含副檔名)、標籤 ID轉換字典、暫存評估資料夾path物件
    def _prepare_eval_dataset(self, test_path, image_list, label_map, eval_temp_dir):
        """建立暫存資料夾：影像轉灰階、標籤轉最終 ID。"""
        shutil.rmtree(eval_temp_dir, ignore_errors=True)   # 清掉上次中斷留下的殘留
        eval_img_dir = eval_temp_dir / 'images'
        eval_label_dir = eval_temp_dir / 'labels'
        eval_img_dir.mkdir(parents=True, exist_ok=True)
        eval_label_dir.mkdir(parents=True, exist_ok=True)

        processed_img_paths = [] #已完成預處理的測試圖片絕對路徑清單
        all_out_ids = set() #測試集中被轉換後的最終類別 ID集合
        print(f"\n==================== 正在預處理測試集影像 (自動轉灰階) ====================")
        for img_name in image_list:
            gray = read_gray_3ch(test_path / img_name) #輸入:單張影像的檔案路徑path物件 回傳:複製成 3 通道的灰階影像矩陣
            if gray is None:
                continue
            save_img_path = eval_img_dir / img_name
            cv2.imwrite(str(save_img_path), gray)
            processed_img_paths.append(str(save_img_path.resolve())) #將一個路徑物件轉換成該檔案在電腦硬碟中的完整絕對路徑物件

            label_path = test_path / (Path(img_name).stem + '.txt') #取出檔名（不包含副檔名）
            if label_path.exists():
                with open(label_path, 'r') as f:
                    lines = f.readlines()
            #回傳: 新標籤內容清單、輸出類別ID 集合
                new_lines, out_ids = convert_label_lines(lines, label_map) #輸入:標籤內容清單、類別ID對應表
                all_out_ids |= out_ids #聯集=all_out_ids = all_out_ids | out_ids
                with open(eval_label_dir / label_path.name, 'w') as f:
                    f.writelines(new_lines)

        # 轉換後 ID 必須落在 0 ~ num_classes-1，否則 YOLO 會悄悄忽略或報錯
        bad_ids = all_out_ids - set(range(self.num_classes)) #range():產生一連串的整數
        if bad_ids:
            print(f"警告：轉換後仍有超出範圍的類別 ID: {sorted(bad_ids)}，請檢查測試集標籤或 species 設定")
        return eval_img_dir, processed_img_paths #回傳 (暫存影像資料夾, 處理後影像的絕對路徑清單)
#輸入:(模式+菌種名稱)後綴、灰階後的圖片絕對路徑清單、暫存評估資料夾路徑
    def _write_eval_files(self, run_id, processed_img_paths, eval_temp_dir):
        """為 YOLO 模型的評估流程準備並寫入清單文字檔與「設定檔（YAML）」。"""
        test_txt_path = f'./test_eval_list_{run_id}.txt'
        with open(test_txt_path, 'w') as f:
            for p in processed_img_paths:
                f.write(f"{p}\n")
        test_yaml_path = f'test_eval_{run_id}.yaml'
        write_data_yaml(test_yaml_path, test_txt_path, test_txt_path, eval_temp_dir, #輸入:YAML檔案路徑、訓練/驗證/測試圖片清單絕對路徑文字檔、資料集根目錄、類別總數、類別名稱清單
                        self.num_classes, self.class_names, test=test_txt_path)
        return test_txt_path, test_yaml_path #回傳:灰階後的圖片絕對路徑清單文字檔、測試集yaml設定檔

    @staticmethod
    def _print_overall_report(run_id, metrics): #輸入:(模式+菌種名稱)後綴、驗證/評估結果物件
        """印出整體指標並回傳 (P, R, mAP50, mAP50-95)"""
        p = float(metrics.box.mp) # Mean Precision (所有類別的平均精確度)
        r = float(metrics.box.mr) # Mean Recall (所有類別的平均召回率)
        map50 = float(metrics.box.map50) # mAP@0.5 (全體在門檻0.5下的平均精度)
        map95 = float(metrics.box.map) # mAP@0.5:0.95
        print("\n" + "=" * 50)
        print(f"【獨立測試集評估報告 - {run_id}】")
        print(f" Precision  : {p:.4f}")
        print(f" Recall     : {r:.4f}")
        print(f" mAP50      : {map50:.4f}")
        print(f" mAP50-95   : {map95:.4f}")
        print("=" * 50)
        return p, r, map50, map95
#輸入:驗證/評估結果物件、CSV 檔案的儲存目標路徑、 (P, R, mAP50, mAP50-95)值組
    def _write_metrics_csv(self, metrics, csv_save_path, overall):
        """寫 CSV：整體一列 + 各類別一列。"""
        p, r, map50, map95 = overall #序列解包
        csv_save_path.parent.mkdir(parents=True, exist_ok=True)
        p_fx, r_fx, f1_fx = self._prf_at_conf(metrics, FIXED_CONF) #取固定信心門檻下，每個測試集有標註的類別的 P / R / F1。（1D Array）(nc,)

        # 峰值 F1 所在的門檻（與 F1_curve.png 圖例一致）
        f1_mean_curve = np.asarray(metrics.box.f1_curve).mean(0) #全體平均的 F1 曲線，對矩陣的第 0 維度（直向/跨類別）計算平均值
        #.argmax():找出這條平滑後的 F1 曲線中，數值最高的那一個點，它到底排在第幾個位置（索引值）
        peak_conf = float(np.asarray(metrics.box.px)[smooth(f1_mean_curve, 0.1).argmax()]) #0.1:Ultralytics 內建的盒狀濾波器比例
        f1_all = float(np.mean(metrics.box.f1))   # 所有類別的 F1 分數加起來，除以類別總數

        with open(csv_save_path, mode='w', newline='', encoding='utf-8-sig') as csv_file: #停用 Python 預設的換行字元自動轉換機制
            writer = csv.writer(csv_file) #初始化並建立一個 CSV 寫入器物件。
            writer.writerow(['Class', 'Precision', 'Recall', 'F1', 'mAP50', 'mAP50-95',
                             f'Precision@{FIXED_CONF}', f'Recall@{FIXED_CONF}', f'F1@{FIXED_CONF}',
                             'peak_conf']) #csv寫入器物件所擁有的方法，用來將一列資料實際寫入到檔案中。
            writer.writerow(['all (Overall)', f"{p:.4f}", f"{r:.4f}", f"{f1_all:.4f}",
                             f"{map50:.4f}", f"{map95:.4f}",
                             f"{p_fx.mean():.4f}", f"{r_fx.mean():.4f}", f"{f1_fx.mean():.4f}",
                             f"{peak_conf:.3f}"])
#(K,C)=(Index,Value)
            cls_pos = {int(c): k for k, c in enumerate(metrics.box.ap_class_index)} # 這次測試集中「實際有出現」的類別索引清單
            for i, class_name in enumerate(self.class_names):
                if i in cls_pos: #檢查「字典的鍵（Keys）」
                    k = cls_pos[i]
                    class_p, class_r = float(metrics.box.p[k]), float(metrics.box.r[k])
                    class_f1 = float(metrics.box.f1[k]) #在同一個F1峰值門檻下抓出各類別的 P、R、F1
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
#輸入:載入好權重的 YOLO 模型物件、暫存影像資料夾、預測信心門檻、(模式+菌種名稱)後綴
    def _predict_and_save(self, model, eval_img_dir, confidence, run_id): #整批大量預測
        """對暫存測試影像做預測，輸出劃有目標框的影像，並逐張印出偵測結果"""
        print(f"\n==================== 開始繪製與輸出劃框影像 ====================")
        results = model.predict( #推論與劃框
            source=str(eval_img_dir), #暫存測試影像資料夾的路徑字串
            imgsz=self.img_size,
            save=True, #是否存檔
            conf=confidence,
            project=RUNS_DIR,
            name=f'predict_{run_id}',
            exist_ok=True,
            batch=4,
            stream=True #開啟「串流生成器（Generator）」模式。模型不會一口氣把所有圖片的結果全部載入記憶體，而是「跑完一張，就吐出一個 result 物件」。
        )
        for result in results:
            print(f"\n影像: {Path(result.path).name}") #這張圖片在硬碟中的完整絕對路徑字串，絕對路徑切開，只留下最尾端的檔名+副檔名
            print_detections(model, result) #印出單張影像的每個偵測框（類別、信心值、座標）

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
        results = model.predict( #前向傳播:執行物件推論與自動劃框
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
        print_detections(model, result) #印出單張影像的每個偵測框（類別、信心值、座標）

        if save_result:
            print(f"[完成] 劃框結果已儲存至 {RUNS_DIR}/predict_single/")
        return results