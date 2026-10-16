# -*- coding: utf-8 -*-
"""
dataset_builder：DatasetBuilderMixin（標籤轉換、混合資料集建立、K-Fold 切分）

這是 Mixin，不能單獨使用，必須由 DetectBacteria 繼承。
方法內的 self.xxx 由宿主類別（detector.DetectBacteria）提供：

  self.SPECIES_RAW_IDS   各物種原始標籤 ID 對照表(字典)
  self.class_mapping     原始標籤 ID → 最終類別 ID(字典)
  self.class_names       最終類別名稱(字串列表)
  self.is_mixed          是否為混合模式
"""
import shutil
from collections import Counter, defaultdict #計數器、預設字典
from pathlib import Path
import numpy as np #批次索引
from sklearn.model_selection import KFold, StratifiedKFold #差別:避免該折訓練或驗證時失真
# # K-Fold 切分種子、 混合模式：訓練+驗證資料相對路徑 、混合模式：測試集相對路徑(path)
from .config import BASE_SEED, MERGED_DATASET_DIR, MERGED_TEST_DIR
 #  轉換第一欄位後的YOLO 標籤檔、利用影像檔名轉換標籤檔名、利用影像檔名前綴取得物種
from .labels import convert_label_lines, label_name_of, species_of
from .staging import is_image_name, list_image_names #副檔名是否為支援的影像格式、回傳影像資料夾內排序後的影像檔名清單

#_內部私有方法、靜態方法
class DatasetBuilderMixin:

    def _species_label_map(self, species): #輸入:細菌的物種名稱字串
        """回傳該菌種的「原始 ID → 最終 ID」轉換字典"""
        single_raw, colony_raw = self.SPECIES_RAW_IDS[species] #取出對應元組10 11/ 20 21
        single_out = self.class_mapping[single_raw] #0/2
        colony_out = self.class_mapping[colony_raw] #1/3
        return {single_raw: single_out, colony_raw: colony_out,
                0: single_out, 1: colony_out}

    def _infer_species(self, folder_name): #輸入:資料夾的名稱
        """由資料夾名稱推測物種（test_image_cocci → 'cocci'）"""
        hits = [sp for sp in self.SPECIES_RAW_IDS if sp in folder_name.lower()] #用 for 迴圈去走訪一個字典時，它預設抓取的是字典的「鍵（Keys）」。
        return hits[0] if len(hits) == 1 else None

    def _resolve_label_map(self, species): #輸入:細菌的物種名稱字串
        """混合模式且已知物種 → 該物種的轉換表；其餘 → class_mapping"""
        if self.is_mixed and species is not None:
            return self._species_label_map(species) #該菌種的「原始 ID → 最終 ID」轉換字典
        return self.class_mapping

    def _convert_label_lines(self, lines, label_map): #輸入:標籤內容清單、類別ID對應表
        """轉換標籤行的類別 ID"""
        return convert_label_lines(lines, label_map) #輸入:標籤內容清單、類別ID對應表 回傳:新標籤內容清單、輸出類別ID 集合
#輸入:原始 /轉換後YOLO 標籤檔路徑、菌種字串、計數器
    def _convert_label_file(self, src, dst, species, stats):
        """讀取單一 YOLO 標籤檔，依該物種的轉換表轉成最終類別 ID 後寫出"""
        label_map = self._species_label_map(species) #單一菌種id字典
        raw_ids = set(self.SPECIES_RAW_IDS[species])   # 該物種的「原始標籤 ID 集合{}
        new_lines = [] #暫存類別 ID 的 YOLO 標籤資料清單
        #編號與內容
        for ln, line in enumerate(src.read_text().splitlines(), 1): #把大字串按照「換行符號（\n）」切開，存入字串清單，並幫清單編號
            parts = line.split() #依據任何空白字元切割欄位，存入字串清單
            if not parts: #空白行
                continue
            try:
                old_id = int(parts[0])
            except ValueError:
                stats['bad_format'].append(f'{src}:{ln}') #檔案路徑:列數
                continue
            if old_id not in label_map:
                stats['unexpected'].append(f'{src}:{ln} (id={old_id}, 物種={species})')
                continue
            # 記錄這行是「原始 ID」還是「已正規化的 0/1」，供最後報告
            kind = 'raw' if old_id in raw_ids else 'normalized'
            stats['source'][(species, kind)] += 1  #對應的值（Value）」遞增運算
            if kind == 'normalized':
                stats['normalized_files'][species].add(src.name) #自定義的鍵:取得純檔名，放進集合(不允許重複)裡
            parts[0] = str(label_map[old_id]) #標籤轉換
            new_lines.append(' '.join(parts) + '\n') #把字串清單資料(一列)寫回新的文字檔案裡
        dst.write_text(''.join(new_lines)) #多列組成新檔
        stats['converted'] += len(new_lines) #計算這一檔案的有效列數(記錄有效標注數)

    # ------ 資料準備：混合資料集
    def build_merged_dataset(self, species_list=('cocci', 'bacilli'), force=False, strict=True):
        """合併 dataset_{sp}/ 與 test_image_{sp}/ → dataset_all/、test_image_all/
        1. 混合模式:檔名加 '{sp}_' 前綴避免重複，並供 get_kfold_splits() 依前綴分層
        2. 標籤依「所屬物種」轉成最終 0~3；同時接受原始 ID (10/11, 20/21) 與已正規化的 0/1
        strict=True：發現異常標籤就中止並清除半成品；False：略過異常行並警告"""
        if not self.is_mixed:
            raise RuntimeError("請以 DetectBacteria('all') 呼叫此方法（需要 4 類別的 class_mapping）")
        for sp in species_list: #當集合去接字典時，它只會抓字典所有的『鍵（Key）』出來，忽略裡面的『值（Value）』」
            missing = set(self.SPECIES_RAW_IDS[sp]) - set(self.class_mapping) #集合差集{}
            if missing:
                raise ValueError(f"class_mapping 缺少 {sp} 的原始 ID: {sorted(missing)}")

        if MERGED_DATASET_DIR.exists() and MERGED_TEST_DIR.exists() and not force:
            print("dataset_all / test_image_all 已存在，略過合併（要重建請 force=True）")
            return #結束目前這個函式
        if force:
            self._remove_merged_dirs()

        dst_img, dst_lbl = MERGED_DATASET_DIR / 'images', MERGED_DATASET_DIR / 'labels'  #元組->多重指定與序列解包
        dst_test = MERGED_TEST_DIR
        for d in (dst_img, dst_lbl, dst_test):
            d.mkdir(parents=True, exist_ok=True) #允許建立父資料夾
        jobs = [] #工作排程清單:處理 cocci 的圖片、標籤、測試集，再加上 bacilli 的圖片、標籤、測試集元組
        for sp in species_list:
            jobs += [(f'dataset_{sp}/images', dst_img, sp),
                     (f'dataset_{sp}/labels', dst_lbl, sp),
                     (f'test_image_{sp}',     dst_test, sp)]

        stats = {'converted': 0, 'bad_format': [], 'unexpected': [],
                 'source': Counter(), 'normalized_files': defaultdict(set)} #集和內的元素不重複
        for src, dst, sp in jobs: #遍歷清單內元組
            self._merge_one_folder(Path(src), dst, sp, stats) #轉換:前綴處理好圖片、標籤、測試集的檔名

        self._check_label_problems(stats, strict)
        self._report_normalized_labels(stats, species_list)
        counter = self._count_label_ids(dst_lbl, dst_test)

        bad_out = set(counter) - set(self.class_mapping.values()) #對一個字典執行 set() 時，它抓取的是「鍵（Keys）」而不是「值（Values）」
        if bad_out:
            raise RuntimeError(f"輸出標籤出現非預期 ID: {sorted(bad_out)}")
        print(f"\n[完成] 共轉換 {stats['converted']} 個標註。各類別實例數（訓練+測試）：")
        for cid, name in enumerate(self.class_names): #列舉:套用在一個列表上時，它會一邊把裡面的元素拿出來，一邊自動幫它們編上流水號（從 0 開始）。
            print(f"  {cid} {name}: {counter.get(cid, 0)}") #字串格式化:字典物件.get(要找的 Key, 找不到時的預設值)

    @staticmethod #靜態方法:完全不需要建立物件，可以直接透過類別名稱來呼叫
    def _remove_merged_dirs():
        shutil.rmtree(MERGED_DATASET_DIR, ignore_errors=True) #遞迴刪除整個資料夾及其底下的所有檔案與子資料夾
        shutil.rmtree(MERGED_TEST_DIR, ignore_errors=True) #忽略刪除過程中的所有錯誤（容錯機制)

    def _merge_one_folder(self, src, dst, species, stats):
        """把單一來源資料夾的影像（複製）與標籤（轉換）加上 '{species}_' 前綴後放進 dst"""
        if not src.exists():
            print(f"警告：找不到 {src}，略過")
            return
        n = 0 #總共有多少個檔案被成功處理
        for f in src.iterdir():
            if not f.is_file():
                continue
            if f.suffix.lower() == '.txt': #副檔名後綴
                self._convert_label_file(f, dst / f'{species}_{f.name}', species, stats)
                n += 1
            elif is_image_name(f.name): #輸入影像檔名 回傳布林值
                shutil.copy(f, dst / f'{species}_{f.name}') #複製的同時指定新檔名
                n += 1
        print(f"{src} → {dst}: {n} 個檔案")

    def _check_label_problems(self, stats, strict):
        """異常標籤處理：strict 時中止並清除半成品"""
        problems = stats['bad_format'] + stats['unexpected'] #兩個 Python 列表（List）的串接（合併）
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
        """回報：哪些物種出現了「已正規化的 0/1」，請使用者抽查順序"""
        for sp in species_list:
            n_raw = stats['source'][(sp, 'raw')] #複合鍵（Compound Key） 例:('cocci', 'raw'): 2,
            n_norm = stats['source'][(sp, 'normalized')] #stats['source']是一個counter物件字典
            print(f"[{sp}] 原始 ID 標註: {n_raw}；已正規化 0/1 標註: {n_norm}")
            if n_norm: #value
                files = sorted(stats['normalized_files'][sp]) #例:'cocci': {'sample_001.txt', 'sample_045.txt'},
                print(f"  ↳ 含 0/1 的標籤檔 {len(files)} 個，例如: {files[:3]}"
                      f"（已依 {sp} 視為 Single/Colony 轉換，請確認順序無誤）")

    @staticmethod
    def _count_label_ids(*folders): #不定長度參數，打包成元組()
        """掃描輸出資料夾內所有標籤檔，統計各類別 ID 的實例數"""
        counter = Counter() #計數器
        for folder in folders:
            for lf in folder.glob('*.txt'): #pathlib 專用的「樣式搜尋方法」（Glob Pattern Matching）
                for line in lf.read_text().splitlines(): #.read_text()（讀取全文為大字串）;.splitlines()（按換行符號切開轉成清單）
                    parts = line.split() #字串按照『任何空白字元』（空格、Tab 換行等）切開
                    if parts:
                        counter[int(parts[0])] += 1
        return counter #字典物件

    #------- 資料準備：K-Fold
    def get_kfold_splits(self, image_dir, n_splits=5, seed=BASE_SEED):
        """取得 K-Fold 的所有摺數分割清單：[(影像訓練清單, 影像驗證清單, 標籤訓練清單, 標籤驗證清單), ...]
        單物種模式：KFold(shuffle=True)；混合模式：StratifiedKFold，依檔名前綴（物種）分層"""
        image_list = list_image_names(image_dir) #輸入:影像資料夾路徑 回傳排序後檔名字串清單
        image_array = np.array(image_list) #普通 Python 列表不接受用另一個「列表」直接當作索引去抓多個值
        label_array = np.array([label_name_of(n) for n in image_list]) #輸入:影像檔名 回傳標籤檔名

        if self.is_mixed:
            species = np.array([species_of(n) for n in image_list]) #輸入:影像檔名 回傳菌種字串
            unknown = set(species) - set(self.SPECIES_RAW_IDS) #抓取字典的KEY
            if unknown:
                raise ValueError(f"檔名缺少物種前綴 (cocci_ / bacilli_): {sorted(unknown)[:5]}")
        
            kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed) #每一折類別比例都要與整體相同
            split_iter = kf.split(image_array, species)   # 依物種分層， 輸出:(訓練集索引陣列, 驗證集索引陣列)元組清單
        else:
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
            split_iter = kf.split(image_array)
        return [(image_array[tr].tolist(), image_array[va].tolist(),
                 label_array[tr].tolist(), label_array[va].tolist())
                for tr, va in split_iter]

    def save_txt(self, filename, data_list, base_dir): #輸入:目標文字檔路徑、檔名清單、根目錄路徑
        """將訓練與驗證資料檔名絕對路徑存入文字檔（保留以相容舊呼叫)"""
        with open(filename, 'w') as f:
            for item in data_list:
                f.write(str(Path(base_dir) / item) + '\n')