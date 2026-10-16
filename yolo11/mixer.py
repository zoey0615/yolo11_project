# -*- coding: utf-8 -*-
"""
mixer：MixerMixin（三來源「固定總量、可調比例」資料集，每折嚴格維持 N）

這是 Mixin，必須由 DetectBacteria 繼承（排在 DatasetBuilderMixin 之前，才能覆蓋 get_kfold_splits）。

流程：
  1. build_source_pool()：把三個來源「完整」轉標籤並複製到 dataset_mix/pool/（只需做一次，所有比例組共用）（來源路徑或內容改變時，請用 force=True 重建）
  2. set_mix()：設定這次實驗的比例
  3. get_kfold_splits()：
       a. 只對真實 DIBAS 影像做 K-Fold（驗證集永遠是真實影像）
       b. N = 所有折的「訓練折」中，各類別實例數的最小值（與比例無關，所有實驗組共用同一個 N）
       c. 每一折：DIBAS 從該折訓練折抽 N*r_dibas；MATLAB / Diffusion 從完整來源抽 N*r
          → 每折、每類別的訓練總量都 ≈ N
trainer.py 只透過 get_kfold_splits() 取得切分，因此不需要修改。
"""
import json
import random #引入 Python 內建的隨機數產生模組
import shutil
from collections import Counter, defaultdict #計數器字典物件、帶預設VALUE的字典
from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
  
# K-Fold 切分種子、混合測試集相對路徑path物件、混合資料集輸出根目錄path物件、三種訓練資料來源值組
from .config import BASE_SEED, MERGED_TEST_DIR, MIX_ROOT, SOURCE_NAMES
#轉換第一欄位後的YOLO 標籤檔、利用影像檔名轉換標籤檔名、利用影像檔名前綴取得物種字串
from .labels import convert_label_lines, label_name_of, species_of
from .staging import list_image_names #影像資料夾內排序後的影像檔名清單(含副檔名)
from scipy.optimize import milp, LinearConstraint, Bounds

def source_of(name): #利用影像檔名(含副檔名)前綴取得來源字串
    """cocci_dibas_001.jpg → 'dibas'；舊格式檔名回傳 None"""
    parts = name.split('_', 2) #2:最大分割次數
    return parts[1] if len(parts) == 3 and parts[1] in SOURCE_NAMES else None


def greedy_select(items, targets, seed, time_limit=60):
    """
    兩階段整數規劃抽圖（函式名稱與回傳格式與舊版相同）。
    items:   {檔名: Counter({類別ID: 實例數})}
    targets: {類別ID: 目標實例數}

    階段 1：最小化「各類別實際數與目標的最大相對偏差 t」
            偏差 = |實際 - 目標| / 目標（高於或低於都算）
    階段 2：固定 最大偏差 ≤ t*，再最小化「各類別偏差的總和」，
            避免階段 1 只管最差的類別、其他類別被隨意放著。
    每張圖最多選一次（0/1 變數）；seed 只用來在多組最佳解之間打破平手（可重現）。
    """
    rng = random.Random(seed)
    names = sorted(items)
    rng.shuffle(names)

    classes = sorted(c for c, t in targets.items() if t > 0)
    if not classes:
        return [], Counter()
    others = sorted({c for nm in names for c in items[nm]} - set(classes))
    allc = classes + others
    idx = {c: j for j, c in enumerate(allc)}
    n, C = len(names), len(classes)

    M = np.zeros((n, len(allc)))                 # 圖 × 類別 的實例數
    for i, nm in enumerate(names):
        for c, k in items[nm].items():
            M[i, idx[c]] = k

    # 含有「沒有目標的類別」的圖不選；空標籤圖也不選（與舊版行為一致）
    ok_img = (M[:, C:].sum(axis=1) == 0) & (M[:, :C].sum(axis=1) > 0)
    T = np.array([targets[c] for c in classes], dtype=float)
    Mc = M[:, :C]
    eps = np.array([rng.random() for _ in range(n)]) * 1e-7   # 極小成本，只用來打破平手
    ub_x = ok_img.astype(float)
    int_x = np.ones(n)

    # ---------- 階段 1：最小化最大相對偏差 t ----------
    # 變數 = [x_1..x_n, t]
    A_over1 = np.hstack([Mc.T, -T[:, None]])      # Σx·M − T·t ≤ T
    A_under1 = np.hstack([-Mc.T, -T[:, None]])    # −Σx·M − T·t ≤ −T
    res1 = milp(
        np.concatenate([eps, [1.0]]),
        constraints=[LinearConstraint(A_over1, -np.inf, T),
                     LinearConstraint(A_under1, -np.inf, -T)],
        integrality=np.concatenate([int_x, [0]]),
        bounds=Bounds(np.zeros(n + 1), np.concatenate([ub_x, [np.inf]])),
        options={'time_limit': time_limit},
    )
    if res1.x is None:
        raise RuntimeError(f"MILP 階段1 無解：{res1.message}")
    t_star = float(res1.x[n])
    print(f"    階段1 最大相對偏差 t = {t_star:.3%}"
          f"（status={res1.status}，0=已證明最佳，1=超時）")
    res = res1

    # ---------- 階段 2：固定 t ≤ t*，最小化各類別偏差總和 ----------
    # 變數 = [x_1..x_n, u_1..u_C]，u_c = 類別 c 的相對偏差
    t_cap = t_star * (1 + 1e-6) + 1e-9            # 留一點浮點誤差空間
    A_over2 = np.hstack([Mc.T, -np.diag(T)])      # Σx·M_c − T_c·u_c ≤ T_c
    A_under2 = np.hstack([-Mc.T, -np.diag(T)])    # −Σx·M_c − T_c·u_c ≤ −T_c
    res2 = milp(
        np.concatenate([eps, np.ones(C)]),
        constraints=[LinearConstraint(A_over2, -np.inf, T),
                     LinearConstraint(A_under2, -np.inf, -T)],
        integrality=np.concatenate([int_x, np.zeros(C)]),
        bounds=Bounds(np.zeros(n + C), np.concatenate([ub_x, np.full(C, t_cap)])),
        options={'time_limit': time_limit},
    )
    if res2.x is not None:
        res = res2
        print(f"    階段2 各類別偏差 {[f'{u:.2%}' for u in res2.x[n:]]}"
              f"（status={res2.status}）")
    else:
        print("    階段2 無解，沿用階段1結果")

    pick = res.x[:n] > 0.5
    chosen = [nm for nm, s in zip(names, pick) if s]
    cur = M[pick].sum(axis=0)
    return chosen, Counter({c: int(cur[idx[c]]) for c in allc if cur[idx[c]] > 0})

class MixerMixin:

    # --------------- 1. 建立來源池
    def _scan_source(self, entries, src): #輸入:資料來源字典清單、資料來源字串
        """掃描來源進行標籤與檔名格式轉換。"""
        pool = {} #要轉換檔名資料來源底下所有「圖片與標籤資料」的空字典
        for e in entries:
            img_dir, lbl_dir = Path(e['images']), Path(e['labels'])
            if not img_dir.exists() or not lbl_dir.exists():
                raise FileNotFoundError(f"[{src}] 找不到資料夾: {img_dir} 或 {lbl_dir}")
            sp = e.get('species') #使用安全取值，如果字典裡面找不到這個鍵，它不會報錯，而是會很安全地回傳 None
            if self.is_mixed and sp not in self.SPECIES_RAW_IDS:
                raise ValueError(f"[{src}] 混合模式必須指定 species ('cocci'/'bacilli')：{e}")
            if not self.is_mixed and sp is not None and sp != self.name:
                raise ValueError(f"[{src}] 目前模式是 '{self.name}'，但來源指定了 species='{sp}'，"
                                 f"請只在 SOURCES 放 {self.name} 的資料夾")
            label_map = self._resolve_label_map(sp if self.is_mixed else None) #dataset_builder.py:混合模式且已知物種 → 該物種的轉換表；其餘 → class_mapping
            prefix = sp if self.is_mixed else 'bact'

            for name in list_image_names(img_dir): #影像資料夾內排序後的影像檔名清單(含副檔名)
                lp = lbl_dir / label_name_of(name) #輸入:影像檔名(含副檔名)  回傳標籤檔名
                if not lp.exists():
                    continue
                #回傳:新標籤內容清單(每個字串元素末尾補回 \n)、輸出類別ID 集合
                #.read_text() 是 Path 專屬的內建方法，它的功能是：直接打開那個檔案，把它裡面的文字內容全部讀出來，變成一串大型的字串
                #.splitlines():按照換行符號（\n）把這一大串文字切開，變成一個「清單（List）」。清單裡面的每一個元素，剛好就是文字檔裡的「每一列」。
                lines, _ = convert_label_lines(lp.read_text().splitlines(), label_map) #輸入:標籤內容清單、類別ID對應表
                pool[f'{prefix}_{src}_{name}'] = {'img': img_dir / name, 'lines': lines}
        return pool #回傳 {輸出檔名: {'img','lines'}}

    def build_source_pool(self, sources, force=False): #輸入:資料來源設定字典
        """把三個來源完整轉標籤後複製到 dataset_mix/pool_模式/。
        """
        pool_dir = Path(MIX_ROOT) / f'pool_{self.name}'   # 依模式分開（標籤 ID 與檔名前綴不同）
        self.custom_dataset_dir = pool_dir #相對路徑
        self._cnt_cache = {} #類別標籤計數快取字典
        if pool_dir.exists() and not force:
            print(f"來源池已存在，略過建立: {pool_dir}（要重建請 force=True）")
            return pool_dir
        shutil.rmtree(pool_dir, ignore_errors=True)
        (pool_dir / 'images').mkdir(parents=True)
        (pool_dir / 'labels').mkdir(parents=True)
        for s in SOURCE_NAMES:
            #回傳 {輸出檔名: {'img','lines'}} 輸入:{來源: [{'images','labels','species'},...],...} 來源底下的資料夾設定清單(裡有字典)
            pool = self._scan_source(sources[s], s) #掃描來源進行標籤與檔名格式轉換
            print(f"[{s}] 共 {len(pool)} 張") #key的數量
            for k, v in pool.items(): #同時把字典裡面的鍵和值，變成一個個的Tuple list物件（View Object）
                shutil.copy(v['img'], pool_dir / 'images' / k) #shutil.copy(來源路徑, 目標路徑)
                #path.write_text():將字串寫入檔案
                (pool_dir / 'labels' / label_name_of(k)).write_text(''.join(v['lines'])) #分隔符號.join(字串清單):把多個字串組合成一個大字串
        return pool_dir #混合來源資料集根目錄路徑

    def _pool_counts(self, image_dir): #輸入:圖片資料夾的路徑
        """讀來源池的標籤檔（有快取）"""
        lbl_dir = Path(image_dir).parent / 'labels'
        #DetectBacteria物件屬性字典:字典的 Key 就是你設定的變數名稱，而 Value 就是指派的值
        #字典.setdefault(鍵, 預設值):幫字典檢查某個 Key（鍵）是否存在，如果沒有就幫它補上預設值；如果已經有了，就直接把現有的值拿出來用。
        cache = self.__dict__.setdefault('_cnt_cache', {}) #巢狀字典
        key = str(lbl_dir)
        if key not in cache:
            d = {} #{影像檔名: Counter(類別ID→實例數)} 暫時收集該資料夾底下所有圖片統計結果
            for n in list_image_names(image_dir): #影像資料夾內排序後的影像檔名清單(含副檔名)
                lp = lbl_dir / label_name_of(n) #輸入:影像檔名(含副檔名)  回傳標籤檔名
                #字串.splitlines():自動尋找字串中的"換行"符號，把這大串文字切開，變成一個 List，清單裡的每一個元素就是檔案裡的一列文字
                lines = lp.read_text().splitlines() if lp.exists() else []
                d[n] = Counter(int(ln.split()[0]) for ln in lines if ln.split()) #把一整列文字按照空白字元切開，變成一個個獨立的單字字串元素清單
            cache[key] = d
        return cache[key] #回傳:{影像檔名: Counter(類別ID→實例數)}

    # ------- 2. 設定比例
    def set_mix(self, ratios, tag, seed=BASE_SEED, N=None):
        # N：每折、每類別的訓練目標量。None → 自動取各折最小值（舊行為）
        if set(ratios) != set(SOURCE_NAMES):
            raise ValueError(f"ratios 的鍵必須是 {SOURCE_NAMES}")
        if abs(sum(ratios.values()) - 1.0) > 1e-6:
            raise ValueError(f"三個比例總和必須為 1，目前 = {sum(ratios.values())}")
        self.mix_ratios, self.mix_tag, self.mix_seed = dict(ratios), tag, seed
        self.mix_N = N
    #--- 測試集合併
    def build_merged_test(self, species_list=('cocci', 'bacilli'), force=False):
        """只合併 test_image_{sp}/ → test_image_all/"""
        if not self.is_mixed:
            raise RuntimeError("請以 DetectBacteria('all') 呼叫")
        if MERGED_TEST_DIR.exists() and not force:
            print("test_image_all 已存在，略過（要重建請 force=True）")
            return
        shutil.rmtree(MERGED_TEST_DIR, ignore_errors=True)
        MERGED_TEST_DIR.mkdir(parents=True)
        stats = {'converted': 0, 'bad_format': [], 'unexpected': [],
                 'source': Counter(), 'normalized_files': defaultdict(set)} #宣告一個預設值是 set 的 defaultdict
        for sp in species_list:
            #把單一來源資料夾的影像（複製）與標籤（轉換）加上 '{species}_' 前綴後放進 dst
            self._merge_one_folder(Path(f'test_image_{sp}'), MERGED_TEST_DIR, sp, stats) #dataset_builder.py
        #異常標籤處理：strict 時中止並清除半成品
        self._check_label_problems(stats, strict=False) #dataset_builder.py
        
    def find_best_N(self, image_dir, n_splits=5, seed=BASE_SEED,
                    max_dev=0.05, step=50, min_N=500):
        """
        自動找出「5 折都能把各類別最大相對偏差壓到 max_dev 以內」的最大 N。
        從各折最小可用量往下掃，每次減 step；找不到就回傳 None。
        只用 DIBAS 與切折，與比例無關，所以所有實驗組共用同一個結果。
        """
        cnt = self._pool_counts(image_dir)
        real = np.array([n for n in list_image_names(image_dir) if source_of(n) == 'dibas'])
        if self.is_mixed:
            sp = np.array([species_of(n) for n in real])
            folds = list(StratifiedKFold(n_splits=n_splits, shuffle=True,
                                         random_state=seed).split(real, sp))
        else:
            folds = list(KFold(n_splits=n_splits, shuffle=True, random_state=seed).split(real))
        class_ids = sorted(set(self.class_mapping.values()))

        def total(ns):
            c = Counter()
            for n in ns:
                c.update(cnt[n])
            return c

        N_auto = min(total(real[tr])[k] for tr, _ in folds for k in class_ids)
        for N in range(N_auto, min_N - 1, -step):
            worst = 0.0
            for tr, _ in folds:
                items = {n: cnt[n] for n in real[tr]}
                _, cur = greedy_select(items, {k: N for k in class_ids}, self.mix_seed)
                worst = max(worst, max(abs(cur[k] - N) / N for k in class_ids))
                if worst > max_dev:
                    break                      # 這個 N 已有一折不合格，換下一個 N
            print(f"[掃描 N] N={N} 五折最差偏差 {worst:.2%}")
            if worst <= max_dev:
                return N
        return None
    # ------- 3. K-Fold + 每折抽樣
    def get_kfold_splits(self, image_dir, n_splits=5, seed=BASE_SEED):
        """
        回傳格式與原版相同：[(訓練影像, 驗證影像, 訓練標籤, 驗證標籤), ...]
        未呼叫 set_mix() 時退回原版行為。
        """
        if getattr(self, 'mix_ratios', None) is None:
            return super().get_kfold_splits(image_dir, n_splits, seed)
#回傳:{影像檔名: Counter(類別ID→實例數)}
        cnt = self._pool_counts(image_dir) #讀來源池的標籤檔（有快取）
        names = list_image_names(image_dir)
        real = np.array([n for n in names if source_of(n) == 'dibas'])
        synth = {s: [n for n in names if source_of(n) == s] for s in SOURCE_NAMES if s != 'dibas'}
        if len(real) == 0:
            raise ValueError("來源池內沒有 DIBAS 影像")

        # (a) 只對真實影像切折
        if self.is_mixed:
            species = np.array([species_of(n) for n in real])
            folds = list(StratifiedKFold(n_splits=n_splits, shuffle=True,
                                         random_state=seed).split(real, species))
        else:
            folds = list(KFold(n_splits=n_splits, shuffle=True, random_state=seed).split(real))

        class_ids = sorted(set(self.class_mapping.values()))

        def total(ns):
            c = Counter()
            for n in ns:
                c.update(cnt[n])
            return c

        N_auto = min(total(real[tr])[k] for tr, _ in folds for k in class_ids)
        if N_auto == 0:
            raise ValueError("某折訓練集有類別實例數為 0，無法平衡")
        N = getattr(self, 'mix_N', None) or N_auto      # 有指定就用指定值
        if N > N_auto:
            raise ValueError(f"指定 N={N} 大於各折可用的最小值 {N_auto}")
        print(f"\n[基準] 每折、每類別訓練目標總量 N = {N}"
              f"（{'手動指定' if N != N_auto else '自動'}；各折最小可用量 {N_auto}）"
              f"｜比例 {self.mix_ratios}")

        # (c) 每折抽樣
        out, record = [], {}
        for i, (tr, va) in enumerate(folds):
            train_names, fold_total = [], Counter()
            for s in SOURCE_NAMES:
                r = self.mix_ratios[s]
                if r <= 0:
                    continue
                cand = real[tr].tolist() if s == 'dibas' else synth[s]
                targets = {k: round(N * r) for k in class_ids}
                sel_seed = self.mix_seed if s == 'dibas' else self.mix_seed + i
                chosen, cur = greedy_select({n: cnt[n] for n in cand}, targets, sel_seed) #回傳 (被選中檔名清單, 實際各類別實例數 Counter)
                train_names += chosen
                fold_total.update(cur)
                short = [k for k in class_ids if cur[k] < targets[k] * 0.95]
                print(f"  第{i + 1}折 [{s}] {len(chosen)} 張｜各類別 {[cur[k] for k in class_ids]}"
                      f"｜目標 {targets[class_ids[0]]}" + (f"  ⚠ 類別{short}不足" if short else ""))
            val_names = real[va].tolist()
            print(f"  第{i + 1}折 合計各類別 {[fold_total[k] for k in class_ids]}｜驗證(真實) {len(val_names)} 張")
            out.append((train_names, val_names,
                        [label_name_of(n) for n in train_names],
                        [label_name_of(n) for n in val_names]))
            record[f'fold{i + 1}'] = {'train': train_names, 'val': val_names}

        Path(MIX_ROOT).mkdir(parents=True, exist_ok=True)
        with open(Path(MIX_ROOT) / f'splits_{self.mix_tag}.json', 'w', encoding='utf-8') as f:
            json.dump({'N': N, 'N_auto': N_auto, 'ratios': self.mix_ratios,
                       'seed': self.mix_seed, 'folds': record},
                      f, ensure_ascii=False, indent=2)
        return out