# -*- coding: utf-8 -*-
"""
mixer：MixerMixin（三來源「固定總量、可調比例」資料集，每折嚴格維持 N）

這是 Mixin，必須由 DetectBacteria 繼承（排在 DatasetBuilderMixin 之前，才能覆蓋 get_kfold_splits）。

流程：
  1. build_source_pool()：把三個來源「完整」轉標籤並複製到 dataset_mix/pool/（只需做一次，所有比例組共用）
  2. set_mix()：設定這次實驗的比例
  3. get_kfold_splits()：
       a. 只對真實 DIBAS 影像做 K-Fold（驗證集永遠是真實影像）
       b. N = 所有折的「訓練折」中，各類別實例數的最小值（與比例無關，所有實驗組共用同一個 N）
       c. 每一折：DIBAS 從該折訓練折抽 N*r_dibas；MATLAB / Diffusion 從完整來源抽 N*r
          → 每折、每類別的訓練總量都 ≈ N
trainer.py 只透過 get_kfold_splits() 取得切分，因此不需要修改。
"""
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold

from .config import BASE_SEED, MERGED_TEST_DIR, MIX_ROOT, SOURCE_NAMES
from .labels import convert_label_lines, label_name_of, species_of
from .staging import list_image_names


def source_of(name):
    """cocci_dibas_001.jpg → 'dibas'；舊格式檔名回傳 None"""
    parts = name.split('_', 2)
    return parts[1] if len(parts) == 3 and parts[1] in SOURCE_NAMES else None


def greedy_select(items, targets, seed):
    """
    貪心抽圖。items: {檔名: Counter(類別ID→實例數)}；targets: {類別ID: 目標實例數}
    固定種子隨機排序後逐張檢查：加入後任何類別超標就跳過，直到所有類別達標。
    回傳 (被選中檔名清單, 實際各類別實例數 Counter)
    """
    rng = random.Random(seed)
    names = sorted(items)
    rng.shuffle(names)
    cur, chosen = Counter(), []
    for n in names:
        cnt = items[n]
        if cnt and all(cur[c] + k <= targets.get(c, 0) for c, k in cnt.items()):
            chosen.append(n)
            cur.update(cnt)
            if all(cur[c] >= t for c, t in targets.items()):
                break
    return chosen, cur


class MixerMixin:

    # ------------------------------------------------------------------ 1. 建立來源池
    def _scan_source(self, entries, src):
        """掃描一個來源（可含多個資料夾）。回傳 {輸出檔名: {'img','lines'}}"""
        pool = {}
        for e in entries:
            img_dir, lbl_dir = Path(e['images']), Path(e['labels'])
            if not img_dir.exists() or not lbl_dir.exists():
                raise FileNotFoundError(f"[{src}] 找不到資料夾: {img_dir} 或 {lbl_dir}")
            sp = e.get('species')
            if self.is_mixed and sp not in self.SPECIES_RAW_IDS:
                raise ValueError(f"[{src}] 混合模式必須指定 species ('cocci'/'bacilli')：{e}")
            if not self.is_mixed and sp is not None and sp != self.name:
                raise ValueError(f"[{src}] 目前模式是 '{self.name}'，但來源指定了 species='{sp}'，"
                                 f"請只在 SOURCES 放 {self.name} 的資料夾")
            label_map = self._resolve_label_map(sp if self.is_mixed else None)
            prefix = sp if self.is_mixed else 'bact'

            for name in list_image_names(img_dir):
                lp = lbl_dir / label_name_of(name)
                if not lp.exists():
                    continue
                lines, _ = convert_label_lines(lp.read_text().splitlines(), label_map)
                pool[f'{prefix}_{src}_{name}'] = {'img': img_dir / name, 'lines': lines}
        return pool

    def build_source_pool(self, sources, force=False):
        """把三個來源完整轉標籤後複製到 dataset_mix/pool/。已存在且 force=False 時略過。
        （來源路徑或內容改變時，請用 force=True 重建）"""
        pool_dir = Path(MIX_ROOT) / f'pool_{self.name}'   # 依模式分開（標籤 ID 與檔名前綴不同）
        self.custom_dataset_dir = pool_dir
        self._cnt_cache = {}
        if pool_dir.exists() and not force:
            print(f"來源池已存在，略過建立: {pool_dir}（要重建請 force=True）")
            return pool_dir
        shutil.rmtree(pool_dir, ignore_errors=True)
        (pool_dir / 'images').mkdir(parents=True)
        (pool_dir / 'labels').mkdir(parents=True)
        for s in SOURCE_NAMES:
            pool = self._scan_source(sources[s], s)
            print(f"[{s}] 共 {len(pool)} 張")
            for k, v in pool.items():
                shutil.copy(v['img'], pool_dir / 'images' / k)
                (pool_dir / 'labels' / label_name_of(k)).write_text(''.join(v['lines']))
        return pool_dir

    def _pool_counts(self, image_dir):
        """{影像檔名: Counter(類別ID→實例數)}，讀來源池的標籤檔（有快取）"""
        lbl_dir = Path(image_dir).parent / 'labels'
        cache = self.__dict__.setdefault('_cnt_cache', {})
        key = str(lbl_dir)
        if key not in cache:
            d = {}
            for n in list_image_names(image_dir):
                lp = lbl_dir / label_name_of(n)
                lines = lp.read_text().splitlines() if lp.exists() else []
                d[n] = Counter(int(ln.split()[0]) for ln in lines if ln.split())
            cache[key] = d
        return cache[key]

    # ------------------------------------------------------------------ 2. 設定比例
    def set_mix(self, ratios, tag, seed=BASE_SEED):
        if set(ratios) != set(SOURCE_NAMES):
            raise ValueError(f"ratios 的鍵必須是 {SOURCE_NAMES}")
        if abs(sum(ratios.values()) - 1.0) > 1e-6:
            raise ValueError(f"三個比例總和必須為 1，目前 = {sum(ratios.values())}")
        self.mix_ratios, self.mix_tag, self.mix_seed = dict(ratios), tag, seed

    # ------------------------------------------------------------------ 測試集合併
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
                 'source': Counter(), 'normalized_files': defaultdict(set)}
        for sp in species_list:
            self._merge_one_folder(Path(f'test_image_{sp}'), MERGED_TEST_DIR, sp, stats)
        self._check_label_problems(stats, strict=False)

    # ------------------------------------------------------------------ 3. K-Fold + 每折抽樣
    def get_kfold_splits(self, image_dir, n_splits=5, seed=BASE_SEED):
        """
        回傳格式與原版相同：[(訓練影像, 驗證影像, 訓練標籤, 驗證標籤), ...]
        未呼叫 set_mix() 時退回原版行為。
        """
        if getattr(self, 'mix_ratios', None) is None:
            return super().get_kfold_splits(image_dir, n_splits, seed)

        cnt = self._pool_counts(image_dir)
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

        # (b) N = 所有折的訓練折中，各類別實例數的最小值（只取決於 DIBAS 與切折，與比例無關）
        N = min(total(real[tr])[k] for tr, _ in folds for k in class_ids)
        if N == 0:
            raise ValueError("某折訓練集有類別實例數為 0，無法平衡")
        print(f"\n[基準] 每折、每類別訓練目標總量 N = {N}｜比例 {self.mix_ratios}")

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
                chosen, cur = greedy_select({n: cnt[n] for n in cand}, targets, self.mix_seed)
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
            json.dump({'N': N, 'ratios': self.mix_ratios, 'seed': self.mix_seed, 'folds': record},
                      f, ensure_ascii=False, indent=2)
        return out