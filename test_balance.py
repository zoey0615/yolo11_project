# -*- coding: utf-8 -*-
"""
test_balance.py：驗證「交叉驗證類別平衡算法」(mixer.get_kfold_splits + greedy_select) 是否正確

放在 yolo11/ 套件外面（與 main.py 同層），在專案根目錄執行：
    python test_balance.py

【彈性設計】目前只有 DIBAS 也能跑：
  - 自動偵測來源池裡實際有哪些來源；比例組需要的來源不存在 → 該組標示 SKIP（不算失敗）
  - USE_SIMULATED_SYNTH = True → 用 DIBAS 的隨機子集「模擬」MATLAB / Diffusion，
    只用來驗證混合邏輯（比例、總量、每折子集不同…）。模擬結果不代表真實合成資料的結果。
    模擬池建在 dataset_mix/pool_{MODE}_simtest/，不會動到真正的來源池。
  - 等真實合成資料到手、重建來源池後，把 USE_SIMULATED_SYNTH 改回 False 再跑一次即可

不會訓練，只讀來源池標籤檔並「獨立重新計數」（不使用 mixer 內部快取），避免自己驗自己。
檢查項目：
  1. 各折「訓練總量」每類別 ≈ N（最大相對偏差 ≤ MAX_DEV）
  2. 各來源（dibas/matlab/diffusion）每類別 ≈ N*比例
  3. 無資料洩漏：訓練 ∩ 驗證 = ∅；驗證集全為真實 DIBAS
  4. 訓練集內無重複圖
  5. 5 折驗證集互不重疊，聯集 = 全部真實 DIBAS
  6. 混合模式：每折驗證集兩個菌種比例接近（StratifiedKFold）
  7. 合成來源每折抽到的子集「不完全相同」（驗證 mix_seed + i 的修改）
  8. 可重現：同參數再跑一次結果相同（REPRO_CHECK=True 時）
"""
import os
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

from yolo11 import DetectBacteria
from yolo11.config import MIX_ROOT, SOURCE_NAMES
from yolo11.labels import label_name_of, species_of
from yolo11.mixer import source_of

# ===================== 要改的參數 =====================
MODE = 'all'                    # 要測的模式（對應 dataset_mix/pool_{MODE}/）
N_TEST = 'auto'                 # 'auto' = 呼叫 find_best_N（較慢）；或直接填整數，例如 1913
MAX_DEV = 0.05                  # 訓練總量每類別可接受的最大相對偏差
MAX_DEV_SRC = 0.10              # 單一來源每類別可接受的最大相對偏差（目標量較小，放寬）
SEED = 42
# 要測的比例組（每組各跑 5 折整數規劃，每折最多約 60 秒，可自行刪減）
# 來源池缺少某來源時，需要該來源的比例組會自動 SKIP
RATIO_SETS = [
    {'dibas': 1.0, 'matlab': 0.0, 'diffusion': 0.0},
    {'dibas': 0.5, 'matlab': 0.5, 'diffusion': 0.0},
    {'dibas': 0.5, 'matlab': 0.0, 'diffusion': 0.5},
    {'dibas': 0.4, 'matlab': 0.3, 'diffusion': 0.3},
]
USE_SIMULATED_SYNTH = False     # True → 缺的合成來源用 DIBAS 隨機子集模擬（僅驗證混合邏輯）
SIM_FRACTION = 0.6              # 每個模擬來源取 DIBAS 的比例（隨機子集）
REPRO_CHECK = False             # True → 每組再跑一次確認結果相同（時間加倍）
# =====================================================

results = []      # (名稱, 是否通過, 說明)
skipped = []      # 被跳過的比例組


def check(name, ok, detail=''):
    results.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"：{detail}" if detail else ''))
    return ok


def read_counts(lbl_dir, img_name):
    """獨立讀標籤檔，回傳 Counter(類別ID→實例數)"""
    lp = Path(lbl_dir) / label_name_of(img_name)
    if not lp.exists():
        return Counter()
    return Counter(int(ln.split()[0]) for ln in lp.read_text().splitlines() if ln.split())


def total_counts(lbl_dir, names):
    c = Counter()
    for n in names:
        c.update(read_counts(lbl_dir, n))
    return c


def max_dev(cnt, class_ids, target):
    return max(abs(cnt[k] - target) / target for k in class_ids) if target > 0 else 0.0


def build_sim_pool(src_img_dir, src_lbl_dir, missing, sim_dir, frac, seed):
    """複製（圖用符號連結）整個來源池，並把缺的來源用 DIBAS 隨機子集改名模擬。"""
    shutil.rmtree(sim_dir, ignore_errors=True)
    (sim_dir / 'images').mkdir(parents=True)
    (sim_dir / 'labels').mkdir(parents=True)

    def put(src_name, dst_name):
        src_img = (Path(src_img_dir) / src_name).resolve()
        dst_img = sim_dir / 'images' / dst_name
        try:
            os.symlink(src_img, dst_img)
        except OSError:
            shutil.copy(src_img, dst_img)
        lp = Path(src_lbl_dir) / label_name_of(src_name)
        text = lp.read_text() if lp.exists() else ''
        (sim_dir / 'labels' / label_name_of(dst_name)).write_text(text)

    names = sorted(p.name for p in Path(src_img_dir).iterdir() if p.is_file())
    for n in names:                                   # 既有來源原樣放入
        put(n, n)
    real = [n for n in names if source_of(n) == 'dibas']
    for k, s in enumerate(missing):
        rng = random.Random(seed + 1000 * (k + 1))    # 每個模擬來源取不同的隨機子集
        pick = rng.sample(real, int(len(real) * frac))
        for n in pick:
            put(n, n.replace('_dibas_', f'_{s}_', 1))
        print(f"  [模擬] {s}：取 DIBAS 隨機 {len(pick)} 張（共 {len(real)} 張）")


def main():
    db = DetectBacteria(MODE)
    pool_dir = Path(MIX_ROOT) / f'pool_{MODE}'
    if not (pool_dir / 'images').exists():
        sys.exit(f"找不到來源池 {pool_dir}，請先執行 main.py 建立（build_source_pool）")
    img_dir, lbl_dir = pool_dir / 'images', pool_dir / 'labels'

    class_ids = sorted(set(db.class_mapping.values()))
    all_names = sorted(p.name for p in Path(img_dir).iterdir() if p.is_file())
    have = {s: any(source_of(n) == s for n in all_names) for s in SOURCE_NAMES}
    if not have['dibas']:
        sys.exit("來源池內沒有 DIBAS 影像，無法測試")

    # ---------- 模擬缺少的合成來源（只用到比例組真的需要的來源） ----------
    simulated = False
    if USE_SIMULATED_SYNTH:
        needed = {s for rs in RATIO_SETS for s in SOURCE_NAMES if rs[s] > 0}
        missing = [s for s in SOURCE_NAMES if s != 'dibas' and s in needed and not have[s]]
        if missing:
            sim_dir = Path(MIX_ROOT) / f'pool_{MODE}_simtest'
            print(f"\n⚠ 來源池缺少 {missing}，改用「DIBAS 隨機子集」模擬（僅驗證混合邏輯，非真實合成資料）")
            build_sim_pool(img_dir, lbl_dir, missing, sim_dir, SIM_FRACTION, SEED)
            pool_dir = sim_dir
            img_dir, lbl_dir = pool_dir / 'images', pool_dir / 'labels'
            all_names = sorted(p.name for p in Path(img_dir).iterdir() if p.is_file())
            have = {s: any(source_of(n) == s for n in all_names) for s in SOURCE_NAMES}
            simulated = True

    db.custom_dataset_dir = pool_dir
    db._cnt_cache = {}

    print("=" * 70)
    print(f"模式 {MODE}｜類別 {class_ids}｜來源池 {pool_dir}" + ("（模擬）" if simulated else ''))
    print("=" * 70)

    # ---------- 0. 來源池本身 ----------
    print("\n[來源池檢查]")
    unknown = [n for n in all_names if source_of(n) is None]
    check("所有檔名都能辨識來源 (dibas/matlab/diffusion)", not unknown,
          f"無法辨識 {len(unknown)} 張，例如 {unknown[:3]}" if unknown else '')
    for s in SOURCE_NAMES:
        ns = [n for n in all_names if source_of(n) == s]
        c = total_counts(lbl_dir, ns)
        tag_ = '' if ns else '（無此來源，需要它的比例組會被跳過）'
        print(f"    {s:<10} {len(ns):>5} 張｜各類別實例 {[c[k] for k in class_ids]}{tag_}")
    stray = Counter()
    for n in all_names:
        for k in read_counts(lbl_dir, n):
            if k not in class_ids:
                stray[k] += 1
    check("標籤只含合法類別 ID", not stray, f"出現未知類別 {dict(stray)}" if stray else '')

    real = sorted(n for n in all_names if source_of(n) == 'dibas')
    synth_all = {s: {n for n in all_names if source_of(n) == s} for s in SOURCE_NAMES if s != 'dibas'}

    # ---------- 篩出可執行的比例組 ----------
    runnable = []
    for ratios in RATIO_SETS:
        lack = [s for s in SOURCE_NAMES if ratios[s] > 0 and not have[s]]
        if lack:
            skipped.append((ratios, lack))
        else:
            runnable.append(ratios)
    if not runnable:
        sys.exit("沒有可執行的比例組（來源池缺少所需來源）。請補來源，或設 USE_SIMULATED_SYNTH=True 模擬")

    # ---------- N ----------
    db.set_mix(runnable[0], 'test_scan', seed=SEED)
    if N_TEST == 'auto':
        N = db.find_best_N(img_dir, max_dev=MAX_DEV)
        if N is None:
            sys.exit("find_best_N 找不到合格的 N，請放寬 MAX_DEV 或改用手動 N_TEST")
    else:
        N = int(N_TEST)
    print(f"\n使用 N = {N}")

    # ---------- 逐比例組 ----------
    for ratios in runnable:
        tag = f"test_d{round(ratios['dibas']*100)}_m{round(ratios['matlab']*100)}_f{round(ratios['diffusion']*100)}"
        print("\n" + "-" * 70)
        print(f"比例組 {ratios}" + ("（含模擬來源）" if simulated and any(
            ratios[s] > 0 for s in SOURCE_NAMES if s != 'dibas') else ''))
        print("-" * 70)
        db.set_mix(ratios, tag, seed=SEED, N=N)
        folds = db.get_kfold_splits(img_dir)
        check("切出 5 折", len(folds) == 5, f"實際 {len(folds)} 折")

        val_sets, synth_picks = [], {s: [] for s in synth_all}
        for i, (tr, va, _, _) in enumerate(folds):
            print(f"\n  --- 第 {i + 1} 折 ---")
            tr_set, va_set = set(tr), set(va)

            # 1. 總量平衡
            tot = total_counts(lbl_dir, tr)
            d = max_dev(tot, class_ids, N)
            check(f"第{i+1}折 訓練總量各類別 ≈ N", d <= MAX_DEV,
                  f"各類別 {[tot[k] for k in class_ids]}，最大偏差 {d:.2%}（門檻 {MAX_DEV:.0%}）")

            # 2. 各來源
            for s in SOURCE_NAMES:
                r = ratios[s]
                part = [n for n in tr if source_of(n) == s]
                if r <= 0:
                    check(f"第{i+1}折 來源 {s}（比例 0）未被抽到", len(part) == 0, f"抽到 {len(part)} 張")
                    continue
                c = total_counts(lbl_dir, part)
                tgt = round(N * r)
                ds = max_dev(c, class_ids, tgt)
                check(f"第{i+1}折 來源 {s} 各類別 ≈ N×{r:g}={tgt}", ds <= MAX_DEV_SRC,
                      f"各類別 {[c[k] for k in class_ids]}，最大偏差 {ds:.2%}（門檻 {MAX_DEV_SRC:.0%}）")
                if s != 'dibas':
                    synth_picks[s].append(frozenset(part))

            # 3. 洩漏
            check(f"第{i+1}折 訓練∩驗證 = ∅", not (tr_set & va_set), f"重疊 {len(tr_set & va_set)} 張")
            check(f"第{i+1}折 驗證集全為真實 DIBAS", all(source_of(n) == 'dibas' for n in va))
            check(f"第{i+1}折 訓練集 DIBAS 不含驗證圖",
                  not ({n for n in tr if source_of(n) == 'dibas'} & va_set))

            # 4. 重複
            check(f"第{i+1}折 訓練集無重複圖", len(tr) == len(tr_set), f"{len(tr) - len(tr_set)} 張重複")
            check(f"第{i+1}折 驗證集無重複圖", len(va) == len(va_set))

            if db.is_mixed:
                sp = Counter(species_of(n) for n in va)
                print(f"    驗證集菌種張數 {dict(sp)}")
            val_sets.append(va_set)

        # 5. 驗證集互斥且完整
        print("\n  --- 跨折檢查 ---")
        union = set().union(*val_sets)
        overlap = sum(len(a & b) for i, a in enumerate(val_sets) for b in val_sets[i + 1:])
        check("5 折驗證集互不重疊", overlap == 0, f"重疊 {overlap} 張")
        check("5 折驗證集聯集 = 全部真實 DIBAS", union == set(real),
              f"聯集 {len(union)} 張 / 真實 {len(real)} 張")

        # 6. 分層比例
        if db.is_mixed:
            sp_ratio = []
            for v in val_sets:
                c = Counter(species_of(n) for n in v)
                sp_ratio.append(c.get('cocci', 0) / max(1, sum(c.values())))
            spread = max(sp_ratio) - min(sp_ratio)
            check("各折驗證集菌種比例接近（分層）", spread <= 0.05,
                  f"cocci 佔比 {[f'{x:.1%}' for x in sp_ratio]}，差距 {spread:.1%}")

        # 7. 合成來源每折子集不應完全相同
        for s, picks in synth_picks.items():
            if len(picks) == 5:
                n_distinct = len(set(picks))
                check(f"來源 {s} 各折抽到的子集不完全相同（per-fold seed 生效）", n_distinct > 1,
                      f"5 折中有 {n_distinct} 種不同子集")

        # 8. 可重現
        if REPRO_CHECK:
            folds2 = db.get_kfold_splits(img_dir)
            same = all(set(a[0]) == set(b[0]) and set(a[1]) == set(b[1])
                       for a, b in zip(folds, folds2))
            check("相同參數再跑一次結果完全相同（可重現）", same)

    # ---------- 總結 ----------
    print("\n" + "=" * 70)
    n_fail = sum(1 for _, ok, _ in results if not ok)
    print(f"共 {len(results)} 項檢查，通過 {len(results) - n_fail}，失敗 {n_fail}")
    if skipped:
        print("\n已跳過（來源池缺少所需來源，不算失敗）：")
        for ratios, lack in skipped:
            print(f"  - {ratios}  缺 {lack}")
    if simulated:
        print("\n⚠ 含模擬來源的比例組只驗證混合邏輯；真實合成資料到手後請關閉 USE_SIMULATED_SYNTH 重測。")
    if n_fail:
        print("\n失敗項目：")
        for name, ok, detail in results:
            if not ok:
                print(f"  ✗ {name}：{detail}")
    else:
        print("✓ 已執行的檢查全部通過")
    print("=" * 70)
    print(f"（測試會在 {MIX_ROOT}/ 產生 splits_test_*.json" + (f" 與 pool_{MODE}_simtest/" if simulated else '')
          + "，可手動刪除）")
    sys.exit(1 if n_fail else 0)


if __name__ == '__main__':
    main()