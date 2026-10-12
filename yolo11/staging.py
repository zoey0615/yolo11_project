# -*- coding: utf-8 -*-
"""
staging(暫存)：影像與資料集的檔案處理
"""
import os #處理檔案路徑
import shutil #Shell Utilities（命令列與高階檔案工具箱）:整批複製檔案
from pathlib import Path #把「檔案路徑」當成一個「物件」來操作
import cv2 #Computer Vision:影像前處理
import yaml #產生 (寫入)YAML 格式設定檔
#套件內部依賴，使用相對匯入的寫法
from .config import IMAGE_EXTS # 常數與模式設定:支援的影像副檔名（比對時一律轉小寫）
from .labels import convert_label_lines, label_name_of #  轉換第一欄位後的YOLO 標籤檔、利用影像檔名轉換標籤檔名 


# ============================================================================
# 影像檔名:確保只抓取真正合法的圖片檔案、確保每次程式跑出來的圖片順序都是固定、一致的
# ============================================================================
def is_image_name(name): #輸入影像檔名
    """副檔名是否為支援的影像格式"""
    return Path(name).suffix.lower() in IMAGE_EXTS #從影像檔名取出檔名後綴(副檔名)是否包含在支援的影像副檔名清單內

def list_image_names(folder): #輸入:影像資料夾路徑
    """回傳影像資料夾內排序後的影像檔名清單"""
    #Iterate Directory（迭代資料夾），如果資料夾中是檔案且副檔名正確，存入f.name (影像檔名)
    #依照字元在電腦編碼（Unicode / ASCII）中的順序，由小到大排好隊。
    return sorted(f.name for f in Path(folder).iterdir() if f.is_file() and is_image_name(f.name)) #回傳檔名字串清單

def to_gray_3ch(img): #輸入:1張影像(圖片高度, 圖片寬度, 3)
    """BGR 影像 → 灰階後複製成 3 通道"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #轉換影像色彩空間(影像矩陣,指定的通道權重公式)
    return cv2.merge([gray, gray, gray]) #回傳:複製成 3 通道的灰階影像矩陣

def read_gray_3ch(path): #輸入:單張影像的檔案路徑path物件
    """讀取影像並轉灰階 3 通道；讀取失敗回傳 None"""
    img = cv2.imread(str(path)) #讀取影像轉換成 NumPy 陣列。
    return None if img is None else to_gray_3ch(img) #回傳:複製成 3 通道的灰階影像矩陣


# ============================================================================
# 暫存資料集建立
# ============================================================================
def stage_image(src, dst): #輸入:原始影像的完整檔案路徑物件、暫存影像檔案路徑物件
    """把影像轉灰階 3 通道後存到 dst"""
    if src.exists() and not dst.exists(): #如果『來源檔案存在』，而且『暫存檔還不存在』
        gray = read_gray_3ch(src)  #回傳:複製成 3 通道的灰階影像矩陣
        if gray is not None: #如果灰階影像矩陣不是空的
            cv2.imwrite(str(dst), gray) #將灰階矩陣寫入硬碟中，變成一個實體的圖片檔案。
        else: #影像讀取失敗（img 為 None）
            shutil.copy(src, dst) #直接把原始檔案複製過去


def stage_label(src, dst, label_map): #輸入:來源標籤檔案路徑物件、暫存標籤檔路徑物件、類別 ID 對應字典
    """若來源標籤存在，轉換類別ID後寫到dst"""
    if src.exists(): #如果來源標籤檔案路徑物件存在
        with open(src, 'r') as f: #開啟檔案，讀取模式
            lines = f.readlines() #讀取所有列，加入換行符號存入清單
        #轉換第一欄位後的YOLO 標籤檔(不在轉換表內的 ID 原樣保留)
        new_lines, _ = convert_label_lines(lines, label_map)  #輸入:標籤內容清單、類別ID對應表 回傳:新標籤內容清單、輸出類別ID 集合
        with open(dst, 'w') as f: #開啟檔案，寫入模式
            f.writelines(new_lines) #一次寫入多行字串（一個新標籤內容清單）到暫存標籤路徑的對應檔案中

#輸入:影像檔名清單、原始影像/標籤來源資料夾路徑、暫存影像/標籤資料夾路徑、類別id字典
def stage_split(names, src_img_dir, src_lbl_dir, dst_img_dir, dst_lbl_dir, label_map):
    """
    把一組影像（檔名清單）/與標籤檔處理後放進暫存資料集
    """
    src_img_dir, src_lbl_dir = Path(src_img_dir), Path(src_lbl_dir) #原始路徑
    dst_img_dir, dst_lbl_dir = Path(dst_img_dir), Path(dst_lbl_dir) #暫存路徑
    if not src_img_dir.exists(): #if not true->不成立;if not false->成立
        return [] #暫存路徑不存在的空清單
    paths = [] #暫存資料夾內影像檔案絕對路徑清單
    for name in names:
        if not is_image_name(name): #if not false->成立
            continue
        dst_img = dst_img_dir / name
        stage_image(src_img_dir / name, dst_img) #轉灰階
        paths.append(str(dst_img.absolute()))
        lbl = label_name_of(name) #輸入:影像檔名 回傳:標籤檔名
        stage_label(src_lbl_dir / lbl, dst_lbl_dir / lbl, label_map) #若來源標籤存在，轉換類別ID後寫到dst
    return paths #  回傳(暫存)處理後影像的絕對路徑清單


# ============================================================================
# 清單檔與 YOLO 設定檔
# ============================================================================
def write_path_list(path, paths): #輸入:包含影像路徑的文字檔、(暫存)處理後影像的絕對路徑清單
    """寫入影像路徑清單檔（每行一個路徑）"""
    with open(path, 'w') as f:
        f.write('\n'.join(paths) + '\n') #把分隔符號插在元素與元素的中間

#輸入:YAML檔案路徑、訓練/驗證/測試圖片清單絕對路徑文字檔、資料集根目錄、類別總數、類別名稱清單
def write_data_yaml(path, train, val, root, nc, names, test=None):
    """寫入 YOLO 資料集設定檔"""
    data = {'train': os.path.abspath(train), 'val': os.path.abspath(val)} #核心必填
    if test is not None: #if test is not false->成立
        data['test'] = os.path.abspath(test) #鍵值(動態選填)
    data.update({'path': os.path.abspath(root), 'nc': nc, 'names': list(names)}) #後設資料
    with open(path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False) #倒（Dump）-區塊風格