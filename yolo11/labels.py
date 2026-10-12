# -*- coding: utf-8 -*-
"""
labels：標籤與檔名的純字串處理
做單元測試。
"""
def label_name_of(img_name): #輸入:影像檔名
    """利用影像檔名轉換標籤檔名 例：cocci_001.jpg → cocci_001.txt"""
    #從右邊開始切1次把結果裝進一個串列（List）裡，抓取第一個元素
    return img_name.rsplit('.', 1)[0] + '.txt'  #回傳標籤檔名

def species_of(filename):  #輸入:影像資料夾內的影像檔名
    """利用影像檔名前綴取得物種（cocci_001.jpg → 'cocci'）"""
    return filename.split('_', 1)[0] #回傳菌種字串


def convert_label_lines(lines, label_map): #輸入:標籤內容清單、類別ID對應表
    """
    轉換第一欄位後的YOLO 標籤檔
    - 不在轉換表內的 ID 原樣保留（交給事後檢查）；
    - 無法解析類別 ID 的行略過；空行略過。
    """
    new_lines, out_ids = [], set() #建立一個空清單(轉換後的新標籤內容)、空集合(不允許元素重複的容器所有出現過的類別 ID)
    for line in lines: #遍歷每一列標籤內容
        parts = line.split() # 將單行文字按照空白字元切開，裝進一個（List）裡
        if not parts: #如果清單裡沒有元素
            continue #進入下一次迴圈(下一列)
        try:
            old_id = int(parts[0]) #原始標籤id:將清單中第一個字串元素轉換成整數格式
        except ValueError:
            continue #進入下一次迴圈
        new_id = label_map.get(old_id, old_id) #轉換:想要在字典裡尋找的鍵對應的值，預設回傳鍵值
        parts[0] = str(new_id) #整數轉字串取代清單第一欄
        out_ids.add(new_id) #輸出類別ID集合內增加1筆新轉換後的類別id
        #把清單裡的每一個字串取出來，中間用空格 (' ') 把它們串在一起。例如:['2 0.52 0.43 0.10 0.12\n', '5 0.75 0.31 0.08 0.09\n',...]
        new_lines.append(' '.join(parts) + '\n')#增加一列字串清單元素:新標籤內容清單內增加1筆(列)轉換後的新標籤內容清單(換列)
    return new_lines, out_ids #新標籤內容清單、輸出類別ID 集合