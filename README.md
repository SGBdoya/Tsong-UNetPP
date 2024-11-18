# Tsong UNetPP

**UNet++練習**

UNet++模型練習，可透過config快速設定

---

## 目錄

**詳細說明請點目錄連結**

1. [簡介](#簡介)

---

## 簡介

UNet++
通過config進行快速參數設定，有patience早停機制，best模型是保存在loss最後一次下降的地方

---

## 模組概覽


| 模組名稱        | 功能簡介   |
| --------------- | ---------- |
| `train.py`      | 訓練主程式 |
| `test.py`       | 測試主程式 |
| `model.py`      | UNet++網路 |
| `dataloader.py` | 資料讀取器 |

---

## dataset準備範例

1. 可以使用labelme標註資料 https://github.com/wkentaro/labelme

2. 使用 `labelme_2_mask.py` 將labelme的json檔案轉換成mask [**labelme_2_mask使用方法**](#labelme_2_mask使用方法)

3. 預設資料夾結構如下

    ```plaintext
    dataset/
    ├── train/
    │   ├── images
    │   ├── masks
    │
    ├── val/
    │   ├── images
    │   ├── masks
    │ 
    └── test/
        ├── images
        ├── groundtruth
    ```

4. 修改config.yaml
    | config                 | 功能簡介                                                    |
    | ---------------------- | ----------------------------------------------------------- |
    | `num_epochs`           | 訓練回合                                                    |
    | `save_interval`        | 多少個epochs保存一次模型                                    |
    | `val_interval`         | 多少個epochs驗證一次模型                                    |
    | `patience`             | 耐心值，當loss持續沒有下降則超過該值後早停                  |
    | `batch_size`           | 批次大小                                                    |
    | `num_workers`          | dataloader的的workers                                       |
    | `train_image_dir`      | 訓練圖片路徑 若為null 則依照以上描述的資料夾結構            |
    | `train_mask_dir`       | 訓練遮罩路徑 若為null 則依照以上描述的資料夾結構            |
    | `val_image_dir`        | 驗證圖片路徑 若為null 則依照以上描述的資料夾結構            |
    | `val_mask_dir`         | 驗證遮罩路徑 若為null 則依照以上描述的資料夾結構            |
    | `test_image_dir`       | 測試圖片路徑 若為null 則依照以上描述的資料夾結構            |
    | `test_groundtruth_dir` | 測試真實狀況路徑 則依照以上描述的資料夾結構                 |
    | `model_path`           | test的載入模型路徑 若為null 則抓saved_models/best_model.pth |


5.運行`train.py`進行訓練

---

## labelme_2_mask使用方法

1. 修改輸入資料夾跟輸出資料夾
    ```python
    # 使用示例
    if __name__ =="__main__":
        input_folder = r'train_json' # json存在的路徑
        output_folder = r'output' # 輸出的資料夾位置
        process_folder(input_folder, output_folder)

    ```
2. 運行 `python labelme_2_mask.py`