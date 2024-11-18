# test.py

import torch
import torch.nn as nn
from model import NestedUNet  # 匯入模型
from dataloader import SegmentationDataset  # 匯入自定義的dataloader
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score
import os
import yaml  # 用於讀取 yaml 檔案
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  # 用於繪製混淆矩陣

# 檢查是否有可用的 GPU，否則使用 CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_config(config_path="config.yaml"):
    """ 加載 YAML 配置檔案 """
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

def plot_confusion_matrix(cm, output_path):
    """ 繪製並儲存混淆矩陣，標籤為'有瑕疵'和'無瑕疵' """
    labels = ['Good', 'Defect']  # 修改爲你需要的分類標籤
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix (有瑕疵 vs 無瑕疵)")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.savefig(output_path)
    plt.close()

def test_model(config):
    # 獲取測試路徑和模型路徑，如果未指定則使用預設路徑
    base_dir = os.path.dirname(os.path.abspath(__file__))
    default_test_image_dir = os.path.join(base_dir, "dataset/test/images")
    default_test_groundtruth_dir = os.path.join(base_dir, "dataset/test/groundtruth")
    default_model_path = os.path.join(base_dir, "saved_models/best_model.pth")

    test_image_dir = config.get('test_image_dir') or default_test_image_dir
    test_groundtruth_dir = config.get('test_groundtruth_dir') or default_test_groundtruth_dir
    model_path = config.get('model_path') or default_model_path

    # 影象和標籤的轉換
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    mask_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # 載入測試數據集
    test_dataset = SegmentationDataset(test_image_dir, test_groundtruth_dir, transform=transform, mask_transform=mask_transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # 載入模型
    model = NestedUNet(in_channels=3, out_channels=1, deep_supervision=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # 評估指標
    all_preds = []
    all_targets = []

    # 建立儲存預測影象的資料夾
    os.makedirs("results", exist_ok=True)

    for idx, (image, groundtruth) in enumerate(test_loader):
        image = image.to(device)
        groundtruth = groundtruth.to(device)
        
        with torch.no_grad():
            output = model(image)
            pred = (output > 0.5).float()  # 二分類閾值

        # 修正影象轉換，去除批量維度
        original_img = image.cpu().squeeze(0).numpy().transpose(1, 2, 0)
        original_img = (original_img - original_img.min()) / (original_img.max() - original_img.min())  # 歸一化到0-1

        pred_img = pred.cpu().numpy().squeeze() * 255
        gt_img = groundtruth.cpu().numpy().squeeze() * 255

        # 將原始影象、預測影象和真實標籤拼接在一起
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))

        ax[0].imshow(original_img)
        ax[0].set_title('Original Image')
        ax[0].axis('off')

        ax[1].imshow(pred_img, cmap='gray')
        ax[1].set_title('Predicted Mask')
        ax[1].axis('off')

        ax[2].imshow(gt_img, cmap='gray')
        ax[2].set_title('Ground Truth')
        ax[2].axis('off')

        # 儲存合成的影象
        plt.savefig(f"results/result_{idx}.png")
        plt.close()

        # 收集用於評估的所有預測和 groundtruth
        all_preds.append(pred.cpu().numpy().flatten())
        all_targets.append(groundtruth.cpu().numpy().flatten())

    # 將所有預測和標籤轉為 NumPy 陣列
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    # 計算混淆矩陣
    conf_matrix = confusion_matrix(all_targets, all_preds)
    print("Confusion Matrix:\n", conf_matrix)

    # 儲存混淆矩陣為圖片
    os.makedirs("runs", exist_ok=True)
    plot_confusion_matrix(conf_matrix, "runs/confusion_matrix.png")

    # 計算評估指標
    recall = recall_score(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds)

    print(f"Recall: {recall:.4f}, Precision: {precision:.4f}, F1 Score: {f1:.4f}")

    # 儲存評估結果到 'runs' 資料夾
    with open("runs/metrics.txt", "w") as f:
        f.write(f"Confusion Matrix:\n{conf_matrix}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")

    print("Test results and metrics saved in 'runs' folder.")

if __name__ == "__main__":
    # 載入配置檔案
    config = load_config()

    # 進行模型測試
    test_model(config)
