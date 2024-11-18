# train.py

import torch
import torch.optim as optim
import torch.nn as nn
from model import NestedUNet  # 匯入模型
from dataloader import get_dataloaders  # 匯入自定義的dataloader
from tqdm import tqdm  # 匯入 tqdm 進度條
import os
import yaml  # 用於讀取 yaml 文件

# 檢查是否有可用的 GPU，否則使用 CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_config(config_path="config.yaml"):
    """ 加載 YAML 配置文件 """
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

def train_model(config):
    # 获取训练和验证路径，如果未指定則使用默認路径
    base_dir = os.path.dirname(os.path.abspath(__file__))
    default_train_image_dir = os.path.join(base_dir, "dataset/train/images")
    default_train_mask_dir = os.path.join(base_dir, "dataset/train/masks")
    default_val_image_dir = os.path.join(base_dir, "dataset/val/images")
    default_val_mask_dir = os.path.join(base_dir, "dataset/val/masks")

    train_image_dir = config.get('train_image_dir') or default_train_image_dir
    train_mask_dir = config.get('train_mask_dir') or default_train_mask_dir
    val_image_dir = config.get('val_image_dir') or default_val_image_dir
    val_mask_dir = config.get('val_mask_dir') or default_val_mask_dir

    # 其他训练参数
    num_epochs = config['num_epochs']
    save_interval = config['save_interval']
    val_interval = config['val_interval']
    patience = config['patience']
    batch_size = config['batch_size']
    num_workers = config['num_workers']

    # 獲取訓練和驗證數據的 DataLoader
    train_loader, val_loader = get_dataloaders(
        train_image_dir, train_mask_dir, val_image_dir, val_mask_dir, batch_size=batch_size, num_workers=num_workers)

    # 實例化模型並將模型移動到計算裝置（GPU 或 CPU）
    model = NestedUNet(in_channels=3, out_channels=1, deep_supervision=False).to(device)

    # 定義優化器
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 定義損失函式 (二類分類任務)
    criterion = nn.BCELoss()

    # 訓練循環
    save_dir = "saved_models"
    os.makedirs(save_dir, exist_ok=True)

    best_val_loss = float('inf')  # 初始化最小驗證集 loss
    trigger_times = 0  # 記錄驗證集未改善的次數

    for epoch in range(num_epochs):
        model.train()  # 設定為訓練模式
        running_loss = 0.0

        # 使用 tqdm 为训练进度条
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch") as pbar:
            for images, masks in train_loader:
                images = images.to(device)
                masks = masks.to(device)

                optimizer.zero_grad()

                outputs = model(images)
                loss = criterion(outputs, masks)

                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
                pbar.update(1)

        print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {running_loss/len(train_loader)}")

        # 每 save_interval 個 epochs 儲存一次模型
        if (epoch + 1) % save_interval == 0:
            model_path = os.path.join(save_dir, f'unetpp_epoch_{epoch+1}.pth')
            torch.save(model.state_dict(), model_path)
            print(f"Model saved at {model_path}")

        # 每 val_interval 個 epochs 驗證一次
        if (epoch + 1) % val_interval == 0:
            model.eval()  # 設定為評估模式
            val_loss = 0.0
            with torch.no_grad():
                with tqdm(total=len(val_loader), desc=f"Validation", unit="batch") as pbar:
                    for val_images, val_masks in val_loader:
                        val_images = val_images.to(device)
                        val_masks = val_masks.to(device)

                        val_outputs = model(val_images)
                        loss = criterion(val_outputs, val_masks)
                        val_loss += loss.item()

                        pbar.set_postfix({"Val Loss": f"{loss.item():.4f}"})
                        pbar.update(1)

            val_loss = val_loss / len(val_loader)
            print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss}")

            # 早停機制
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))  # 保存最佳模型
                print(f"Best model saved at Epoch {epoch+1} with Validation Loss: {val_loss}")
                trigger_times = 0  # 重置 trigger 次數
            else:
                trigger_times += 1
                print(f"Trigger times: {trigger_times}")
                if trigger_times >= patience:
                    print("Early stopping triggered")
                    # 保存早停時的模型
                    final_model_path = os.path.join(save_dir, f'unetpp_epoch_{epoch+1}_earlystop.pth')
                    torch.save(model.state_dict(), final_model_path)
                    print(f"Early stopped model saved at {final_model_path}")
                    break

        torch.cuda.empty_cache()  # 清理顯存

if __name__ == "__main__":
    # 加載配置文件
    config = load_config()

    # 進行模型訓練
    train_model(config)
