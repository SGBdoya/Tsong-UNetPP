# dataloader.py

import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms


class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, mask_transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.mask_transform = mask_transform
        # 使用 sorted 對檔名進行排序，確保順序一致
        self.images = sorted(os.listdir(image_dir))
        self.masks = sorted(os.listdir(mask_dir))  # 同樣對 masks 進行排序

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])  # 確保 masks 按相同順序讀取

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # 轉換為單通道灰度圖

        # 應用預處理
        if self.transform is not None:
            image = self.transform(image)
        if self.mask_transform is not None:
            mask = self.mask_transform(mask)

        return image, mask




def get_dataloaders(train_image_dir, train_mask_dir, val_image_dir, val_mask_dir, batch_size=8, num_workers=4):
    # 影像和標籤的轉換
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])

    mask_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # 訓練集和驗證集
    train_dataset = SegmentationDataset(
        train_image_dir, train_mask_dir, transform=transform, mask_transform=mask_transform)
    val_dataset = SegmentationDataset(
        val_image_dir, val_mask_dir, transform=transform, mask_transform=mask_transform)

    # DataLoader
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader
