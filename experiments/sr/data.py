import os
import random
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms as T


class DIV2KTrainDataset(Dataset):
    def __init__(self, hr_dir, patch_size=96, scale=3, rgb_range=1.0):
        self.hr_files = sorted(Path(hr_dir).glob("*.png"))
        self.patch_size = patch_size
        self.scale = scale
        self.rgb_range = rgb_range
        self.transform = T.ToTensor()

    def __len__(self):
        return len(self.hr_files)

    def __getitem__(self, idx):
        hr = Image.open(self.hr_files[idx]).convert("RGB")
        w, h = hr.size
        # Random crop
        x = random.randint(0, w - self.patch_size)
        y = random.randint(0, h - self.patch_size)
        hr = hr.crop((x, y, x + self.patch_size, y + self.patch_size))
        lr = hr.resize((self.patch_size // self.scale, self.patch_size // self.scale), Image.BICUBIC)
        hr = self.transform(hr) * self.rgb_range
        lr = self.transform(lr) * self.rgb_range
        return lr, hr


class SRBenchmarkDataset(Dataset):
    def __init__(self, hr_dir, scale=3, rgb_range=1.0):
        self.hr_files = sorted(Path(hr_dir).rglob("*_HR.png"))
        if len(self.hr_files) == 0:
            raise FileNotFoundError(f"No HR images found in {hr_dir}")
        self.scale = scale
        self.rgb_range = rgb_range
        self.transform = T.ToTensor()

    def __len__(self):
        return len(self.hr_files)

    def __getitem__(self, idx):
        hr_path = self.hr_files[idx]
        name = hr_path.stem  # e.g. "img_001_SRF_2_HR"

        # Извлекаем базовое имя и масштаб
        base_name = name.replace("_HR", "")  # "img_001_SRF_2"
        lr_name = base_name + "_LR.png"     # "img_001_SRF_2_LR.png"

        # Путь к LR: в той же директории, что и HR
        lr_path = hr_path.parent / lr_name

        if not lr_path.exists():
            raise FileNotFoundError(f"LR image not found: {lr_path}")

        hr = Image.open(hr_path).convert("RGB")
        lr = Image.open(lr_path).convert("RGB")

        # Убедитесь, что размеры совпадают по масштабу
        w_lr, h_lr = lr.size
        w_hr_target = w_lr * self.scale
        h_hr_target = h_lr * self.scale
        w_hr, h_hr = hr.size

        if w_hr != w_hr_target or h_hr != h_hr_target:
            # Обрезаем HR до нужного размера
            hr = hr.crop((0, 0, w_hr_target, h_hr_target))

        hr = self.transform(hr) * self.rgb_range
        lr = self.transform(lr) * self.rgb_range
        return lr, hr, name


def get_train_loader(config):
    ds = DIV2KTrainDataset(
        hr_dir=config['data']['train_dir'],
        patch_size=config['data']['patch_size'],
        scale=config['model']['upscale_factor'],
        rgb_range=config['data']['rgb_range']
    )
    return DataLoader(
        ds,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        pin_memory=config['training']['pin_memory']
    )


def get_val_datasets(config):
    return [
        SRBenchmarkDataset(hr_dir=dir, scale=config['model']['upscale_factor'], rgb_range=config['data']['rgb_range'])
        for dir in config['data']['val_dirs']
    ]
