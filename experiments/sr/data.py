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
        self.hr_files = sorted(Path(hr_dir).glob("*.png"))
        self.scale = scale
        self.rgb_range = rgb_range
        self.transform = T.ToTensor()

    def __len__(self):
        return len(self.hr_files)

    def __getitem__(self, idx):
        hr = Image.open(self.hr_files[idx]).convert("RGB")
        w, h = hr.size
        lr = hr.resize((w // self.scale, h // self.scale), Image.BICUBIC)
        hr = self.transform(hr) * self.rgb_range
        lr = self.transform(lr) * self.rgb_range
        name = self.hr_files[idx].stem
        return lr, hr, name


def get_train_loader(config):
    ds = DIV2KTrainDataset(
        hr_dir=config.data.train_dir,
        patch_size=config.data.patch_size,
        scale=config.model.upscale_factor,
        rgb_range=config.data.rgb_range
    )
    return DataLoader(
        ds,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory
    )


def get_val_datasets(config):
    return [
        SRBenchmarkDataset(hr_dir=dir, scale=config.model.upscale_factor, rgb_range=config.data.rgb_range)
        for dir in config.data.val_dirs
    ]
