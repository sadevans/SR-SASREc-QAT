import random
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

class DIV2KTrainDataset(Dataset):
    def __init__(self, hr_dir, patch_size=96, upscale_factor=4, rgb_range=1.0):
        hr_dir = Path(hr_dir)
        if not hr_dir.exists():
            raise FileNotFoundError(f"HR directory does not exist: {hr_dir}")
        self.hr_files = sorted(
            p for ext in ("*.png", "*.jpg", "*.jpeg")
            for p in hr_dir.rglob(ext)
        )
        if not self.hr_files:
            raise FileNotFoundError(f"No HR images found in {hr_dir}")
        if patch_size % upscale_factor != 0:
            raise ValueError(f"patch_size ({patch_size}) must be divisible by scale ({upscale_factor})")
        self.gt_image_size = int(17 * upscale_factor)
        self.patch_size = patch_size
        self.upscale_factor = upscale_factor
        self.rgb_range = rgb_range
        self.transform = T.ToTensor()

    def __len__(self):
        return len(self.hr_files)

    def __getitem__(self, idx):
        hr = Image.open(self.hr_files[idx]).convert("RGB")
        w, h = hr.size
        if w < self.patch_size or h < self.patch_size:
            raise ValueError(
                f"Patch size {self.patch_size} is larger than HR image size {(w, h)} for {self.hr_files[idx]}"
            )
        x = random.randint(0, w - self.patch_size)
        y = random.randint(0, h - self.patch_size)
        hr = hr.crop((x, y, x + self.patch_size, y + self.patch_size))
        lr = hr.resize((self.patch_size // self.upscale_factor, self.patch_size // self.upscale_factor), Image.BICUBIC)
        hr = self.transform(hr) * self.rgb_range
        lr = self.transform(lr) * self.rgb_range
        return lr, hr


class SRBenchmarkDataset(Dataset):
    def __init__(self, hr_dir, scale=3, rgb_range=1.0):
        hr_dir = Path(hr_dir)
        if not hr_dir.exists():
            raise FileNotFoundError(f"HR directory does not exist: {hr_dir}")
        scale_tag = f"SRF_{scale}_HR.png"
        self.hr_files = sorted(hr_dir.rglob(f"*{scale_tag}"))
        if len(self.hr_files) == 0:
            raise FileNotFoundError(f"No HR images for scale x{scale} found in {hr_dir}")
        self.scale = scale
        self.rgb_range = rgb_range
        self.transform = T.ToTensor()

    def __len__(self):
        return len(self.hr_files)

    def __getitem__(self, idx):
        hr_path = self.hr_files[idx]
        name = hr_path.stem

        base_name = name.replace("_HR", "")
        lr_name = base_name + "_LR.png"

        lr_path = hr_path.parent / lr_name

        if not lr_path.exists():
            raise FileNotFoundError(f"LR image not found: {lr_path}")

        hr = Image.open(hr_path).convert("RGB")
        lr = Image.open(lr_path).convert("RGB")

        w_lr, h_lr = lr.size
        w_hr_expected = w_lr * self.scale
        h_hr_expected = h_lr * self.scale
        w_hr, h_hr = hr.size
        if w_hr != w_hr_expected or h_hr != h_hr_expected:
            hr = hr.crop((0, 0, w_hr_expected, h_hr_expected))

        hr_tensor = T.ToTensor()(hr) * self.rgb_range  # (1, H, W)
        lr_tensor = T.ToTensor()(lr) * self.rgb_range  # (1, h, w)

        return lr_tensor, hr_tensor, name


def get_train_loader(config):
    ds = DIV2KTrainDataset(
        hr_dir=config['data']['train_dir'],
        patch_size=config['data']['patch_size'],
        upscale_factor=config['model']['upscale_factor'],
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
