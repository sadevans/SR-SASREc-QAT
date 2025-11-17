#!/usr/bin/env python3
"""
Download DIV2K, Set5, and Set14 datasets for Super Resolution.
Saves to:
  data/DIV2K_train_HR/
  data/Set5/
  data/Set14/
"""

import os
import sys
import zipfile
import tarfile
import argparse
from pathlib import Path
import urllib.request
from tqdm import tqdm


def download_t91(data_root: Path):
    t91_dir = data_root / "T91"
    if not t91_dir.exists():
        print("📥 Downloading T91 (training set)...")
        url = "https://github.com/tony-yin/SRGAN-PyTorch/raw/master/data/T91.zip"
        zip_path = data_root / "T91.zip"
        download_url(url, zip_path)
        extract_archive(zip_path, data_root)
        zip_path.unlink()
    else:
        print("✅ T91 already exists.")


def download_and_extract_figshare(data_root: Path):
    archive_path = data_root / "sr_benchmarks.zip"
    set5_target = data_root / "Set5"
    set14_target = data_root / "Set14"

    # Скачиваем основной архив, если нужно
    if not archive_path.exists():
        print("📥 Downloading SR benchmarks from figshare...")
        url = "https://figshare.com/ndownloader/articles/21586188/versions/1"
        download_url(url, archive_path)

    # Временная папка для распаковки .zip файлов
    temp_dir = data_root / "_temp_sr_benchmarks"

    if not set5_target.exists() or not set14_target.exists():
        print("📦 Extracting inner ZIP archives...")

        # Распаковываем основной архив → получаем Set5.zip, Set14.zip и т.д.
        extract_archive(archive_path, temp_dir)

        # Распаковываем Set5.zip → Set5/
        if (temp_dir / "Set5.zip").exists() and not set5_target.exists():
            extract_archive(temp_dir / "Set5.zip", data_root)
        else:
            print("⚠️  Set5.zip not found or already extracted.")

        # Распаковываем Set14.zip → Set14/
        if (temp_dir / "Set14.zip").exists() and not set14_target.exists():
            extract_archive(temp_dir / "Set14.zip", data_root)
        else:
            print("⚠️  Set14.zip not found or already extracted.")

        # Очистка
        import shutil
        shutil.rmtree(temp_dir)

    archive_path.unlink(missing_ok=True)

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url: str, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=output_path.name) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def extract_archive(archive_path: Path, extract_to: Path):
    extract_to.mkdir(parents=True, exist_ok=True)
    if archive_path.suffix == ".zip":
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
    elif archive_path.suffixes[-2:] == [".tar", ".gz"]:
        with tarfile.open(archive_path, "r:gz") as tar_ref:
            tar_ref.extractall(extract_to)
    else:
        raise ValueError(f"Unsupported archive format: {archive_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="data", help="Root directory for datasets")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    data_root.mkdir(exist_ok=True)

    # # === DIV2K (High-Resolution Training Set) ===
    # div2k_dir = data_root / "DIV2K_train_HR"
    # if not div2k_dir.exists() or not any(div2k_dir.iterdir()):
    #     print("📥 Downloading DIV2K_train_HR...")
    #     url = "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip"
    #     zip_path = data_root / "DIV2K_train_HR.zip"
    #     if not zip_path.exists():
    #         download_url(url, zip_path)
    #     print("📦 Extracting DIV2K...")
    #     extract_archive(zip_path, data_root)
    #     zip_path.unlink()  # remove zip after extraction
    # else:
    #     print("✅ DIV2K_train_HR already exists.")

    # === Set5 ===
    # set5_dir = data_root / "Set5"
    # if not set5_dir.exists():
    print("📥 Downloading Set5&Set14...")
    download_and_extract_figshare(data_root)
    #     url = "https://github.com/brohrer/Set5_SR/raw/main/Set5.zip"
    #     zip_path = data_root / "Set5.zip"
    #     download_url(url, zip_path)
    #     extract_archive(zip_path, data_root)
    #     zip_path.unlink()
    # else:
    #     print("✅ Set5 already exists.")

    # # === Set14 ===
    # set14_dir = data_root / "Set14"
    # if not set14_dir.exists():
    #     print("📥 Downloading Set14...")
    #     url = "https://github.com/brohrer/Set14_SR/raw/main/Set14.zip"
    #     zip_path = data_root / "Set14.zip"
    #     download_url(url, zip_path)
    #     extract_archive(zip_path, data_root)
    #     zip_path.unlink()
    # else:
    #     print("✅ Set14 already exists.")

    print("\n🎉 All datasets downloaded and ready!")
    print(f"📁 Location: {data_root.resolve()}")


if __name__ == "__main__":
    main()
