import os
import json
import numpy as np
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from utils.file import download_file, unzip_file, extract_tar_gz
from config.config import get_config


class Caltech101(Dataset):
    def __init__(self, transform=None, download=True):
        self.transform = transform

        # Load configuration
        config = get_config()
        self.dataset_root = Path(config.get('dataset_root', './datas/'))

        # Ensure dataset directory exists
        self.dataset_root.mkdir(parents=True, exist_ok=True)

        # Paths
        self.zip_path = self.dataset_root / 'caltech-101.zip'
        self.json_path = self.dataset_root / 'split_zhou_Caltech101.json'
        self.extract_path = self.dataset_root / 'caltech-101' / '101_ObjectCategories'

        # Download dataset if requested
        if download:
            self._download_dataset()

        # Load split information
        if not self.json_path.exists():
            raise FileNotFoundError(f"Split JSON not found: {self.json_path}")

        with open(self.json_path, 'r') as f:
            split_data = json.load(f)

        train_data = np.array(split_data['train'])

        self.fnames = train_data[:, 0].tolist()
        self.labels = train_data[:, 1].astype(int)
        old_list = train_data[:, 2]
        self.classes = list(dict.fromkeys(old_list))

        # Precompute full image paths
        self.full_paths = [self.extract_path / fname for fname in self.fnames]

        if len(self.full_paths) == 0:
            raise RuntimeError("No images found in Caltech101 dataset.")

    def _download_dataset(self):
        # Download zip if missing
        if not self.zip_path.exists():
            print("Downloading Caltech101 zip...")
            download_file(
                'https://data.caltech.edu/records/mzrjq-6wc02/files/caltech-101.zip?download=1',
                str(self.zip_path)
            )

        # Download split JSON if missing
        if not self.json_path.exists():
            print("Downloading split JSON...")
            download_file(
                'https://drive.usercontent.google.com/download?id=1hyarUivQE36mY6jSomru6Fjd-JzwcCzN&export=download&authuser=0',
                str(self.json_path)
            )

        # Extract dataset
        if not self.extract_path.exists():
            print("Extracting Caltech101 zip...")
            unzip_file(str(self.zip_path), str(self.dataset_root / 'caltech-101'))
            tar_gz_file = self.dataset_root / 'caltech-101' / '101_ObjectCategories.tar.gz'
            if tar_gz_file.exists():
                print("Extracting tar.gz of object categories...")
                extract_tar_gz(str(tar_gz_file), str(self.dataset_root / 'caltech-101'))

    def __len__(self):
        return len(self.full_paths)

    def __getitem__(self, idx):
        img_path = self.full_paths[idx]

        # Open image and convert to RGB
        try:
            img = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            raise FileNotFoundError(f"Image not found: {img_path}")

        label = int(self.labels[idx])

        if self.transform:
            img = self.transform(img)

        return img, label
