import os
import json
import numpy as np
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from utils.file import download_file, extract_tar_gz
from config.config import get_config


class DTD(Dataset):
    def __init__(self, transform=None, download=True):
        self.transform = transform
        config = get_config()
        self.dataset_root = Path(config.get('dataset_root', './datas/')) / 'dtd'
        self.dataset_root.mkdir(parents=True, exist_ok=True)

        self.tar_path = self.dataset_root / 'dtd-r1.0.1.tar.gz'
        self.json_path = self.dataset_root / 'split_zhou_DescribableTextures.json'
        self.extract_path = self.dataset_root / 'dtd' / 'images'

        if download:
            self._download_dataset()

        if not self.json_path.exists():
            raise FileNotFoundError(f"Split JSON not found: {self.json_path}")

        with open(self.json_path, 'r') as f:
            split_data = json.load(f)

        train_data = np.array(split_data['train'])
        self.fnames = train_data[:, 0].tolist()
        self.labels = train_data[:, 1].astype(int)
        old_list = train_data[:, 2]
        self.classes = list(dict.fromkeys(old_list))
        self.full_paths = [self.extract_path / fname for fname in self.fnames]

    def _download_dataset(self):
        if not self.tar_path.exists():
            print("Downloading DTD dataset...")
            download_file(
                'https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz',
                str(self.tar_path)
            )
        if not self.json_path.exists():
            print("Downloading split JSON...")
            download_file(
                'https://drive.usercontent.google.com/download?id=1u3_QfB467jqHgNXC00UIzbLZRQCg2S7x&export=download&authuser=0',
                str(self.json_path)
            )
        if not self.extract_path.exists():
            print("Extracting DTD tar.gz...")
            extract_tar_gz(str(self.tar_path), str(self.dataset_root / 'dtd'))

    def __len__(self):
        return len(self.full_paths)

    def __getitem__(self, idx):
        img_path = self.full_paths[idx]
        try:
            img = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            raise FileNotFoundError(f"Image not found: {img_path}")

        label = int(self.labels[idx])
        if self.transform:
            img = self.transform(img)
        return img, label
