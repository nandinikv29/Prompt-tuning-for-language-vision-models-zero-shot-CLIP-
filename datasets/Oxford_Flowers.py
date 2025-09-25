class OxfordFlowers(Dataset):
    def __init__(self, transform=None, download=True):
        self.transform = transform
        self.dataset_root = Path('./datas/oxford_flowers')
        self.dataset_root.mkdir(parents=True, exist_ok=True)

        self.tgz_path = self.dataset_root / '102flowers.tgz'
        self.json_path = self.dataset_root / 'split_zhou_OxfordFlowers.json'
        self.extract_path = self.dataset_root / 'jpg'

        if download:
            self._download_dataset()

        if not self.json_path.exists():
            raise FileNotFoundError(f"Split JSON not found: {self.json_path}")

        with open(self.json_path, 'r') as f:
            split_data = json.load(f)

        train_data = sorted(split_data['train'], key=lambda x: x[1])
        train_data = np.array(train_data)
        self.fnames = train_data[:, 0].tolist()
        self.labels = train_data[:, 1].astype(int)
        old_list = train_data[:, 2]
        self.classes = list(dict.fromkeys(old_list))
        self.full_paths = [self.extract_path / fname for fname in self.fnames]

    def _download_dataset(self):
        if not self.tgz_path.exists():
            print("Downloading Oxford Flowers dataset...")
            download_file(
                'https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz',
                str(self.tgz_path)
            )
        if not self.json_path.exists():
            print("Downloading split JSON...")
            download_file(
                'https://drive.usercontent.google.com/download?id=1Pp0sRXzZFZq15zVOzKjKBu4A9i01nozT&export=download&authuser=0',
                str(self.json_path)
            )
        if not self.extract_path.exists():
            print("Extracting Oxford Flowers tgz...")
            extract_tar_gz(str(self.tgz_path), str(self.dataset_root))
    
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
