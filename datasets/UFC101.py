class UFC(Dataset):
    def __init__(self, transform=None, download=True):
        self.transform = transform
        config = get_config()
        self.dataset_root = Path(config.get('dataset_root', './datas/')) / 'ufc_101'
        self.dataset_root.mkdir(parents=True, exist_ok=True)

        self.zip_path = self.dataset_root / 'UCF-101-midframes.zip'
        self.json_path = self.dataset_root / 'split_zhou_UCF101.json'
        self.extract_path = self.dataset_root / 'UCF-101-midframes'

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
        if not self.zip_path.exists():
            print("Downloading UCF101 dataset zip...")
            download_file(
                'https://drive.usercontent.google.com/download?id=10Jqome3vtUA2keJkNanAiFpgbyC9Hc2O&export=download&authuser=0&confirm=t&uuid=e964f2b0-972a-42b3-8498-1305edb6abba&at=APZUnTXK33oPxMyW3KaM0uyzTNPD%3A1715431851790',
                str(self.zip_path)
            )
        if not self.json_path.exists():
            print("Downloading split JSON...")
            download_file(
                'https://drive.usercontent.google.com/download?id=1I0S0q91hJfsV9Gf4xDIjgDq4AqBNJb1y&export=download&authuser=0',
                str(self.json_path)
            )
        if not self.extract_path.exists():
            print("Extracting UCF101 zip...")
            unzip_file(str(self.zip_path), str(self.dataset_root))

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
