import os
import pickle
from PIL import Image
from tqdm import tqdm
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset, random_split


class MyDataset(Dataset):
    def listdir(dir):
        return sorted([os.path.join(dir, f) for f in os.listdir(dir) if not f.startswith('_')])

    def load_cache(self):
        try:
            with open('img_cache.pkl', "rb") as f:
                self.img_cache = pickle.load(f)
        except:
            self.img_cache = {}

    def save_cache(self):
        with open('img_cache.pkl', "wb") as f:
            pickle.dump(self.img_cache, f)

    def __init__(self, dataset_dir, extractor):
        # dataset_dir = './Dataset-06'
        # self.imgs
        # [图片路径, 图片类别]
        # 如：['./Dataset-06\\06_street\\9934.jpg', 5]
        self.imgs = []
        for cls, cls_dir in enumerate(MyDataset.listdir(dataset_dir)):
            self.imgs += [[img, cls] for img in MyDataset.listdir(cls_dir)]

        # extractor: 预训练的图片特征提取器
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.extractor = extractor.to(device)

        # 更新数据集所有图片的特征缓存
        # 所有图片先过一遍预训练的网络，即：[1, 3, 224, 224] -> [1, 768]
        # 生成图片特征缓存哈希表 img_cache，图片文件名 => [1, 768]
        self.load_cache()
        with torch.no_grad():
            for img_path, cls in tqdm(self.imgs):
                if img_path in self.img_cache:
                    continue
                img = Image.open(img_path).convert('RGB')
                tensor = self.extractor.transform(img).unsqueeze(0).to(device)  # transform and add batch dimension
                self.img_cache[img_path] = self.extractor(tensor).detach().cpu().reshape(-1)
        self.save_cache()

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.__len__():
            raise IndexError

        img_path, cls = self.imgs[idx]
        feature = self.img_cache[img_path]

        return (feature, cls)


class MyDataModule(pl.LightningDataModule):
    def __init__(self, extractor, batch_size=128):
        super().__init__()
        self.extractor = extractor
        self.batch_size = batch_size

    def prepare_data(self):
        dataset = MyDataset(
            dataset_dir='./Dataset-06',
            extractor=self.extractor
        )

        # 训练集:验证集:测试集 = 6:2:2
        total_length = len(dataset)
        train_length = int(total_length * 0.6)
        val_length = int(total_length * 0.2)
        test_length = total_length - train_length - val_length

        self.train, self.val, self.test = random_split(dataset, [train_length, val_length, test_length])

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train,
            batch_size=self.batch_size,
            shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val,
            batch_size=self.batch_size,
            shuffle=False
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test,
            batch_size=self.batch_size,
            shuffle=False
        )
