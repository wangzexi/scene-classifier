import os
from PIL import Image
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset, random_split
import functools

class MyDataset(Dataset):
    def listdir(dir):
        return sorted([os.path.join(dir, f) for f in os.listdir(dir) if not f.startswith('_')])

    def __init__(self, dataset_dir, transform):
        self.transform = transform

        # 从数据集目录读入图片文件列表，每个子目录内的图片赋予同一个整数作为类别标签
        # dataset_dir = './Dataset-06'
        # self.imgs: [(图片路径, 图片类别), ...]
        # 如：[('./Dataset-06\\06_street\\9934.jpg', 5), ...]
        self.imgs = []
        for cls, cls_dir in enumerate(MyDataset.listdir(dataset_dir)):
            self.imgs += [(img, cls) for img in MyDataset.listdir(cls_dir)]
        
    def __len__(self):
        return len(self.imgs)

    # 缓存transform后的图片结果，减少硬盘读取提高速度，但可能增加内存占用量
    @functools.cache
    def __getitem__(self, idx):
        if idx < 0 or idx >= self.__len__():
            raise IndexError

        img_path, cls = self.imgs[idx]
        tensor = self.transform(Image.open(img_path).convert('RGB'))

        return (tensor, cls)


class MyDataModule(pl.LightningDataModule):
    def __init__(self, transform, dataset_dir='./dataset/Dataset-06', batch_size=16):
        super().__init__()
        self.transform = transform
        self.dataset_dir = dataset_dir
        self.batch_size = batch_size

    def prepare_data(self):
        dataset = MyDataset(
            dataset_dir=self.dataset_dir,
            transform=self.transform
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
