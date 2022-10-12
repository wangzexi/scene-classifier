import os
import pickle
from PIL import Image
from tqdm import tqdm
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset, Subset, ConcatDataset, random_split
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

# 预训练的图片模型
model = timm.create_model('vit_base_patch16_224', pretrained=True).eval()

# 加速训练的图片特征缓存
# 所有图片先过一遍预训练的网络，即：[1, 3, 224, 224] -> [1, 1000]
# 生成图片特征缓存哈希表 img_cache，图片文件名 => [1, 1000]
cache_file = 'img_cache.pkl'


def read_cache():
    try:
        with open(cache_file, "rb") as f:   # Unpickling
            return pickle.load(f)
    except:
        return {}


def save_cache():
    with open(cache_file, "wb") as f:  # Pickling
        pickle.dump(img_cache, f)


img_cache = read_cache()

class MyDataset(Dataset):
    def listdir(dir):
        return sorted([os.path.join(dir, f) for f in os.listdir(dir) if not f.startswith('_')])

    def __init__(self, dataset_dir, model):
        # dataset_dir = './Dataset-06'
        # self.imgs
        # [图片路径, 图片类别]
        # 如：['./Dataset-06\\06_street\\9934.jpg', 5]
        self.imgs = []
        for cls, cls_dir in enumerate(MyDataset.listdir(dataset_dir)):
            self.imgs += [[img, cls] for img in MyDataset.listdir(cls_dir)]

        # model: timm.create_model('vit_base_patch16_224', pretrained=True)
        self.model = model
        self.transform = create_transform(**resolve_data_config({}, model=self.model))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.__len__():
            raise IndexError

        img_path, cls = self.imgs[idx]

        if img_path not in img_cache:
            # transform
            img = Image.open(img_path).convert('RGB')
            tensor = self.transform(img).unsqueeze(0)  # transform and add batch dimension
            with torch.no_grad():
                img_cache[img_path] = self.model(tensor).detach().cpu().numpy().reshape(-1)
            save_cache()

        feature = img_cache[img_path]

        return (
            feature,
            cls
        )


class MyDataModule(pl.LightningDataModule):
    def __init__(self, model=model, batch_size=128):
        super().__init__()
        self.model = model
        self.batch_size = batch_size

    def prepare_data(self):
        # model
        dataset = MyDataset(
            dataset_dir='./Dataset-06',
            model=self.model
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


if __name__ == '__main__':
    # model = timm.create_model('vit_base_patch16_224', pretrained=True).eval()

    # d = MyDataset(
    #     dataset_dir='./Dataset-06',
    #     model=model
    # )

    # feat, cls = d[0]

    # # print(im)
    # # print(im.shape)

    # dl = MyDataLoader(data_config=resolve_data_config({}, model=model))
    pass
