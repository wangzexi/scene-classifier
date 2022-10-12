## 场景分类器

### 整体流程

图片 --- VIT ---> 特征向量[1000] --- MyModel ---> 分类向量[6]

### 结果

学习率0.00001，权重随机初始化运行200轮，在验证集上场景分类准确度能达到94.39%。

![result](result.png)

### 运行环境

```
# 创建python3.9虚拟环境
conda create --name py39 python=3.9
conda activate py39 

# Windows下安装pytorch以及cudatookit 11.3
# 需要支持nvidia显卡
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

pip install pytorch-lightning
pip install einops
pip install matplotlib
```

### 数据集

数据集位于`Dataset-06`文件夹中，通过`dataset.py`载入。
`img_cache.pkl`是通过[预训练的VIT网络](https://rwightman.github.io/pytorch-image-models/models/vision-transformer/)计算出的图片特征，用于加速训练。

数据集被随机6:2:2分割为训练集、验证集和测试集。

### 训练

```
# 训练模型
python train.py

# 启动训练图表面板
tensorboard --logdir .
```

### 测试

如果需要准确度（Accuracy）外的其它指标，可以参考：https://torchmetrics.readthedocs.io/en/stable/all-metrics.html
所有Classification类型的指标都可以添加。

```
## 测试模型
python test.py
```
