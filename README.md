## 场景分类器

### 整体流程

图片 --- ViT ---> 特征向量[768] --- MyModel ---> 分类向量[6]

### 结果

学习率0.000001，权重随机初始化运行200轮，在测试集上场景分类准确度能达到94.39%。

![result](pics/result.png)

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
pip install torchmetrics
pip install tqdm
pip install tensorboard
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
# 测试模型
python test.py
```

### HTTP服务

使用以下命令启动HTTP服务器，默认监听在`localhost:3000`，使用HTTP POST上传任意图片即可获得分类结果，正确返回`0~5`，错误返回`-1`。

```
# 启动服务
python server.py

# 上传图片获得结果
curl --request POST --data-binary @pics/1.jpg localhost:3000
```
