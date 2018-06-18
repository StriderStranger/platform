# platform
A general framework for deep learning

## 通用的功能
* 数据预处理
* 多线程打包数据
* 基本模型和损失函数
* 训练框架
* 测试框架
* 可定制的Log


## 文件目录
* DataProcess > packing.py     : 通用数据整理统一,会生成一个划分字典的pkl文件   (对于简单的数据可以跳过这步)
* DataProcess > preprocess.py  : 图像预处理类
* DataProcess > dataset.py     : 多线程数据打包功能
* ModelLoss   > model.py       : 定义模型
* MOdelLoss   > loss.py        : 定义复杂的损失函数
* Evaluation  > train.py       : 一个训练模板
* Evaluation  > eval.py        : 一个测试模板
* Evaluation  > analysis.py    : 分析测试结果常用的手段
* Tools       > utils.py       : 基本工具


## 需要定制的参数
* **路径参数**

|Args|Commit|Eg|
|--|--|--|
|from_dir|原始数据路径|-|
|data_dir|整理后数据路径|就是to_dir|
|ckpt_path|保存模型参数路径|-|
|log_path|保存训练记录路径|-|

* **预处理参数**

|Args|Commit|Eg|
|--|--|--|
|crop_prob|crop概率|0.5|
|crop_ratio|crop比率|0.7|
|resize_h_w|网络输入尺寸|(256,128)|
|scale|是否归一化|True|
|im_mean|白化参数|[0.486, 0.459, 0.408]|
|im_std|白化参数|[0.229, 0.224, 0.225]|
|batch_dims|图片风格|'NCHW'|

* **训练参数**

|Args|Commit|Eg|
|--|--|--|
|base_lr|初始学习率|2e-4|
|total_epochs|总循环次数|300|
|num_threads|加载线程数|2|
|batch_size|数据包大小|64|
|dataset_size|所有数据大小|15932|
|decay_ep|decay开始的ep次数|151|



## 需要定制的函数
* **packing.py**
1. 数据集的划分方法
2. 获取label信息并保存到字典

* **Dataset类**
1. __init__()函数中确定图片地址
2. 自定义的get_sample()函数

* **Log类**
1. 自定义需要记录的指标
2. 自定义打印指标的格式

* **model & loss**
1. 几乎全都要自定义