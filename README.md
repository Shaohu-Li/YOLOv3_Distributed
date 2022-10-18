# 代码支持在cuda下进行训练，可以训练自己的数据集

对应的博客说明链接：https://blog.csdn.net/To_be_little/article/details/126130928?spm=1001.2014.3001.5501

​		代码在之前的版本上，实现了 pytorch 上 分布式 多gpu 训练，选用了目前 pytorch 推荐的 DDP 分布式模式。

## 1、首先安装相关的依赖包，根据requirements.txt
&emsp;&emsp;安装包的版本不一定需要一模一样，存在相应安装包就行，当运行的时候报错，再更改版本，没必要将自己搭配好的环境进行大修大改

## 2、更改训练集路径
&emsp;&emsp;如果要使用coco，或者 voc 进行训练的话，需要使用 coco_annotation.py 或者 voc_annotation.py 生成自己的可用于训练的txt文件
&emsp;&emsp;coco_annotation.py中修改下面两项：
```
# 自己的coco数据的路径
datapath            = "/home/adr/datasets/vision/coco/14+17"
classes_path        = datapath + "\\ImageSets\\SelfMain\\voc_classes.txt"
```
&emsp;&emsp;voc_annotation.py中修改下面两项：
```
# 自己的voc数据的路径
datapath            = "E:\\数据集\\VOC\\VOCdevkit\\VOC07+12"
classes_path        = datapath + "\\ImageSets\\SelfMain\\voc_classes.txt"
```
&emsp;&emsp;如果要使用自己定义的数据和标签的文件的话，需要修改 selfdata_annotaion.py 中的
```
path = "自己的数据集根目录"
    dataset_divide_annotation(path)
```
## 3、修改配置 在 config/config.py 中，几乎所有的文件的配置文件都在其中，需要修改下面的内容：
&emsp;&emsp;分别对应着上面你得到的训练文件
```
NUM_CLASSES             = 80
DATASET                 = '/home/adr/datasets/vision/coco/14+17'
IMG_DIR                 = DATASET + "/images/"
TRAIN_LABEL_DIR         = DATASET + "/Annotations/"
VAL_LABEL_DIR           = DATASET + "/Annotations/"
```
## 4、运行 begin_train.sh 进行训练即可

&emsp;&emsp; 1、出现 gpu 内存不足，可以调整batch_size的大小，也在config.py中

&emsp;&emsp; 2、出现 albumentations的版本问题，可以提升一下版本

​		 3、我是在 Ubuntu 上在 直接运行 .sh 是可以运行的。如果不行，将 .sh 中的运行代码拷贝到终端中进行运行即可。

```python
# 先激活你的终端环境
torchrun --nnodes=1 --standalone --nproc_per_node=2 train.py
# 或者
CUDA_VISLBLE_DEVICES=0,1 python -m torch.distributed.launch  --nproc_per_node=2 train.py
```

实践证明，使用 pytorch 最新的 torchrun 命令好像运行的更快一点，不知道是不是错觉，具体的上面两个命令的解释可以查看pytorch的[官方文档](https://pytorch.org/docs/stable/elastic/run.html)；

## 5、参考：

https://github.com/bubbliiiing/yolo3-pytorch

https://youtu.be/YDkjWEN8jNA

"如果有任何的问题，请给位大佬批评指正,后续将加上预测和测试的代码，以及单机多卡训练的代码"