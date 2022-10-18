# 代码支持在cuda下进行训练，可以训练自己的数据集

对应的博客说明链接：https://blog.csdn.net/To_be_little/article/details/126130928?spm=1001.2014.3001.5501

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
## 4、运行 train.py 进行训练即可
&emsp;&emsp; 1、出现 gpu 内存不足，可以调整batch_size的大小，也在config.py中

&emsp;&emsp; 2、出现 albumentations的版本问题，可以提升一下版本

## 5、参考：
https://github.com/bubbliiiing/yolo3-pytorch

https://youtu.be/YDkjWEN8jNA

"如果有任何的问题，请给位大佬批评指正,后续将加上预测和测试的代码，以及单机多卡训练的代码"