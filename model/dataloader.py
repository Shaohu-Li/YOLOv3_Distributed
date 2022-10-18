
#=================================================#
#   用来存放网络训练数据的加载的一些函数             #
#=================================================#

import cv2
import math
import torch
import numpy as np
from PIL import Image
from config.config import IMAGE_SIZE, IMAGE_TRANS_SCALE, IMG_GRID
import albumentations as A
from albumentations.pytorch import ToTensorV2
from utils.utils import iou_width_height as iou_wh, preprocess_input, cvtColor
from torch.utils.data.dataset import Dataset
"""
一般来说PyTorch中深度学习训练的流程是这样的:
    1. 创建Dateset
        dataset = self_def_dataset 
    2. Dataset传递给DataLoader
         dataloader = torch.utils.data.DataLoader(dataset,batch_size=64,shuffle=False,num_workers=8,.....)
    3. DataLoader迭代产生训练数据提供给模型
        for i in range(epoch):
        for index,(img,label) in enumerate(dataloader):
            pass
"""

# 继承 pytorch 的 Dataset，来写自己的数据的加载类
class Dataset_loader(Dataset):
    def __init__(self, annotation_lines, box_mode, input_shape, anchors,  train, diff_out = IMG_GRID, transform=None):
        """自定义自己的数据集类

        Arguments:
            annotation_lines        -- 要加载的 annotation 文件中的每一行的的句柄
            box_mode                -- 对应对检测框进行不同的处理
                                    == 0 : 代表输出的检测框的格式为 (x_min, y_min, x_max,y_max)
                                    == 1 : 代表输出的检测框的格式为 (x_cent, y_cent, w, h)
            input_shape             -- 网络期望得到的图片的大小
            anchors                 -- 聚类获得的先验框
            train                   -- 是否是训练模式
            diff_out                -- 三种不同的特征层的输出大小
                                        : 依据输入图片的不同可以主动调整
            transform               -- 对图片进行的一些的增强操作
        """
        super().__init__()
        self.annotations_lines      = annotation_lines
        self.annotations_lenght     = len(annotation_lines) 
        self.box_mode               = box_mode 
        self.input_shape            = input_shape
        self.anchors                = torch.Tensor(anchors[0] + anchors[1] + anchors[2])
        self.train                  = train
        self.diff_out               = diff_out
        self.transform              = transform

        self.num_anchors            = self.anchors.shape[0]
        self.num_anchors_per_scale  = self.num_anchors // 3
        self.ignore_iou_thresh      = 0.5
        self.length                 = len(self.annotations_lines)

    def __len__(self):
        """
        返回当前数据的总长度
        """
        return  self.length
    
    def __getitem__(self, index):
        """返回一个组对应的图片, 检测框和类别的真值，

        Arguments:
            index            -- 选取照片的开始的索引值 
        """
        
        # -----------------------------------------------------#
        # 进行检测框和图片的读取操作
        # -----------------------------------------------------#
        index       = index % self.length
        line        = self.annotations_lines[index].split()
        # 确保读取到的图片都是 RGB, 并且已经进行了归一化
        image       = np.array(Image.open(line[0]).convert("RGB"))
        bboxes      = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])

        # 因为下面要使用的为coco数据集格式，在这里需要转化一下

        # 是否使用albumentations定义的一些数据增强的操作
        if self.transform:
            augmentations = self.transform(image=image, bboxes=bboxes)
            image = augmentations["image"]
            bboxes = augmentations["bboxes"]

        # -----------------------------------------------------#
        # 针对 yolo 来说，会有三个不同大小的特征层的输出，
        # 在任意一个特征层是输出，经过放缩之后能够对应上原图的检测框
        # 我们都可以认为这个目标被检测出来了，
        # 所以我们需要将 真实框 转换到三种尺寸下
        # -----------------------------------------------------#
        
        img_h                   = self.input_shape
        img_w                   = self.input_shape
        # 每个特征层的每个像素点负责又 3 中 anchors， 像素的个数为 out_size * out_size ， 一个anchor需要预测类别和坐标
        #                            3         *         out_size * out_size       *  (1 + 4)
        targets                 = [torch.zeros( ( self.num_anchors // 3, out_size, out_size, 6) ) for out_size in self.diff_out]

        for box in bboxes:
            # 计算当前 标定框 和 9 个先验框的iou
            iou_anchors         = iou_wh(torch.tensor(box[2:4]), self.anchors)
            # 按照 iou 的顺序进行从大到小进行排序，得到排序顺序
            anchor_idxs         = iou_anchors.argsort( dim=0, descending=True )
            x, y, w, h, label   = box 
            # 每一个像素都要匹配三个 anchor
            has_anchors         = [False] * 3
            
            # 需要将 一个 标定框映射到 三种 不同的特征层
            for anchor_idx in anchor_idxs:
                # 先找到当前标定框 位于哪一个 anchor 层 或者说 特征层 -> 0, 1, 2
                scale_idx       = torch.div(anchor_idx, self.num_anchors_per_scale, rounding_mode='trunc') # anchor_idx // self.num_anchors_per_scale 
                # 再找到 位于当前anchors(3个)中的哪一个位置
                anchor_on_scale = anchor_idx % self.num_anchors_per_scale
                # 找到 对应特征层
                feat_scale      = self.diff_out[scale_idx]
                # 计算位于当前的像素点的位置
                # print(x / img_w, y / img_h)
                i, j            = int(feat_scale * (x / img_w)), int(feat_scale * (y / img_h))
                # print(i, j)
                anchor_taken    = targets[scale_idx][anchor_on_scale, i, j, 0]

                if not anchor_taken and not has_anchors[scale_idx]:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = 1
                    has_anchors[scale_idx] = True
                    # 计算中心坐标的偏移量，和相对于 anchor 的偏移量
                    x_cell, y_cell, w_cell, h_cell = feat_scale * x / img_w - i, feat_scale * y  /img_h - j, feat_scale * w / img_w, feat_scale * h / img_h
                    box_coordinates = torch.tensor(
                        [x_cell, y_cell, w_cell, h_cell]
                    )
                    targets[scale_idx][anchor_on_scale, i, j, 1:5] = box_coordinates
                    targets[scale_idx][anchor_on_scale, i, j, 5] = int(label)

                elif not anchor_taken and iou_anchors[anchor_idx] > self.ignore_iou_thresh:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = -1  # ignore prediction

        return image, targets


train_transforms = A.Compose(
    [
        # 保持图片的比例进行图片的放大
        A.LongestMaxSize(max_size=IMAGE_SIZE),
        # 不保证图片比例进行放大，需要的时候会对图片添加 0 
        A.PadIfNeeded(
            min_height=IMAGE_SIZE, min_width=IMAGE_SIZE, border_mode=cv2.BORDER_CONSTANT
        ),
        # 进行正则化
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255,),
        # 转变为 tensor
        ToTensorV2(),
    ],
    # 对检测框进行调整
    bbox_params=A.BboxParams(format="coco", label_fields=[],check_each_transform=False),
) 

val_transforms = A.Compose(
    [
        # 保持图片的比例进行图片的放大
        A.LongestMaxSize(max_size=IMAGE_SIZE),
        # 不保证图片比例进行放大，需要的时候会对图片添加 0 
        A.PadIfNeeded(
            min_height=IMAGE_SIZE, min_width=IMAGE_SIZE, border_mode=cv2.BORDER_CONSTANT
        ),
        # 进行正则化
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255,),
        # 转变为 tensor
        ToTensorV2(),
    ],
    # 对检测框进行调整
    bbox_params=A.BboxParams(format="coco", min_visibility=0, label_fields=[], check_each_transform=False),
)

# DataLoader中collate_fn使用
def yolo_dataset_collate(batch):
    images = []
    bboxes = []
    for img, box in batch:
        images.append(img)
        bboxes.append(box)
    images = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    bboxes = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in bboxes]
    return images, bboxes

train_transforms_v2 = A.Compose(
    [
        # 保持图片的比例进行图片的放大
        A.LongestMaxSize(max_size=int(IMAGE_SIZE * IMAGE_TRANS_SCALE)),
        # 不保证图片比例进行放大，需要的时候会对图片添加灰色边
        A.PadIfNeeded(
            min_height=int(IMAGE_SIZE * IMAGE_TRANS_SCALE),
            min_width=int(IMAGE_SIZE * IMAGE_TRANS_SCALE),
            border_mode=cv2.BORDER_CONSTANT,
        ),
        # 对图片进行随机的剪裁
        A.RandomCrop(width=IMAGE_SIZE, height=IMAGE_SIZE),
        # 随机改变图像的亮度、对比度和饱和度。
        A.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.6, p=0.4),
        # 下面的操作选择其中的一个
        A.OneOf(
            [
                # 随机应用仿射变换：平移、缩放和旋转输入。
                A.ShiftScaleRotate(
                    rotate_limit=20, p=0.5, border_mode=cv2.BORDER_CONSTANT
                ),
                # A.IAAAffine(shear=15, p=0.5, mode="constant"),
            ],
            p=1.0,
        ),
        # 围绕 y 轴水平翻转输入。
        A.HorizontalFlip(p=0.5),
        # 使用随机大小的内核模糊输入图像。
        A.Blur(p=0.1),
        # 将对比度受限的自适应直方图均衡应用于输入图像。就是让图片的色彩分布变得均匀
        A.CLAHE(p=0.1),
        # 减少每个颜色通道的位数。
        A.Posterize(p=0.1),
        # 将输入的 RGB 图像转换为灰度。 如果生成图像的平均像素值大于 127，则反转生成的灰度图像。
        A.ToGray(p=0.1),
        # 随机重新排列输入 RGB 图像的通道。
        A.ChannelShuffle(p=0.05),
        # 对图片进行正则化或则说归一化
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255,),
        # 将变量加载为 tensor，并且 numpy HWC 图像被转换为 pytorch CHW 张量。 如果图像是 HW 格式（灰度图像），它将被转换为 pytorch HW tensor。
        ToTensorV2(),
    ],
    # 转换 box 的形式
    bbox_params=A.BboxParams(format="coco", min_visibility=0.4, label_fields=[],),
)

"""
Dataset内容学习链接: https://blog.csdn.net/weixin_41560402/article/details/108121344

在pytorch中, 提供了一种十分方便的数据读取机制, 即,使用torch.utils.data.Dataset与torch.utils.data.DataLoader组合得到数据迭代器。
在每次训练时, 利用这个迭代器输出每一个batch数据, 并能在输出时对数据进行相应的预处理或数据增强等操作。

torch.utils.data.Dataset
    1、代表自定义数据集方法的类, 用户可以通过继承该类来自定义自己的数据集类,
    2、在继承时要求用户重载__len__()和__getitem__()这两个魔法方法。

torch.utils.data.DataLoader
作用：
    1、DataLoader将Dataset对象或自定义数据类的对象封装成一个迭代器;
    2、这个迭代器可以迭代输出Dataset的内容;
    3、同时可以实现多进程、shuffle、不同采样策略, 数据校对等等处理过程。

"""