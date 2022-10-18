
#=================================================#
#   放置一些网络的参数配置                          #
#=================================================#
import string
import torch

# 网络配置
model_config = [
    (32, 3, 1),
    (64, 3, 2),
    ["B", 1],
    (128, 3, 2),
    ["B", 2],
    (256, 3, 2),
    ["B", 8],
    (512, 3, 2),
    ["B", 8],
    (1024, 3, 2),
    ["B", 4],  # To this point is Darknet-53
    (512, 1, 1),
    (1024, 3, 1),
    "S",
    (256, 1, 1),
    "U",
    (256, 1, 1),
    (512, 3, 1),
    "S",
    (128, 1, 1),
    "U",
    (128, 1, 1),
    (256, 3, 1),
    "S",
]

#------------------------------------------------------------------#
#   GPU                     是否使用gpu进行训练，
#                           为 False,则接下来不能使用 dp 或者 dpp分布式训练
# 
#   DISTRIBUTED             是否使用单机多卡训练
#                           终端指令仅支持ubuntu。 CUDA_VISLBLE_DEVICES 用于在ubuntu终端指定显卡
#                           Windows 下系统默认使用DP模式调用所有显卡，不支持 DDP
# 
#   DP模式(简介：以一张卡为服务器卡，其余的客户端卡，服务器负责整理各个客户端的梯度和数据进行汇总，再将相同信息分发下去给每一个客户端卡，
#           所以DP受限与服务器卡的带宽，通信成本随着客户端卡的数量的增加而线程增长(服务端卡一次只能和一张客户端卡进行通信)
#           “”)
#                           设置    DISTRIBUTED = False
#                           终端模式使用命令：  CUDA_VISLBLE_DEVICES=0，1... python train.py
# 
#   DDP模式(简介：所有卡地位相同，每张卡都要逐渐收集到所有信息，但是因为每张卡可以接受信息(卡1)也可以发送信息(卡2)，
#           且只发送部分信息，可以并行执行只要有一张卡收集完信息，再将信息传递出去，基本上带宽不是问题，虽则gpu的增加，通信成本
#           与(N-1)/N 成正比)
#                           设置   DISTRIBUTED = True
#                           终端模式使用命令：CUDA_VISLBLE_DEVICES=0，1... python -m distributed.launch train.py
# 
#   SYNC_BN                 是否使用sync_bn, DDP模式下可以使用
#                           BN 在分布式训练的一种升级版，数据并行，每个gpu上的batchsize就会变小，BN效果就会差一点，
#                           是否使用sync_bn，将各显卡上分配的数据先汇总在一起，然后进行规范化处理。提升 BN 效果
#------------------------------------------------------------------#
GPU                     = True 
DISTRIBUTED             = True
SYNC_BN                 = True
FP16                    = True # if float(string(torch.__version__).split('+')[0]) >= 1.6 else False

#------------------------------------------------------------------#
# 相关阈值的设置
#   CONF_THRESHOLD           置信度阈值
#   MAP_IOU_THRESH
#   NMS_IOU_THRESH           非极大值抑制的时候，低于此阈值不再进行相似查找           
#------------------------------------------------------------------#
CONF_THRESHOLD          = 0.05
MAP_IOU_THRESH          = 0.5
NMS_IOU_THRESH          = 0.45

#------------------------------------------------------------------#
#   IMAGE_SIZE              输入图片的大小，网络接受图片的大小
#   IMG_GRID                网络三个输出特征层大小
#                           根据输入图片的不一样，输出特征层也会相应调整
#
#   IMAGE_TRANS_SCALE       数据增强时候放大的尺寸比例(Dataset_loader中)
# 
#   SHUFFLR                 加载数据数据的时候，每个批次内的数据是否打乱
#   PIN_MEMORY              当计算机的内存充足的时候，可以设置pin_memory=True。
#                           意味着生成的Tensor数据最开始是属于内存中的锁页内存，
#                           这样将内存的Tensor转义到GPU的显存就会更快一些。
#------------------------------------------------------------------#
IMAGE_SIZE              = 416
IMG_GRID                = [IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8] # 13 26 52
IMAGE_TRANS_SCALE       = 1.1
SHUFFLR                 = True
PIN_MEMORY              = True

#------------------------------------------------------------------#
#   ANCHORS                 先验框的大小
#------------------------------------------------------------------#
ANCHORS                 = [
                            [(116, 90), (156, 198), (373, 326)],
                            [(30, 61), (62, 45),   (59, 119)],
                            [(10, 13), (16, 30),   (33, 23)],
                        ]


#------------------------------------------------------------------#
#   NUM_WORKERS             预加载图片使用的线程数
#   BATCH_SIZE              每个 batch 读取的薯片的数量
#------------------------------------------------------------------#
NUM_WORKERS             = 6
BATCH_SIZE              = 64

#------------------------------------------------------------------#
# 有关epoch的相关配置
#   NUM_EPOCHS             总共训练多少个 epoch
#   WEIGHT_SAVE_PERIOD     多少个 epoch 保存一次权值
#------------------------------------------------------------------#
NUM_EPOCHS              = 50
WEIGHT_SAVE_PERIOD      = 5


#------------------------------------------------------------------#
# 学习率相关的配置
# 
#   OPTIMIZER_TYPE              使用到的优化器种类，可选的有adam、sgd
#                               当使用Adam优化器时建议设置  Init_lr=1e-3
#                               当使用SGD优化器时建议设置   Init_lr=1e-2
# 
#   MOMENTUM                    优化器内部使用到的momentum参数
# 
#   WEIGHT_DECAY                权值衰减，可防止过拟合
#                               adam会导致weight_decay错误，使用adam时建议设置为0。
# 
#   INIT_LEARNING_RATE         模型的最大学习率
#   MIN_LEARNING_RATE          模型的最小学习率，默认为最大学习率的0.01
#------------------------------------------------------------------#
OPTIMIZER_TYPE          = "adam"
MOMENTUM                = 0.937
WEIGHT_DECAY            = 1e-4
INIT_LEARNING_RATE      = 1e-3
MIN_LEARNING_RATE       = INIT_LEARNING_RATE * 0.01

#------------------------------------------------------------------#
#   LEARNING_RATE_DECAY_TYPE   使用到的学习率下降方式，可选的有step、cos
#------------------------------------------------------------------#
LEARNING_RATE_DECAY_TYPE = "COS"

#------------------------------------------------------------------#
#权值保存和加载相关配置
#   save_dir            权值与日志文件保存的文件夹
#   SAVE_MODEL          是否保存权值
#   LOAD_MODEL          是否从保存的文件中加载权值
#                       为 True,则下面需要填上自己保存的权值
#   LOAD_WEIGHT_NAME    加载权值的文件名
#------------------------------------------------------------------#
SAVE_DIR                = "logs"
SAVE_MODEL              = True
LOAD_MODEL              = True
LOAD_WEIGHT_NAME        =  "model_data/weights/" + "best_epoch_weights.pth"

#------------------------------------------------------------------#
# 数据集相关配置
#   NUM_CLASSES         当前训练集的类别
#   DATASET             自己存放数据集的地方
#   IMG_DIR             训练需要用到的图片的路径
#   TRAIN_LABEL_DIR     训练集 annotation txt 文件的路径
#   VAL_LABEL_DIR       验证集 annotation txt 文件的路径
#------------------------------------------------------------------#
NUM_CLASSES             = 80
DATASET                 = '/home/adr/datasets/vision/coco/14+17'
IMG_DIR                 = DATASET + "/images/"
TRAIN_LABEL_DIR         = DATASET + "/Annotations/"
VAL_LABEL_DIR           = DATASET + "/Annotations/"

#------------------------------------------------------------------#
#   PASCAL_CLASSES      voc 数据的类别
#------------------------------------------------------------------#
PASCAL_CLASSES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor"
]

#------------------------------------------------------------------#
#   COCO_LABELS      coco   数据集的标签
#------------------------------------------------------------------#
COCO_LABELS = ['person',
 'bicycle',
 'car',
 'motorcycle',
 'airplane',
 'bus',
 'train',
 'truck',
 'boat',
 'traffic light',
 'fire hydrant',
 'stop sign',
 'parking meter',
 'bench',
 'bird',
 'cat',
 'dog',
 'horse',
 'sheep',
 'cow',
 'elephant',
 'bear',
 'zebra',
 'giraffe',
 'backpack',
 'umbrella',
 'handbag',
 'tie',
 'suitcase',
 'frisbee',
 'skis',
 'snowboard',
 'sports ball',
 'kite',
 'baseball bat',
 'baseball glove',
 'skateboard',
 'surfboard',
 'tennis racket',
 'bottle',
 'wine glass',
 'cup',
 'fork',
 'knife',
 'spoon',
 'bowl',
 'banana',
 'apple',
 'sandwich',
 'orange',
 'broccoli',
 'carrot',
 'hot dog',
 'pizza',
 'donut',
 'cake',
 'chair',
 'couch',
 'potted plant',
 'bed',
 'dining table',
 'toilet',
 'tv',
 'laptop',
 'mouse',
 'remote',
 'keyboard',
 'cell phone',
 'microwave',
 'oven',
 'toaster',
 'sink',
 'refrigerator',
 'book',
 'clock',
 'vase',
 'scissors',
 'teddy bear',
 'hair drier',
 'toothbrush'
]

if __name__ == "__main__":
    print("config")