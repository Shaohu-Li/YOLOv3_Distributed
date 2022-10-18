import random
import torch
import torch.nn as nn

from utils.utils import intersection_over_union


class YoloLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # 平均损失函数
        self.mse     = nn.MSELoss()
        # 包含了sigmoid的交叉熵损失函数
        self.bce     = nn.BCEWithLogitsLoss()
        # 单纯的交叉熵损失函数
        self.entropy = nn.CrossEntropyLoss()
        self.sigmoid = nn.Sigmoid()

        # 每部分损失前面的约束系数
        self.lambda_class = 1
        self.lambda_noobj = 10
        self.lambda_obj   = 1
        self.lambda_box   = 10

    def forward(self, predictions, target, anchors):

        # 因为你输入的预测是三个不同特征层的输入，所以，在计算的损失的时候，要么
        # 你给的目标是和预测是一一对应的，要么在这个地方对目标框进行特征层之间的变换，
        # 我们选择在 数据加载的时候进行处理，使得这里仅仅是计算损失的地方

        # 先检查真值的 obj and noobj，如果 target == -1，我们将忽略
        # 文中为 Iobj_i
        obj   = target[..., 0]      == 1 
        # 文中为 Inoobj_i
        noobj = target[..., 0]      == 0  


        # =================================== #
        #   没有物体的 检测框置信度的损失函数    #
        # =================================== #
        # print(predictions.shape, target.shape)
        no_object_loss = self.bce(
            (predictions[..., 0:1][noobj]), (target[..., 0:1][noobj]),
        )

        # ================================= #
        #   有物体的 检测框置信度的损失函数    #
        # ================================= #

        # 将其转换到可以和预测结果相乘的形状， 三种 anchors 和 宽高
        anchors = anchors.reshape(1, 3, 1, 1, 2)
        # 将中心坐标和宽高做处理，然后乘上每一个 anchor，得到真正的预测框
        # 再将所有检测框进行堆叠,后续和真值计算iou
        box_preds = torch.cat([self.sigmoid(predictions[..., 1:3]), torch.exp(predictions[..., 3:5]) * anchors], dim=-1)
        # 每个预测狂都和真值做 iou,得分最高的预测框 ,负责预测真实框
        ious = intersection_over_union(box_preds[obj], target[..., 1:5][obj]).detach()
        object_loss = self.mse(self.sigmoid(predictions[..., 0:1][obj]), ious * target[..., 0:1][obj])

        # ================================= #
        #   检测框的中心坐标和宽高损失函数     #
        # ================================= #

        # 将中心坐标和宽高做处理
        predictions[..., 1:3] = self.sigmoid(predictions[..., 1:3])  # x,y coordinates
        target[..., 3:5] = torch.log( 
            (1e-16 + target[..., 3:5] / anchors) 
        )  # width, height coordinates
        box_loss = self.mse(predictions[..., 1:5][obj], target[..., 1:5][obj])

        # ====================== #
        #   检测框的类别损失函数   #
        # ====================== #

        class_loss = self.entropy(
            (predictions[..., 5:][obj]), (target[..., 5][obj].long()),
        )

        #print("__________________________________")
        #print(self.lambda_box * box_loss)
        #print(self.lambda_obj * object_loss)
        #print(self.lambda_noobj * no_object_loss)
        #print(self.lambda_class * class_loss)
        #print("\n")

        # 返回总的损失函数
        return (
            self.lambda_box * box_loss
            + self.lambda_obj * object_loss
            + self.lambda_noobj * no_object_loss
            + self.lambda_class * class_loss
        )