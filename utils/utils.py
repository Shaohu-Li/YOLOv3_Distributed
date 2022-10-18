
#=================================================#
#   用来存放一些常用的工具                          #
#=================================================#
import numpy as np
import torch
def cvtColor(image):
    """  
    将图像转换成RGB图像, 防止灰度图在预测时报错。
    代码仅仅支持RGB图像的预测, 所有其它类型的图像都会转化成RGB
    """
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image 
    else:
        image = image.convert('RGB')
        return image

def preprocess_input(image):
    """
    对图片进行归一化操作
    """
    image /= 255.0
    return image

def iou_width_height(boxes1, boxes2):
    """
    Parameters:
        boxes1 (tensor): width and height of the first bounding boxes
        boxes2 (tensor): width and height of the second bounding boxes
    Returns:
        tensor: Intersection over union of the corresponding boxes
    """
    intersection = torch.min(boxes1[..., 0], boxes2[..., 0]) * torch.min(
        boxes1[..., 1], boxes2[..., 1]
    )
    union = (
        boxes1[..., 0] * boxes1[..., 1] + boxes2[..., 0] * boxes2[..., 1] - intersection
    )
    return intersection / union

def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):
    """

    计算两个框之间的面积的交并比(iou)的函数

    Parameters:
        boxes_preds (tensor) : 网络预测出来的框的坐标 (BATCH_SIZE, 4)
        boxes_labels (tensor): 真实标签下的框的坐标   (BATCH_SIZE, 4)
        box_format (str)     : 选择自己的模式, midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)

    Returns:
        tensor: 返回检测框之间的 iou
    """

    if box_format == "midpoint":
        box1_x1 = boxes_preds[..., 0:1]  - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2]  - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1]  + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2]  + boxes_preds[..., 3:4] / 2
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    if box_format == "corners":
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    # 确保交集的框的宽高不会是负数
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)

def non_max_suppression(bboxes, iou_threshold, threshold, box_format="corners"):
    """
    对网络输出的所有检测框进行非极大值抑制

    Parameters:
        bboxes (list)        : 包含所有的网络输出的检测框,每一个检测框的形式为 [class_pred, prob_score, x1, y1, x2, y2]
        iou_threshold (float): 要计算 检测框和其他检测框 iou 的阈值(计算所有检测框的,并依次排序)
        threshold (float)    : 小于这个阈值的认为网络输出不正确
        box_format (str)     : "midpoint" or "corners" 检测框坐标所使用的形式

    Returns:
        list: bboxes after performing NMS given a specific IoU threshold
    """

    assert type(bboxes) == list

    # 筛出网络输出不正确的,并从大到小依次排序
    bboxes = [box for box in bboxes if box[1] > threshold]
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    bboxes_after_nms = []

    while bboxes:
        # 选当前列表中得分最高的
        chosen_box = bboxes.pop(0)

        # 注意:这是一个列表推导,别看错了

        #-------------------------------------------------#
        # 如果 下一个检测框 和当前出栈的得分最高的检测框,
        # 类别不一样, 或则 iou 小于给定阈值, 在列表中继续保留
        # 否则认为是同一物体,从列表中删除当前检测框
        #-------------------------------------------------#
        bboxes = [
            box
            for box in bboxes
            if box[0] != chosen_box[0]
            or intersection_over_union(
                torch.tensor(chosen_box[2:]),
                torch.tensor(box[2:]),
                box_format=box_format,
            )
            < iou_threshold
        ]

        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms