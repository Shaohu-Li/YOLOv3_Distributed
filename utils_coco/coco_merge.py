
import json
import os
from collections import defaultdict

def change_cat(cat):
    """
    改变原有的coco数据集的标签类别, 使其连续
    """
    if cat >= 1 and cat <= 11:
        cat = cat - 1
    elif cat >= 13 and cat <= 25:
        cat = cat - 2
    elif cat >= 27 and cat <= 28:
        cat = cat - 3
    elif cat >= 31 and cat <= 44:
        cat = cat - 5
    elif cat >= 46 and cat <= 65:
        cat = cat - 6
    elif cat == 67:
        cat = cat - 7
    elif cat == 70:
        cat = cat - 9
    elif cat >= 72 and cat <= 82:
        cat = cat - 10
    elif cat >= 84 and cat <= 90:
        cat = cat - 11
    return cat

def combine(path_14="", path_17="", path_1417="", mode = 1417):
    """
    path_14     : coco2014 的路径
    path_17     : coco2017 的路径
    path_14_17  : 自己的 coco14+17 的路径
    mode        : 
                == 14 代表只对 coco2014 进行处理                        : 只需要填入参数 path_14 和 切换 mode
                == 17 代表只对 coco2017 进行处理                        : 只需要填入参数 path_14 和 切换 mode
                == 1417 代表对 coco2014 + 2017 进行处理，生成 coco14+17  : 只需要填入参数 path_1417 和 切换 mode

    无论是哪一种模式, 都会再当前路径的跟目录下面生成 train.txt 和 val.txt,
    """
    # 建立两个用来装载 train 和 val 中 路径 + 坐标 + 类别的容器
    name_box_id_train = defaultdict(list)
    name_box_id_val = defaultdict(list)
    if mode == 14:
        path = path_14
        if not os.path.exists(os.path.join( path, "Annotations/instances_train2014.json" )):
            print("当前模式和输入的参数不匹配, 请重新输入参数: coco14的正确路径 + mode = 14")
            return -1
    elif mode == 17:
        path = path_17
        if not os.path.exists(os.path.join( path, "Annotations/instances_train2017.json" )):
            print("当前模式和输入的参数不匹配, 请重新输入参数: coco17的正确路径 + mode = 17")
            return -1
    elif mode == 1417:
        path = path_1417
        if not os.path.exists(os.path.join( path, "Annotations/instances_train2014.json" )) and \
           not os.path.exists(os.path.join( path, "Annotations/instances_train2017.json" )) :
            print("当前模式和输入的参数不匹配, 请重新输入参数: coco14 和 coco 的正确路径 + mode = 14")
            return -1

    if mode ==14 or mode == 1417:
        annotations_train_path_14              = os.path.join( path, "Annotations/instances_train2014.json" ) 
        annotations_val_path_14                = os.path.join( path, "Annotations/instances_val2014.json" )

        data_annotations_train_path_14         = json.load(open(annotations_train_path_14, encoding='utf-8'))['annotations']
        data_annotations_val_path_14           = json.load(open(annotations_val_path_14, encoding='utf-8'))['annotations']
        for ant in data_annotations_train_path_14:
            id = ant['image_id']
            name = os.path.join( os.path.abspath(path), 'train%d/COCO_train2014_%012d.jpg' % (mode, id) )
            cat = change_cat(ant['category_id'])

            # 将 坐标和类别添加进去
            name_box_id_train[name].append([ant['bbox'], cat])

        for ant in data_annotations_val_path_14:
            id = ant['image_id']
            name = os.path.join( os.path.abspath(path), 'val%d/COCO_val2014_%012d.jpg' % (mode, id) )
            cat = change_cat(ant['category_id'])

            # 建 坐标和类别添加进去
            name_box_id_val[name].append([ant['bbox'], cat])
    len14_train = len(name_box_id_train)
    len14_val   = len(name_box_id_val) 
    print("In coco 2014, the train  and val image is : {}-----{}".format( len14_train, len14_val ))
    
    if mode ==17 or mode == 1417:
        annotations_train_path_17              = os.path.join( path, "Annotations/instances_train2017.json" ) 
        annotations_val_path_17                = os.path.join( path, "Annotations/instances_val2017.json" )

        data_annotations_train_path_17         = json.load( open(annotations_train_path_17, encoding='utf-8') )['annotations']
        data_annotations_val_path_17           = json.load( open(annotations_val_path_17, encoding='utf-8') )['annotations']

        for ant in data_annotations_train_path_17:
            id = ant['image_id']
            name = os.path.join( os.path.abspath(path), 'train%d/%012d.jpg' % (mode, id) )
            cat = change_cat(ant['category_id'])
        
            # 将 坐标和类别添加进去
            name_box_id_train[name].append([ant['bbox'], cat])

        for ant in data_annotations_val_path_17:
            id = ant['image_id']
            name = os.path.join( os.path.abspath(path), 'val%d/%012d.jpg' % (mode, id) )
            cat = change_cat(ant['category_id'])
            
            # 将 坐标和类别添加进去
            name_box_id_val[name].append([ant['bbox'], cat])

    total_train = len(name_box_id_train)
    total_val   = len(name_box_id_val)
    print("In coco 2017, the train  and val image is : {}-----{}".format( total_train - len14_train, total_val - len14_val ))

    print("The train and val image total is : {}-----{}".format( total_train, total_val ))

    # 再统一放入训练的 train.txt 中 
    f = open(os.path.join(path, 'Annotations/train.txt'), 'w')
    for key in name_box_id_train.keys():
        # print(key)
        f.write(key)
        box_infos = name_box_id_train[key]
        for info in box_infos:
            x_min = int(info[0][0])
            y_min = int(info[0][1])
            width =  int(info[0][2])
            height =  int(info[0][3])
            if width == 0:
                width = width + 1
            if height == 0:
                height = height + 1

            # x_max = x_min + int(info[0][2])
            # y_max = y_min + int(info[0][3])

            # 最后一个为 类别
            box_info = " %d,%d,%d,%d,%d" % (
                x_min, y_min, width, height, int(info[1]))
            f.write(box_info)
        f.write('\n')
    f.close()

    # 再统一放入训练的 train.txt 中 
    f = open(os.path.join(path, 'Annotations/val.txt'), 'w')
    for key in name_box_id_val.keys():
        # print(key)
        f.write(key)
        box_infos = name_box_id_val[key]
        for info in box_infos:
            x_min = int(info[0][0])
            y_min = int(info[0][1])
            width =  int(info[0][2])
            height =  int(info[0][3])
            if width == 0:
                width = width + 1
            if height == 0:
                height = height + 1
            # x_max = x_min + int(info[0][2])
            # y_max = y_min + int(info[0][3])

            # 最后一个为 类别
            box_info = " %d,%d,%d,%d,%d" % (
                x_min, y_min, width, height, int(info[1]))
            f.write(box_info)
        f.write('\n')
    f.close()

if __name__ == "__main__":
    path_14 = ""
    path_17 = ""
    path_1417 = "/home/adr/datasets/vision/coco/14+17"
    combine(path_14, path_17, path_1417, mode = 1417)