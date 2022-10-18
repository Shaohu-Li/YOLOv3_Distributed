import os
import random
try:
    import xml.etree.cElementTree as ET # 尝试导入C语言编写的api，可以加快速度
except ImportError:
    import xml.etree.ElementTree as ET
"""
ET 学习了解链接
https://zhuanlan.zhihu.com/p/152207687
"""

def get_classes(classes_path):
    with open(classes_path, encoding='utf-8') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names, len(class_names)
    

def convert_annotation( img_id, path, self_txt_file ):
    """
     函数用于读取 annotation 中的 xml 文件, 并写入到传入的文件中。

     参数：
     img_id         : 给定的xml文件的名称
     path           : 传入的路径
     self_txt_file  : 将要写入文件的句柄
    """
    # print(path)

    xml_file = open( os.path.join(path, "Annotations\\%s.xml" % (img_id)), encoding='utf-8' )
    tree=ET.parse(xml_file)
    root = tree.getroot()

    for obj in root.iter('object'): #遍历该Element所有后代，指定object进行遍历寻找，也就是查找 "object"
        difficult = 0 
        if obj.find('difficult') != None:
            difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1: # 比较难识别的，eval中不计
            continue
        # 转换 类别 为 数字
        cls_id = classes.index(cls)
        # 提取坐标
        xmlbox = obj.find('bndbox')
        b = (int(float(xmlbox.find('xmin').text)), int(float(xmlbox.find('ymin').text)), int(float(xmlbox.find('xmax').text)), int(float(xmlbox.find('ymax').text)))
        self_txt_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))


def divide_voc_dataset(path, trainval_percent = 0.8, train_percent = 0.6, mode = 0):
    """
    参数：
        path: 输入的路径
        trainval_percent: 训练集和验证集在整个数据集所占的比例
        train_percent   : 训练集在训练集和验证集所占的比例
        mode            : 
                        == 0 -> 代表同时生成 Mian 和 SelfMain 里面的文件划分
                        == 1 -> 代表仅仅生成 Main 里面的文件划分
                        == 2 -> 代表 进行生成 SelfMain 里面的文件划分
    """
    #---------------------------------------------------------------------#
    # 训练集:0.48    -> 大约 12639
    # 测试集:0.32    -> 大约 8426
    # 验证集:0.2     -> 大约 5267
    #---------------------------------------------------------------------#
    
    txt_name_list = ["train", "val", "test"]

    if mode == 0 or mode == 1:
        # 确定要读写的xml文件夹和要保存到的文件
        xmlfilepath  = os.path.join(path, "\\Annotations")
        txtsavepath  = os.path.join(path, "\\ImageSets\\Main")
        traintxtpath = os.path.join(path, "\\ImageSets\\SelfMain")

        temp_xml     = os.listdir(xmlfilepath)
        total_xml    = []
        # 为了防止 文件中存在其他的文件
        for xml in temp_xml:
            if xml.endswith(".xml"):
                total_xml.append(xml)

        num = len(total_xml)

        # 随机确定那些图片将被划分到训练集、验证集和测试集
        tv       = int(num * trainval_percent)
        tr       = int(tv * train_percent)
        trainval = random.sample(range(num), tv)
        train    = random.sample(trainval, tr)

        print("train and val size",tv)
        print("train size",tr)

        # 打开四个单纯写入划分图片名称的文件
        ftrainval       = open( os.path.join(txtsavepath, '\\trainval.txt'), 'w' )  
        ftest           = open( os.path.join(txtsavepath, '\\test.txt'), 'w' )
        ftrain          = open( os.path.join(txtsavepath, '\\train.txt'), 'w' )
        fval            = open( os.path.join(txtsavepath, '\\val.txt'), 'w' )


        for i in range(num):
            name = total_xml[i][:-4] + '\n'
            if i in trainval:
                ftrainval.write(name)  
                if i in train: # 不是 训练集 肯定就是 验证集
                    ftrain.write(name)
                else:
                    fval.write(name)
            else: # 不是 训练集 和 验证集，就是 测试集
                ftest.write(name)

        ftrainval.close()  
        ftrain.close()
        fval.close()
        ftest.close()
    
    if mode == 0 or mode == 2:
        print( "Generate SelfMain/selftrain.txt、SelfMain/selfval.txt and SelfMain/selfval.txt for train." )
        for txt_name in txt_name_list: 
            img_ids = open( os.path.join(path, "ImageSets\\Main\\%s.txt" % (txt_name)), encoding='utf-8' ).read().strip().split()
            self_txt_file = open( os.path.join(path, 'ImageSets\\SelfMain\\self%s.txt' % txt_name), 'w', encoding='utf-8' )
            for img_id in img_ids:
                print(img_id)

                self_txt_file.write("%s/JPEGImages/%s.jpg" % ( os.path.abspath(path), img_id ))
                convert_annotation(img_id, path, self_txt_file)
                self_txt_file.write('\n')

if __name__ == "__main__":
    random.seed(0)

    # 自己的voc数据的路径
    datapath= "/home/adr/datasets/vision/coco/14+17"
    classes_path        = datapath + "\\ImageSets\\SelfMain\\voc_classes.txt"

    classes, _      = get_classes(classes_path)
    divide_voc_dataset(datapath, mode=2)