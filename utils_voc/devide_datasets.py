import os
import random

def set_txt(path, trainval_percent = 0.8, train_percent = 0.6):
    """
    训练集:0.48    -> 大约 12639
    测试集:0.32    -> 大约 8426
    验证集:0.2     -> 大约 5267
    """

    # 确定要读写的xml文件夹和要保存到的文件
    xmlfilepath = path + "\\Annotations"
    txtsavepath = path  + "\\ImageSets\\Main"
    total_xml = os.listdir(xmlfilepath)
    num = len(total_xml)

    # 随机确定那些图片将被划分到训练集、验证集和测试集
    tv = int(num * trainval_percent)
    tr = int(tv * train_percent)
    trainval = random.sample(range(num), tv)
    train = random.sample(trainval, tr)

    # 打开四个即将写入的文件夹
    ftrainval = open( txtsavepath + '\\trainval.txt', 'w' )
    ftest = open( txtsavepath +'\\test.txt', 'w' )
    ftrain = open( txtsavepath + '\\train.txt', 'w' )
    fval = open( txtsavepath + '\\val.txt', 'w' )
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

if __name__ == "__main__":
    set_txt("./")