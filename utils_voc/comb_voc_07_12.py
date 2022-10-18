import os
import shutil

def combine(path_07, path_12, path_07_12):
    """
    path_07: 2007 的路径
    path_12: 2012 的路径
    """
    annotations_path_07 = path_07 + "\\Annotations" 
    jpegimages_path_07  = path_07 + "\\JPEGImages"
    annotations_path_12 = path_12 + "\\Annotations" 
    jpegimages_path_12  = path_12 + "\\JPEGImages"
    
    jpegimages_07 = os.listdir(jpegimages_path_07)
    jpegimages_12 = os.listdir(jpegimages_path_12)
    num_07 = 0
    for img in jpegimages_07:
        num_07 += 1  
        # 将 2007 的图片拷贝到自己 path_07_12
        shutil.copy(jpegimages_path_07+'/'+img, path_07_12 +'\\JPEGImages\\'+img)
        # 将 2007 的 annotation 拷贝到自己 path_07_12
        shutil.copy(annotations_path_07+'/'+img.split(".")[0] + ".xml", path_07_12 +'\\Annotations\\'+img.split(".")[0] + ".xml")
    
    num_12 = 0
    for img in jpegimages_12:
        # print(img.split(".")[0])

        if img.split("_")[0] != "2007":
            num_12 += 1
            # 将 2007 的图片拷贝到自己 path_07_12
            shutil.copy(jpegimages_path_12+'/'+img, path_07_12 +'\\JPEGImages\\'+img)
            # 将 2007 的 annotation 拷贝到自己 path_07_12
            shutil.copy(annotations_path_12+'/'+img.split(".")[0] + ".xml", path_07_12 +'\\Annotations\\'+img.split(".")[0] + ".xml")
        else:
            print("same! ")
    if ( num_07 + num_12 ) == len(os.listdir( path_07_12 +'\\Annotations' )):
        print("正确 !")

if __name__ == "__main__":

    combine("E:\\数据集\\VOC\\VOCdevkit\\VOC2007", "E:\\数据集\\VOC\\VOCdevkit\\VOC2012", "E:\\数据集\\VOC\\VOCdevkit\\VOC07+12")