# 用来处理"全球小麦检测"数据集

import os
import json
import numpy as np
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def get_bbox(row):
    bboxes = []
    list1 = row.bbox.replace("[", "").replace("]","").replace(",","")   ## 去掉逗号和方括号
    for i in list1.split(" "):
        bboxes.append(float(i))   ## 字符串转数字
    return bboxes

# Convert the bounding boxes in YOLO format.
def get_yolo_format_bbox(img_w, img_h, bboxes):
    yolo_boxes = []
    x = bboxes[0]       ## 左上角坐标
    y = bboxes[1]       ## 左上角坐标
    w = bboxes[2] 
    h = bboxes[3] 
    
    ## 中心坐标
    xc = x + int(w/2) # xmin + width/2
    yc = y + int(h/2) # ymin + height/2

    ## 坐标点归一化: 高,宽 以及 中心点坐标值 都用对应的边长进行归一化
    yolo_boxes=[1, xc/img_w, yc/img_h, w/img_w, h/img_h] # confidence, x_center y_center width height
    yolo_boxes = str(yolo_boxes).replace("[", "").replace("]","").replace(",","")
    return yolo_boxes

def split_dataset(TRAIN_PATH):
    #使用pandas读取csv文件，新建path列，保存图片路径
    df = pd.read_csv(TRAIN_PATH.replace("train/","train.csv"))   ## 读取标签文件, 获得 ground truth bbox
    df['path'] = df.apply(lambda row:TRAIN_PATH+row.image_id+'.jpg', axis=1)    ## csv中第一列记录的是图片的文件名(但没有扩展名), 这里补齐文件扩展名, 然后将扩展好的数据新增为一列, 叫做 'path'
    print(df.head())
    train_list, valid_list = train_test_split(df.image_id.unique(), test_size=0.2, random_state=1)  ## 用sklearn的train_test_split函数, 将数据集划分为训练集和验证集: 
                                                                                                    ## 传入图片id(由于一张图有多个bbox, 所以需要去重, 用unique), 测试集占比20%, 随机种子为1
                                                                                                    ## 得到的是两个列表, 分别是训练集和验证集的图片id
    train_df = df.loc[df.image_id.isin(train_list)]             ## 根据行索引取出训练集的数据
    valid_df = df.loc[df.image_id.isin(valid_list)]             ## 根据行索引取出验证集的数据

    ## 利用 pandas 的 loc 函数, 新增一列, 叫做 'split', 用来标记这张图属于 train 还是 valid
    train_df.loc[:, 'split'] = 'train'                          
    valid_df.loc[:, 'split'] = 'valid'

    df = pd.concat([train_df, valid_df]).reset_index(drop=True)  ## 拼接两个表格, 去掉重新排序索引
    return df,train_df,valid_df

def copy_train_test(TRAIN_PATH,TEST_PATH,WORK_ROOT):
    import shutil
    
    ## ----------- 创建目录, 用于存放数据集图片 -----------
    os.makedirs(WORK_ROOT+'dataset/images/train', exist_ok=True)
    os.makedirs(WORK_ROOT+'dataset/images/valid', exist_ok=True)
    os.makedirs(WORK_ROOT+'dataset/images/test', exist_ok=True)

    os.makedirs(WORK_ROOT+'dataset/labels/train', exist_ok=True)
    os.makedirs(WORK_ROOT+'dataset/labels/valid', exist_ok=True)
    os.makedirs(WORK_ROOT+'dataset/labels/test', exist_ok=True)
    ## -----------------------------------------------
    
    df,train_df,valid_df= split_dataset(TRAIN_PATH)
    train_list = train_df.path.unique()  ## 用unique去重, 得到训练集的图片路径
    valid_list = valid_df.path.unique()  ## 用unique去重, 得到验证集的图片路径
    ## 复制图片到指定目录
    for file in tqdm(train_list):
        shutil.copy2(file,WORK_ROOT+'dataset/images/train') ## copy2 会复制文件和元数据
    for file in tqdm(valid_list):
        shutil.copy(file, WORK_ROOT+'dataset/images/valid') ## copy 单纯复制文件, 不复制元数据
    
    ## 处理测试集
    src = TEST_PATH
    trg = WORK_ROOT+'dataset/images/test'

    files=os.listdir(src)
    for fname in files:
        shutil.copy2(os.path.join(src,fname), trg)
        
    return df

def csv2yolo(df,WORK_ROOT):
    # Prepare the txt files for bounding box    按行处理数据集, 生成yolo格式的标签文件
    for i in tqdm(range(len(df))):
        row = df.loc[i]
        img_id = row.image_id
        split = row.split
        
        if split=='train':
            file_name = WORK_ROOT+'dataset/labels/train/{}.txt'.format(img_id)
        else:
            file_name = WORK_ROOT+'dataset/labels/valid/{}.txt'.format(img_id)
            
        bboxes = get_bbox(row)  ## 将字符串数据转换为数字, 得到当前行的bbox的坐标
            
        # Format for YOLOv5
        yolo_bboxes = get_yolo_format_bbox(IMG_SIZE, IMG_SIZE, bboxes)   ## (confidence, cx, cy, w, h), 注意是bbox中心点坐标
            
        with open(file_name, 'a') as f:
            f.write(yolo_bboxes)
            f.write('\n')
    print("+++Done!+++")

if __name__ == '__main__':
    ##  -------------- 以下路径需要修改为当前机器的 绝对路径, 注意不要漏掉最后的 "/"  ----------------
    TRAIN_PATH = "/home/ning/Desktop/YOLOv8_notes/ultralytics/dataset/train/"
    TEST_PATH = '/home/ning/Desktop/YOLOv8_notes/ultralytics/dataset/test/'
    WORK_ROOT = "/home/ning/Desktop/YOLOv8_notes/ultralytics/"
    ## -----------------------------------------------------------------

    IMG_SIZE = 1024   ## 数据集图片大小

    df = copy_train_test(TRAIN_PATH,TEST_PATH,WORK_ROOT)
    csv2yolo(df,WORK_ROOT)
    
