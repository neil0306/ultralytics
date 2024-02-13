## ---------------- 相比 detect.py, 这里是不进行封装的写法(标准的推理写法), 但是功能是一样的 ------------
from ultralytics import YOLO
from ultralytics.utils import ROOT, ops
from ultralytics.nn.tasks import  attempt_load_weights
import os
import cv2
import torch
import numpy as np

# 前处理和YOLOv5相同
def preprocess_image(image_path):
    image_raw = cv2.imread(image_path)         # 1.opencv读入图片  1000，500   100，100  100，25+50+25 padding
    h, w, c = image_raw.shape                  # 2.记录图片大小
    image = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)  # 3. BGR2RGB     这里一样要和训练的时候保持一致
    # Calculate widht and height and paddings     
    ## ning: 保持宽高比例的缩放方式: 计算缩放比例(要保持宽高中较大者, 如宽1000,高500, 要缩放10倍, 此时缩放后的宽为100,高为50, 为了维持原图的宽高比, 此时只需要对高进行填充, 高度填充为25+50+25=100, 看上去图片是上下有黑边的)
    r_w = INPUT_W / w  # INPUT_W=INPUT_H=640  # 4.计算宽高缩放的倍数 r_w,r_h
    r_h = INPUT_H / h
    if r_h > r_w:       # 5.如果原图的高小于宽(长边），则长边缩放到640，短边按长边缩放比例缩放
        tw = INPUT_W
        th = int(r_w * h)
        tx1 = tx2 = 0
        ty1 = int((INPUT_H - th) / 2)  # ty1=（640-短边缩放的长度）/2 ，这部分是YOLOv5为加速推断而做的一个图像缩放算法
        ty2 = INPUT_H - th - ty1       # ty2=640-短边缩放的长度-ty1
    else:
        tw = int(r_h * w)
        th = INPUT_H
        tx1 = int((INPUT_W - tw) / 2)
        tx2 = INPUT_W - tw - tx1
        ty1 = ty2 = 0
    # Resize the image with long side while maintaining ratio
    image = cv2.resize(image, (tw, th),interpolation=cv2.INTER_LINEAR)  # 6.图像resize,按照cv2.INTER_LINEAR方法
    # Pad the short side with (114,114,114)   ning: 以往我们在填充图片区域的时候, 都用0填充(黑色), 但是在YOLOv8中, 它用的是灰度值114进行填充
    image = cv2.copyMakeBorder(
        image, ty1, ty2, tx1, tx2, cv2.BORDER_CONSTANT, (114, 114, 114))  # image:图像， ty1, ty2.tx1,tx2: 相应方向上的边框宽度，添加的边界框像素值为常数，value填充的常数值
    image = image.astype(np.float32)   # 7.unit8-->float
    # Normalize to [0,1]
    image /= 255.0    # 8. 逐像素点除255.0    ning: YOLOv8的归一化没有减去均值, 也没有除以方差, 而是直接除以255, 将像素值转换到[0,1]之间; 有不少网络中也是这么干的
    # HWC to CHW format:
    image = np.transpose(image, [2, 0, 1])   # 9. HWC2CHW
    # CHW to NCHW format
    image = np.expand_dims(image, axis=0)    # 10.CWH2NCHW[1,3,640,640]    转换为pytorch的格式 (n,c,h,w)
    # Convert the image to row-major order, also known as "C order":
    image = np.ascontiguousarray(image)  # 11.ascontiguousarray函数将一个内存不连续存储的数组转换为内存连续存储的数组，使得运行速度更快   (ning: 估计是底层矩阵乘法用的C/C++, 要求内存连续)
    return image, image_raw, h, w  # 处理后的图像，原图， 原图的h,w

def postprocess(preds, img, orig_img):
    preds = ops.non_max_suppression(preds,
                                    conf_thres=0.25,
                                    iou_thres=0.45,
                                    agnostic=False,
                                    max_det=300)
    return preds

if __name__ == '__main__':
    # model = YOLO("./runs/detect/train/weights/last.pt")  
    model = attempt_load_weights("/home/ning/Desktop/YOLOv8_notes/ultralytics/runs/detect/train2/weights/best.pt")    ## 从checkpoint加载模型
    test_root = "/home/ning/Desktop/YOLOv8_notes/ultralytics/dataset/images/test"   ## 指定测试图片路径

    INPUT_W=640
    INPUT_H=640

    names = ["none","wheat"]   ## 这个要和数据集保持一致, 数据集写的也是none和wheat
    files = os.listdir(test_root)


    """
    处理每张图片:
        1. 数据预处理
        2. 前向推理
        3. 结果后处理
    """
    for file in files:
        print(file)
        img_path = os.path.join(test_root,file)
        # 数据预处理
        image,image_raw,h,w = preprocess_image(img_path)   ## 图片大小为(640, 640)
        input_ = torch.tensor(image)
        # 模型前向推理
        preds = model(input_)  ## preds 是list, 里面是一个tensor 和 一个list
                                ## 第一个tensor, 维度是(1,6,8400), 6指的是(x1,y1,x2,y2 + 置信度 + num_classes), 8400则对应了三个尺度的特征图的预测结果(网络的金字塔部分, 下采样出来3个分支, 1/32, 1/16, 1/8, 预测得到的Anchor点数量分别是20*20, 40*40, 80*80), 
                                ##      每个Anchor点对应一个框, 故一共预测出来8400个框, 每个框6个参数(x1,y1,x2,y2 + 置信度 + num_classes)
                                ## 第二个list里面装了3个tensor, 分别对应三个尺度的特征图(1/32, 1/16, 1/8), 这三个东西相当于loss, 用于训练的, 推理的时候可以不管它
        # 结果后处理
        preds = postprocess(preds, image, image_raw)    ## 将预测结果进行后处理, 进行 Non-Maximum Suppression, 保留置信度高的框

        for i, det in enumerate(preds):  # detections per image
            gn = torch.tensor(image_raw.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if det is not None and len(det):
                # 恢复到原图尺度
                det[:, :4] = ops.scale_boxes(image.shape[2:], det[:, :4], image_raw.shape).round()
                for *xyxy, conf, cls_ in det:   # x1,y1,x2,y2
                    # 获取标签
                    label_text = names[int(cls_)]
                    prob = round(conf.cpu().detach().numpy().item(),2)
                    # 设置线宽
                    tl = round(0.02 * (image.shape[0] + image.shape[1]) / 2) + 1  # line/font thickness
                    color = (255, 255, 0)
                    # 绘制预测结果
                    c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                    cv2.rectangle(image_raw, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
                    # 绘制分类信息
                    tf = max(tl - 1, 1)  # font thickness
                    t_size = cv2.getTextSize(label_text+":"+str(prob), 0, fontScale=tl / 2, thickness=tf)[0]
                    c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
                    cv2.rectangle(image_raw, c1, c2, color, -1, cv2.LINE_AA)  # filled
                    cv2.putText(image_raw, label_text+":"+str(prob), (c1[0], c1[1] - 2), 0, tl / 2, [0, 0, 255],    ## 写上类别和置信度
                        thickness=tf, lineType=cv2.LINE_AA)

                    if not os.path.exists("./detect_res"):
                        os.makedirs("./detect_res")
                    cv2.imwrite("./detect_res/"+file,image_raw)










