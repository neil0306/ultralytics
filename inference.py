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
    image = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)  # 3. BGR2RGB
    # Calculate widht and height and paddings
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
    # Pad the short side with (128,128,128)   
    image = cv2.copyMakeBorder(
        image, ty1, ty2, tx1, tx2, cv2.BORDER_CONSTANT, (114, 114, 114))  # image:图像， ty1, ty2.tx1,tx2: 相应方向上的边框宽度，添加的边界框像素值为常数，value填充的常数值
    image = image.astype(np.float32)   # 7.unit8-->float
    # Normalize to [0,1]
    image /= 255.0    # 8. 逐像素点除255.0
    # HWC to CHW format:
    image = np.transpose(image, [2, 0, 1])   # 9. HWC2CHW
    # CHW to NCHW format
    image = np.expand_dims(image, axis=0)    # 10.CWH2NCHW[1,3,640,640]
    # Convert the image to row-major order, also known as "C order":
    image = np.ascontiguousarray(image)  # 11.ascontiguousarray函数将一个内存不连续存储的数组转换为内存连续存储的数组，使得运行速度更快
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
    model = attempt_load_weights(r"D:\workspace\detect_project\ultralytics\runs\detect\train3\weights\best.pt")
    test_root = r"D:\workspace\detect_project\ultralytics_source\dataset\images\test"

    INPUT_W=640
    INPUT_H=640

    names = ["none","wheat"]
    files = os.listdir(test_root)

    for file in files:
        print(file)
        img_path = os.path.join(test_root,file)
        # 数据预处理
        image,image_raw,h,w = preprocess_image(img_path)
        input_ = torch.tensor(image)
        # 模型前向推理
        preds = model(input_)
        # 结果后处理
        preds = postprocess(preds, image, image_raw)

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
                    cv2.putText(image_raw, label_text+":"+str(prob), (c1[0], c1[1] - 2), 0, tl / 2, [0, 0, 255], 
                        thickness=tf, lineType=cv2.LINE_AA)

                    if not os.path.exists("./detect_res"):
                        os.makedirs("./detect_res")
                    cv2.imwrite("./detect_res/"+file,image_raw)










