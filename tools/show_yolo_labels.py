# 在对应图片上显示处理好的数据集标签
import os
import cv2

## 返归一化BBox坐标
def parse_yolo_labels(file_path,h,w):
    with open(file_path) as f:
        bbox = []
        for i in f:
            split_ = i.strip().split(" ")
            box = split_[1:]
            box[::2] = [w*float(i) for i in box[::2]]
            box[1::2] = [h*float(i) for i in box[1::2]]
            bbox.append(box)
    return bbox

def show_bbox(img,bbox,save_path):
    for box_ in bbox:
        x1 = int(box_[0] - box_[2]/2)
        y1 = int(box_[1] - box_[3]/2)
        x2 = int(box_[0] + box_[2]/2)
        y2 = int(box_[1] + box_[3]/2)
        cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,255),2,1)
    
    cv2.imwrite(save_path,img)
    # cv2.imshow('result',img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
if __name__ == '__main__':
    label_path = '/home/ning/Desktop/YOLOv8_notes/ultralytics/dataset/labels/train'
    image_root = "/home/ning/Desktop/YOLOv8_notes/ultralytics/dataset/images/train"
    
    save_root = "/home/ning/Desktop/YOLOv8_notes/ultralytics/dataset/labels/show_labels"
    os.makedirs(save_root, exist_ok=True)
    
    for file in os.listdir(label_path)[:]:
        file_path = os.path.join(label_path,file)
        img_path = os.path.join(image_root,file.replace('txt','jpg'))
        img = cv2.imread(img_path)
        h,w,r = img.shape
        bbox = parse_yolo_labels(file_path,h,w)
        save_path = os.path.join(save_root,file).replace('.txt','.jpg')
        show_bbox(img,bbox,save_path)

