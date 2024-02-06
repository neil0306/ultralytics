## ----------- 按照官方推荐的做法, 测试自己训练好的YOLO模型 ------------

# 引入opencv
import cv2

# 引入YOLO模型
from ultralytics import YOLO

# 打开图像
img_path = "/home/ning/Desktop/YOLOv8_notes/ultralytics/dataset/images/test/2fd875eaa.jpg"  # 这里修改你图像保存路径

# 打开图像
img = cv2.imread(filename=img_path)

# 加载模型
model = YOLO(model="/home/ning/Desktop/YOLOv8_notes/ultralytics/runs/detect/train2/weights/best.pt") # 这里修改你图像保存路径

# 正向推理 (使用 CPU 推理)
res = model(img)

# 绘制推理结果
annotated_img = res[0].plot()

# 显示图像
# cv2.imshow(winname="YOLOV8", mat=annotated_img)

# # 等待时间
# cv2.waitKey(0)

# 绘制推理结果
cv2.imwrite(filename=img_path.replace('.jpg','_result.jpg'), img=annotated_img)
