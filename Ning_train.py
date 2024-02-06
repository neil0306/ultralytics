from ultralytics import YOLO

# 加载模型
model = YOLO("yolov8s.yaml")             # 从头开始构建新模型, 如果需要魔改结构, 就直接修改里面的配置
model = YOLO("./pretrain/yolov8s.pt")    # 加载预训练模型（建议用于训练）

# 使用模型
model.train(data="/home/ning/Desktop/YOLOv8_notes/ultralytics/wheat_dataset.yaml", epochs=100)  # 训练模型, 指定数据集的相关参数
metrics = model.val()  # 在验证集上评估模型性能
results = model("https://ultralytics.com/images/bus.jpg")  # 对图像进行预测
success = model.export(format="onnx")  # 将模型导出为 ONNX 格式