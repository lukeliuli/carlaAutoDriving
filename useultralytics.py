from ultralytics import YOLO
import cv2
import numpy as np

# 加载YOLO模型，确保你选择了一个适合做分割任务的模型

def process_carla_imageseg(img_rgb,model_name = 'yolov8n-seg.pt'):
  

    model = YOLO(model_name)  # 使用YOLOv8的分割模型，请确保你已经下载了该模型文件

    # 读取输入图像
    img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)  # 转换成RGB格式
    img_rgb = cv2.resize(img_rgb, (640, 480))  # 
    # 执行预测
    results = model(img_rgb)  # 注意：在Ultralytics库中，直接调用模型即可进行预测

    # 初始化输出图像
    mask_image = np.zeros_like(img_rgb)  # 用于存储分割mask
    annotated_image = img_rgb.copy()  # 用于绘制分割结果

    # 遍历每个检测结果
    for result in results:
        boxes = result.boxes  # 获取所有边界框
        masks = result.masks  # 获取所有分割掩膜
        probs = result.probs  # 获取类别概率（如果适用）

        if masks is not None:
            # 遍历每个检测到的对象
            for i, (box, mask_data) in enumerate(zip(boxes, masks.data)):
                # 解析边界框信息
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # 提取边界框坐标
                conf = box.conf[0]  # 置信度分数
                cls = int(box.cls[0])  # 类别ID
                
                # 将分割掩膜转换为numpy数组
                mask_np = mask_data.cpu().numpy()
                mask_np = (mask_np > 0.5).astype(np.uint8)  # 转换为二值mask
                
                # 生成随机颜色用于可视化
                color = [int(c) for c in np.random.randint(0, 255, size=3)]
                
                # 将分割掩膜应用到mask_image上
                colored_mask = np.zeros_like(img_rgb)
                colored_mask[mask_np == 1] = color
                mask_image = colored_mask
                
                # 在原图上绘制分割结果
                overlay = annotated_image.copy()
                overlay[mask_np == 1] = color
                alpha = 0.5  # 透明度
                cv2.addWeighted(overlay, alpha, annotated_image, 1 - alpha, 0, annotated_image)
                
                # 可选：在图像上绘制边界框和标签
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
                label = f'Class {cls} Conf: {conf:.2f}'
                cv2.putText(annotated_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)


        return mask_image, annotated_image