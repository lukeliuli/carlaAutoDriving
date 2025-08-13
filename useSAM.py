from ultralytics import FastSAM
import cv2
import numpy as np
'''
def process_carla_imagesamseg(img_rgb, model_name="FastSAM-s.pt"):
    """
    使用 FastSAM 模型对输入的 RGB 图像进行分割，并在原图上绘制分割结果。
    
    :param img_rgb: 输入的RGB图像(numpy数组形式).
    :param model_name: FastSAM 模型权重文件的路径或名称.
    :return: 绘制了分割结果的原图像和合成的mask图像。
    """
    # 加载 FastSAM 模型
    model = FastSAM(model_name)
    
    # 分割参数
    conf_threshold = 0.1
    iou_threshold = 0.2
    
    # 使用模型进行预测
    everything_results = model(img_rgb, device='cpu', retina_masks=True, conf=conf_threshold, iou=iou_threshold)
    img_rgb2 = img_rgb.copy()
    
    # 创建一个与输入图像大小相同的全零数组作为初始mask
    combined_mask = np.zeros(img_rgb.shape[:2], dtype=np.uint8)
    
    # 提取分割结果并绘制到原图上，同时更新combined_mask
    for result in everything_results:
        masks = result.masks.data.cpu().numpy() if len(result.masks.data) else []
        
        # 对每个mask进行处理
        for mask in masks:
            # 创建一个随机颜色用于绘制当前mask
            color = tuple(np.random.randint(0, 256, 3).tolist())
            
            # 将mask转换为uint8类型，并找到轮廓以绘制边界
            mask_uint8 = (mask * 255).astype(np.uint8)
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # 检查是否找到了轮廓
            if contours is not None and len(contours) > 0:
                # 在原图上绘制轮廓
                img_rgb2 = cv2.drawContours(img_rgb2, contours, -1, color=color, thickness=2)
                # 将当前mask添加到combined_mask中
                combined_mask = cv2.bitwise_or(combined_mask, mask_uint8)
            else:
                print("未找到有效的轮廓")

    return img_rgb2, combined_mask

# 示例调用
# 确保img_rgb是一个有效的numpy数组形式的图像，比如通过cv2.imread('path/to/image')读取
# img_segmented, final_mask = process_carla_imagesamseg(img_rgb)
'''
from ultralytics import FastSAM
import cv2
import numpy as np

def calculate_mask_area(mask):
    """计算mask的面积"""
    return np.sum(mask > 0)

def process_carla_imagesamseg(img_rgb, model_name="FastSAM-s.pt"):
    """
    使用 FastSAM 模型对输入的 RGB 图像进行分割，并在原图上绘制分割结果。
    
    :param img_rgb: 输入的RGB图像(numpy数组形式).
    :param model_name: FastSAM 模型权重文件的路径或名称.
    :return: 绘制了分割结果的原图像和合成的mask图像。
    """
    # 加载 FastSAM 模型
    model = FastSAM(model_name)
    
    # 分割参数
    conf_threshold = 0.02
    iou_threshold = 0.5
    
    # 使用模型进行预测
    everything_results = model(img_rgb, device='cpu', retina_masks=True, conf=conf_threshold, iou=iou_threshold)
    img_rgb2 = img_rgb.copy()
    
    masks_list = []
    
    for result in everything_results:
        masks = result.masks.data.cpu().numpy() if len(result.masks.data) else []
        for mask in masks:
            masks_list.append(mask)
            
    # 计算每个mask的面积并按面积从小到大排序
    masks_list_with_area = [(calculate_mask_area(mask), mask) for mask in masks_list]
    masks_list_sorted = sorted(masks_list_with_area, key=lambda x: x[0])
    
    combined_mask = np.zeros(img_rgb.shape[:2], dtype=np.uint8)
    
    for area, mask in masks_list_sorted:
        mask_uint8 = (mask * 255).astype(np.uint8)
        
        # 如果当前mask与combined_mask有重叠，忽略此mask
        overlap = cv2.bitwise_and(combined_mask, mask_uint8)
        if np.sum(overlap > 0) <area*0.01 or area< img_rgb.shape[0]*4: # 没有重叠
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours is not None and len(contours) > 0:
                color = tuple(np.random.randint(0, 256, 3).tolist())
                img_rgb2 = cv2.drawContours(img_rgb2, contours, -1, color=color, thickness=2)
                combined_mask = cv2.bitwise_or(combined_mask, mask_uint8)
        else:
            print(f"Mask with area {area} overlaps with existing masks and will be ignored.")
    
    return img_rgb2, combined_mask