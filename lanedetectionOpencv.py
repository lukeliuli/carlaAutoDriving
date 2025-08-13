import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_road_mask(image):
    """
    使用 HSV 颜色空间提取白色和黄色车道线
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 白色车道线范围
    lower_white = np.array([0, 0, 170])
    upper_white = np.array([255, 40, 255])

    # 黄色车道线范围
    lower_yellow = np.array([18, 40, 100])
    upper_yellow = np.array([34, 255, 255])

    mask_white = cv2.inRange(hsv, lower_white, upper_white)
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    road_mask = cv2.bitwise_or(mask_white, mask_yellow)

    return road_mask

def region_of_interest(image):
    """
    定义感兴趣区域 ROI（梯形）
    """
    if len(image.shape) == 3:  # 彩色图像
        height, width, _ = image.shape
    else:  # 灰度图像
        height, width = image.shape

    vertices = np.array([[
        (0, height),
        (width // 2 - 60, height // 2 + 40),
        (width // 2 + 60, height // 2 + 40),
        (width, height)
    ]], dtype=np.int32)

    mask = np.zeros_like(image)
    # 如果图像是RGB，则需要调整mask的形状以匹配
    if len(image.shape) > 2:
        mask = np.zeros_like(image[:,:,0])
    
    cv2.fillPoly(mask, vertices, 255)
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    return masked_image

def sliding_window_search(binary_warped, nwindows=9, margin=100, minpix=50):
    """
    滑动窗口搜索 + 多项式拟合
    """
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255

    midpoint = np.int(histogram.shape[0] // 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    window_height = np.int(binary_warped.shape[0] // nwindows)
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    leftx_current = leftx_base
    rightx_current = rightx_base
    left_lane_inds = []
    right_lane_inds = []

    for window in range(nwindows):
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height

        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        cv2.rectangle(out_img, (win_xleft_low, win_y_low),
                      (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low),
                      (win_xright_high, win_y_high), (0, 255, 0), 2)

        good_left = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                     (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                      (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        left_lane_inds.append(good_left)
        right_lane_inds.append(good_right)

        if len(good_left) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left]))
        if len(good_right) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right]))

    try:
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        return left_fit, right_fit, left_fitx, right_fitx, ploty, out_img

    except:
        return None, None, None, None, None, None

def draw_lane_lines(image, left_fit, right_fit, ploty):
    """
    绘制拟合的车道线
    """
    overlay = np.zeros_like(image)
    pts_left = np.array([np.transpose(np.vstack([left_fit, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fit, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    cv2.fillPoly(overlay, np.int_([pts]), (0, 255, 0))
    result = cv2.addWeighted(image, 1, overlay, 0.3, 0)
    return result

##效果不好，适应性不强，只做演示
def process_image(image):


    result = None
    # 1. 颜色空间过滤
    road_mask = get_road_mask(image)
    #return road_mask
    # 2. ROI 掩码
    masked_road = region_of_interest(road_mask)
    #return masked_road
    # 3. 滑动窗口 + 多项式拟合
    left_fit, right_fit, left_fitx, right_fitx, ploty, out_img = sliding_window_search(masked_road)

    if left_fit is not None and right_fit is not None:
        result = draw_lane_lines(image, left_fitx, right_fitx, ploty)
        #cv2.imshow("Lane Detection", result)
        #cv2.waitKey(0)
    else:
        print("未检测到车道线。")

    #cv2.destroyAllWindows()
    #cv2.saveImage("output.jpg", result)
    return result

# 示例调用
#process_image(image)


import cv2
import numpy as np

def simple_process_image(image):
    final_image = None
    # 读取输入图像
    
    height, width = image.shape[:2]

    # 边缘检测
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 灰度转换    
    blur = cv2.GaussianBlur(gray, (5, 5), 0)  # 高斯模糊
    edges = cv2.Canny(blur, 10, 150)  # Canny边缘检测

    #image = gray
    # 定义感兴趣区域(ROI)，这里假设是一条直路
    mask = np.zeros_like(edges)
    ignore_mask_color = 255
    
    # 假设图像底部三分之二为感兴趣区域，可以根据实际情况调整
    vertices = np.array([[
        (0, height),
        (width // 10, height // 2 + height // 10),
        (width-width // 10,height // 2 + height // 10),
        (width, height)
    ]], dtype=np.int32)

    mask = np.zeros_like(gray)
     
    cv2.fillPoly(mask, vertices, 255)
    masked_edges = cv2.bitwise_and(edges, mask)

    # 使用霍夫变换检测直线
    rho = 1  # 距离分辨率以像素为单位
    theta = np.pi / 180  # 角度分辨率以弧度为单位
    threshold = 15     # 需要多少个连接点来检测一条直线
    min_line_length = 20  # 直线的最小长度（以像素为单位）
    max_line_gap = 10    # 线段之间最大允许间隔以认为是同一直线
    lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)

    # 在原图上画出检测到的线条
    line_image = np.copy(image) * 0  # 创建一个空白图像来画线
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)

    # 将原始图像与线条图像合并
    color_edges = np.dstack((edges, edges, edges)) 
    final_image = cv2.addWeighted(color_edges, 0.8, line_image, 1, 0) 
   
    return final_image