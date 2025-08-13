import cv2
 
# 检查 OpenCV 版本
print(cv2.__version__)
 
# 读取一张图片并显示
image = cv2.imread('your_image_path.jpg')  # 替换 'your_image_path.jpg' 为你的图片路径
if image is not None:
    cv2.imshow('Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Error: Image not found")