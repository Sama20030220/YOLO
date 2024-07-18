from ultralytics import YOLO
import cv2

# 所需加载的模型目录
path = '/root/5/ultralytics-main/runs/detect/train/weights/best.pt'
# 需要检测的图片地址
img_path = "/root/5/ultralytics-main/11.jpg"

# 加载预训练模型
model = YOLO(path, task='detect')

# 检测图片
results = model(img_path)

print(results)

# 获取裁剪后的检测图像
cropped_images = results.save_crop()

# 保存裁剪后的检测图像
output_dir = "/root/5/ultralytics-main/cropped_images/"
for i, cropped_img in enumerate(cropped_images):
    output_path = f"{output_dir}/cropped_{i}.jpg"
    cv2.imwrite(output_path, cropped_img)
    print(f"保存裁剪后的检测图像 {i} 到 {output_path}")

# 显示原始图像
img = cv2.imread(img_path)
cv2.imshow("Original Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
