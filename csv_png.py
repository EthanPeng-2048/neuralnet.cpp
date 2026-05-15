import numpy as np
from PIL import Image

# 读取 CSV 文件（假设数据在一行内，以逗号分隔）
csv_file = "./build/digit.csv"
with open(csv_file, "r") as f:
    data = f.read().strip().split(",")

# 转换为浮点数数组
pixels = np.array(data, dtype=np.float32)

# 检查像素数量是否为 784 (28x28)
if pixels.size != 784:
    raise ValueError(f"像素数量错误，期望 784，实际 {pixels.size}")

# 重塑为 28x28 矩阵
img_array = pixels.reshape(28, 28)

# 将像素值从 [0,1] 范围映射到 [0,255] 并转换为 uint8
img_array = (img_array * 255).astype(np.uint8)

# 创建图像并保存
img = Image.fromarray(img_array, mode="L")  # "L" 表示灰度图
img.save("./build/output.png")

print("图片已保存为 output.png")
