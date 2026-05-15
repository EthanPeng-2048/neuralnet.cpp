"""
从 test.csv 中提取每个数字图像，保存为 digit.csv 格式（纯像素，不含标签）
输出到 build/test/ 目录下，按标签分子文件夹
"""
import csv
import os


def extract_digits(test_csv_path, output_dir):
    """读取 test.csv，将每行提取为单独的 digit CSV 文件"""
    os.makedirs(output_dir, exist_ok=True)

    # 按标签统计，用于文件命名
    label_counts = {}

    with open(test_csv_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            label = row[0]
            pixels = row[1:]  # 784 个像素值

            # 为每个标签创建子文件夹
            label_dir = os.path.join(output_dir, label)
            os.makedirs(label_dir, exist_ok=True)

            # 统计该标签已保存的数量
            if label not in label_counts:
                label_counts[label] = 0
            label_counts[label] += 1

            # 写入 digit CSV
            digit_path = os.path.join(label_dir, f"digit_{label_counts[label]:04d}.csv")
            with open(digit_path, 'w', newline='') as out:
                writer = csv.writer(out)
                writer.writerow(pixels)

    # 打印统计信息
    total = sum(label_counts.values())
    print(f"提取完成！共 {total} 张图片")
    for label in sorted(label_counts.keys(), key=int):
        print(f"  标签 {label}: {label_counts[label]} 张")


if __name__ == "__main__":
    test_csv = "./build/mnist_data/test.csv"
    output = "./build/test"
    extract_digits(test_csv, output)
