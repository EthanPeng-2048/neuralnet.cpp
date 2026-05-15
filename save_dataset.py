# save_mnist_csv.py
import torch
import torchvision
import csv
import os

def export_to_csv(dataset, filename):
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        for img, label in dataset:
            # 将图像展平为列表，像素值已是 float 0~1
            pixels = img.view(-1).tolist()
            writer.writerow([label] + pixels)

def main():
    os.makedirs("./build/mnist_data", exist_ok=True)

    train_dataset = torchvision.datasets.MNIST(
        "./mnist_data/raw", train=True, download=True,
        transform=torchvision.transforms.ToTensor()
    )
    test_dataset = torchvision.datasets.MNIST(
        "./mnist_data/raw", train=False, download=True,
        transform=torchvision.transforms.ToTensor()
    )

    export_to_csv(train_dataset, "./build/mnist_data/train.csv")
    export_to_csv(test_dataset, "./build/mnist_data/test.csv")
    print("Saved train.csv and test.csv to ./build/mnist_data/")

if __name__ == "__main__":
    main()
