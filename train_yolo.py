import os
import torch
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolov8s.pt")
    model.train(
        # 指定数据集的配置文件路径，data.yaml 包含训练、验证和测试数据的路径及类别信息
        data="E:/AiProject/dataset/google/data.yaml",
        # 设置训练的轮数（epochs），模型将对整个数据集进行 10 次完整训练
        epochs=20,
        # 设置每个批次（batch）的大小，每次训练处理 8 张图像
        batch=8,
        # 设置输入图像的尺寸，图像将被调整为 640x640 像素
        imgsz=640,
        # 设置数据加载的工作进程数，2 个工作进程用于并行加载数据
        # 由于使用了 if __name__ == "__main__"，可以安全设置 workers 大于 0
        workers=2,
        # 指定训练使用的设备，device=0 表示使用第一个 GPU（CUDA 设备）
        device = 0 if torch.cuda.is_available() else 'cpu',
        # 设置早停（early stopping）的耐心值，若 50 个 epoch 内验证性能无提升则停止训练
        patience=50
    )