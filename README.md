# 简介
自动驾驶是目前非常有前景的行业，而视觉感知作为自动驾驶中的“眼睛”，有着非常重要的地位和作用。为了能有效地识别到行驶在路上的动态目标，如汽车、行人等，我们需要提前对这些目标的进行训练，从而能够有效地避开，防止事故的发生。
目标检测是我们用于识别图像中目标位置的技术。如果图像中有单个目标，并且我们想要检测该目标，则称为图像定位。如果图像中有多个目标怎么办？嗯，这就是目标检测！
自动驾驶是目标检测最有趣和最近的应用之一，自动驾驶汽车是能够在很少或无人为引导的情况下自行移动的车辆。现在，为了让汽车决定它的行动，即要么向前移动，要么停车，要么转弯，它必须知道它周围所有物体的位置。使用目标检测技术，汽车可以检测其他汽车，行人，交通信号等物体。
而大多数应用程序需要实时分析，实时检测。行业的动态性质倾向于即时结果，而这正是实时目标检测的结果。

# 运行前
首先，在解释器终端或环境命令提示符中进行依赖库安装  
或使用[Anaconda3](https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Windows-x86_64.exe)来配置环境：
打开Anaconda Prompt输入  
```
conda create -n py39  python=3.9
```
来创建python=3.9的环境解释器  
```
conda activate py39
```
打开创建的py39环境解释器  
```
cd <your project location>
```
在Anconda PowerShell Prompt中打开项目地址，然后输入以下代码进行依赖项的安装
```
pip install -r requirements.txt
pip install -r requirements-torch.txt
```
### 谨记：项目文件的保存路径不能出现中文，否则会出现预处理失效
# 运行
## 数据标注
在解释器终端或命令提示符输入LabelImg
![python exe_20250615_231145](https://github.com/user-attachments/assets/133a09ee-c301-4acd-beaa-20a6c9916445)
页面左侧Open/Open Dir来选取需要进行数据标注的图片  
页面左侧Change Save Dir来选择数据表之后所保存的文件夹  
页面左侧Next Image/Prev Image来进行下一张或上一张图片的切换  
页面左侧Save进行图片的保存  
页面左侧YOLO/CreatML/Pascal VOC来选择数据标注所保存的类型    
页面左侧Create RectBox、Duplicate RectBox、Delete RectBox来进行标注框的创建、复制和删除  
![python exe_20250616_121055](https://github.com/user-attachments/assets/45e35862-3fbc-41a1-87a3-4a8877971d5a)

## 可视化显示
在解释器终端或命令提示符输入
```
python load.py
```
或者在Jetbrains Pycharm等其他IDE中点击运行（Shift+F10）
![python exe_20250616_111839](https://github.com/user-attachments/assets/da2f5506-88da-4a5c-872d-5901e320e360)
在左下角可进行选择所显示的图片或视频，图片支持jpg、png、jpeg格式，视频支持mp4、avi格式，同时还支持进行数据标注后的文件格式：.json、.xml、.txt。  
注：只有选择视频后才可点击右下角的播放按钮，不然不会生效  
示例效果：  
![python exe_20250616_113816](https://github.com/user-attachments/assets/354571ad-a071-412b-b6a6-a1f4b6bb1154)

## 预处理
在解释器终端或命令提示符输入
```
python pretreat.py
```
或者在Jetbrains Pycharm等其他IDE中点击运行（shift+F10）
这时候回自动读取input/文件夹进行预处理，处理后的图片保存在augmented_output/文件夹中
![局部截取_20250616_114612](https://github.com/user-attachments/assets/40005793-ed62-40ec-af7a-769709927c77)

## 模型训练
在解释器终端或命令提示符输入
```
python train_yolo.py
```
或者在Jetbrains Pycharm等其他IDE中点击运行（shift+F10）
在train_yolo.py的第7行可以选择训练的模型，默认是yolov8s
```
    model = YOLO("yolov8s.pt")
```
在第10行设置训练数据集的配置文件路径，data.yaml 包含训练、验证和测试数据的路径及类别信息  
```
        data="dataset/google/data.yaml",
```
在第12行设置训练的轮数（epochs），模型将对整个数据集进行 10 次完整训练
```
        epochs=20,
```
在第14行设置每个批次（batch）的大小，每次训练处理 8 张图像
```
        batch=8,
```
在第16行设置输入图像的尺寸，图像将被调整为 640x640 像素
```
        imgsz=640,
```
在第19行设置数据加载的工作进程数，2 个工作进程用于并行加载数据，由于使用了 if __name__ == "__main__"，可以安全设置 workers 大于 0
```
        workers=2,
```
在第21行指定训练使用的设备，device=0 表示使用第一个 GPU（CUDA 设备）
```
        device = 0 if torch.cuda.is_available() else 'cpu',
```
在第23行设置早停（early stopping）的耐心值，若 50 个 epoch 内验证性能无提升则停止训练
```
        patience=50
```
训练后的结果保存在runs/detect/train/文件夹中，训练好的模型保存在runs/detect/train/weughts/文件夹中：best.pt文件和last.pt文件
![局部截取_20250616_120658](https://github.com/user-attachments/assets/9d7d95df-51a6-4cc2-9db0-7958a1da2468)
![局部截取_20250616_120741](https://github.com/user-attachments/assets/44795ad9-172f-4299-bc21-7e076016db2d)
![results](https://github.com/user-attachments/assets/fbe80517-9221-4957-b16f-9ced091cd283)

## 目标车辆检测
![python exe_20250615_224709](https://github.com/user-attachments/assets/fe6d8d21-e889-489f-b643-8719057ce035)
可进行图像与视频检测，检测包括物体检测与实例分割
点击图像，点击选择图像，在image文件夹中选择要进行检测的图像，然后点击显示结果即可获得检测后的图像，检测前的图像默认保存在image/,检测后的图像默认保存在runs/detect/predict/
![python exe_20250615_225016](https://github.com/user-attachments/assets/73254497-d7de-4c1a-ab31-eeae187e6789)
点击视频，点击选择视频，在video文件夹中选择要进行检测的视频，然后点击显示结果即可获得检测后的视频并开始播放，检测前的视频迷人保存在video/,检测后的视频默认保存在video_output/
你可以在main.py文件的290或415行进行修改物体检测或实例分割的模型（默认使用yolo11n.pt和yolo11n-seg.pt）
```
model = YOLO('yolo11n.pt') if self.model1.currentText() == '物体检测' else YOLO('yolo11n-seg.pt')
```
