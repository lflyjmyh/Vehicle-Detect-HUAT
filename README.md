# 简介
自动驾驶是目前非常有前景的行业，而视觉感知作为自动驾驶中的“眼睛”，有着非常重要的地位和作用。为了能有效地识别到行驶在路上的动态目标，如汽车、行人等，我们需要提前对这些目标的进行训练，从而能够有效地避开，防止事故的发生。
目标检测是我们用于识别图像中目标位置的技术。如果图像中有单个目标，并且我们想要检测该目标，则称为图像定位。如果图像中有多个目标怎么办？嗯，这就是目标检测！
自动驾驶是目标检测最有趣和最近的应用之一，自动驾驶汽车是能够在很少或无人为引导的情况下自行移动的车辆。现在，为了让汽车决定它的行动，即要么向前移动，要么停车，要么转弯，它必须知道它周围所有物体的位置。使用目标检测技术，汽车可以检测其他汽车，行人，交通信号等物体。
而大多数应用程序需要实时分析，实时检测。行业的动态性质倾向于即时结果，而这正是实时目标检测的结果。

# 运行前
首先，在解释器终端或环境命令提示符中进行依赖库安装
```
pip install -r requirements.txt
pip install -r requirements-torch.txt
```
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



## 目标车辆检测
![python exe_20250615_224709](https://github.com/user-attachments/assets/fe6d8d21-e889-489f-b643-8719057ce035)
可进行图像与视频检测，检测包括物体检测与实例分割
点击图像，点击选择图像，在image文件夹中选择要进行检测的图像，然后点击显示结果即可获得检测后的图像，检测后的图像保存在runs/detect/predict/
![python exe_20250615_225016](https://github.com/user-attachments/assets/73254497-d7de-4c1a-ab31-eeae187e6789)
点击视频，点击选择视频，在video文件夹中选择要进行检测的视频，然后点击显示结果即可获得检测后的视频并开始播放，检测后的视频保存在video_output/
你可以在main.py文件的290或415行进行修改物体检测或实例分割的模型（默认使用yolo11n.pt和yolo11n-seg.pt）
```
model = YOLO('yolo11n.pt') if self.model1.currentText() == '物体检测' else YOLO('yolo11n-seg.pt')
```
