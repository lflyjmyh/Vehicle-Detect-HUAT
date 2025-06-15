import os  # 导入 os 模块，用于文件和目录操作
import sys  # 导入 sys 模块，用于系统相关操作，如退出程序
import cv2  # 导入 OpenCV 库，用于图像和视频处理
import glob  # 导入 glob 模块，用于查找最新预测目录和图像文件
from ultralytics import YOLO  # 从 ultralytics 库导入 YOLO 类，用于加载 YOLOv8 模型
from PyQt5.QtWidgets import QApplication, QWidget, QFileDialog, QGraphicsScene, QDialog, QHBoxLayout, QLabel, \
    QVBoxLayout, QPushButton  # 导入 PyQt5 的界面组件类，用于构建图形用户界面
from PyQt5.QtGui import QPixmap, QFont, QIcon, QPalette, QColor, QImage  # 导入 PyQt5 的图形相关类
from PyQt5.QtCore import Qt, QTimer, QRectF  # 导入 PyQt5 的核心类，包含 QRectF
from PyQt5 import uic  # 导入 PyQt5 的 uic 模块，用于加载 UI 文件
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent  # 导入 PyQt5 的多媒体类，用于播放视频
from PyQt5.QtMultimediaWidgets import QVideoWidget  # 导入 PyQt5 的视频显示控件类


class MyWindow(QWidget):  # 定义 MyWindow 类，继承自 QWidget，用于创建主窗口
    def __init__(self):  # 构造函数，初始化窗口
        super().__init__()  # 调用父类 QWidget 的构造函数
        self.init_ui()  # 调用初始化界面方法
        self.video_timer = QTimer(self)  # 创建定时器，用于更新视频帧
        self.video_timer.timeout.connect(self.update_video_frame)  # 绑定定时器超时信号到更新视频帧方法
        self.video_cap = None  # 初始化视频捕获对象
        self.output_video_path = None  # 初始化输出视频路径
        self.raw_video_cap = None  # 初始化原始视频捕获对象
        self.raw_video_timer = QTimer(self)  # 创建用于更新原始视频帧的定时器
        self.raw_video_timer.timeout.connect(self.update_raw_video_frame)  # 绑定定时器超时信号

    def init_ui(self):  # 初始化用户界面的方法
        self.ui = uic.loadUi("./ui/main.ui")  # 使用 PyQt5 的 uic 模块加载 UI 文件

        # 菜单项
        self.actiondefault = self.ui.actiondefault  # 获取默认主题菜单项
        self.actionblack = self.ui.actionblack  # 获取黑色主题菜单项
        self.actionwhite = self.ui.actionwhite  # 获取白色主题菜单项
        self.actionblue = self.ui.actionblue  # 获取蓝色主题菜单项
        self.actionintro = self.ui.actionintro  # 获取介绍菜单项
        self.actionversion = self.ui.actionversion  # 获取版本菜单项
        self.actionexit = self.ui.actionexit  # 获取退出菜单项

        # 侧边栏按钮
        self.tab_image = self.ui.tab_image  # 获取图像处理选项卡按钮
        self.tab_video = self.ui.tab_video  # 获取视频处理选项卡按钮

        # tab1_image
        self.raw_img = self.ui.raw_image  # 获取原始图像显示区域
        self.res_img = self.ui.res_image  # 获取结果图像显示区域
        self.select_btn = self.ui.select_btn  # 获取选择图像按钮
        self.show_btn = self.ui.show_btn  # 获取显示检测结果按钮
        self.model1 = self.ui.combo1  # 获取模型选择下拉框
        self.conf1 = self.ui.conf1  # 获取置信度调节控件
        self.conf1.setRange(0.0, 1.0)  # 设置置信度调节范围
        self.conf1.setSingleStep(0.01)  # 设置置信度调节步长
        self.conf1.setValue(0.25)  # 设置置信度默认值
        self.IOU1 = self.ui.IOU1  # 获取 IOU 调节控件
        self.IOU1.setRange(0.0, 1.0)  # 设置 IOU 调节范围
        self.IOU1.setSingleStep(0.01)  # 设置 IOU 调节步长
        self.IOU1.setValue(0.45)  # 设置 IOU 默认值
        self.class1 = self.ui.class1  # 获取类别输入框

        # tab2_video
        self.raw_video = self.ui.raw_video  # 获取原始视频显示区域
        self.res_video = self.ui.res_video  # 获取结果视频显示区域
        self.choose_video = self.ui.choose_video  # 获取选择视频按钮
        self.show_video = self.ui.show_video  # 获取显示视频结果按钮
        self.model2 = self.ui.combo2  # 获取视频模型选择下拉框
        self.conf2 = self.ui.conf2  # 获取视频置信度调节控件
        self.conf2.setRange(0.0, 1.0)  # 设置置信度调节范围
        self.conf2.setSingleStep(0.01)  # 设置置信度调节步长
        self.conf2.setValue(0.25)  # 设置置信度默认值
        self.IOU2 = self.ui.IOU2  # 获取视频 IOU 调节控件
        self.IOU2.setRange(0.0, 1.0)  # 设置 IOU 调节范围
        self.IOU2.setSingleStep(0.01)  # 设置 IOU 调节步长
        self.IOU2.setValue(0.45)  # 设置 IOU 默认值
        self.class2 = self.ui.class2  # 获取视频类别输入框
        self.video_path = self.ui.video_path  # 获取视频路径显示框

        # 事件绑定
        self.tab_image.clicked.connect(self.open_image_tab)  # 绑定图像选项卡按钮点击事件
        self.tab_video.clicked.connect(self.open_video_tab)  # 绑定视频选项卡按钮点击事件
        self.model1.currentIndexChanged.connect(self.combo1_change)  # 绑定模型选择下拉框变化事件
        self.select_btn.clicked.connect(self.select_image)  # 绑定选择图像按钮点击事件
        self.show_btn.clicked.connect(self.detect_objects)  # 绑定显示检测结果按钮点击事件
        self.choose_video.clicked.connect(self.select_video_file)  # 绑定选择视频按钮点击事件
        self.show_video.clicked.connect(self.process_and_play_video)  # 绑定显示视频结果按钮点击事件

        # 菜单事件
        self.actionwhite.triggered.connect(self.menu_white)  # 绑定白色主题菜单项触发事件
        self.actionblack.triggered.connect(self.menu_black)  # 绑定黑色主题菜单项触发事件
        self.actionblue.triggered.connect(self.menu_blue)  # 绑定蓝色主题菜单项触发事件
        self.actiondefault.triggered.connect(self.menu_default)  # 绑定默认主题菜单项触发事件
        self.actionintro.triggered.connect(self.menu_intro)  # 绑定介绍菜单项触发事件
        self.actionversion.triggered.connect(self.menu_version)  # 绑定版本菜单项触发事件
        self.actionexit.triggered.connect(self.myexit)  # 绑定退出菜单项触发事件

    def menu_default(self):  # 设置默认主题样式
        stylesheet1 = "QMainWindow{background-color: rgb(240,240,240);}"  # 主窗口背景颜色
        stylesheet2 = "QWidget{background-color: rgb(240,240,240);}"  # 控件背景颜色
        stylesheet3 = "QLabel{color: rgb(20, 120, 80); font: 18pt '黑体'; font-weight: bold;}"  # 标签字体样式
        self.ui.setStyleSheet(stylesheet1)
        self.ui.centralwidget.setStyleSheet(stylesheet2)
        self.ui.tab.setStyleSheet(stylesheet2)
        self.ui.tab_2.setStyleSheet(stylesheet2)
        self.ui.label_11.setStyleSheet(stylesheet3)

    def menu_white(self):  # 设置白色主题样式
        stylesheet1 = "QMainWindow{background-color: rgb(250,250,250);}"
        stylesheet2 = "QWidget{background-color: rgb(250,250,250);}"
        stylesheet3 = "QLabel{color: rgb(20, 120, 80); font: 18pt '黑体'; font-weight: bold;}"
        self.ui.setStyleSheet(stylesheet1)
        self.ui.centralwidget.setStyleSheet(stylesheet2)
        self.ui.tab.setStyleSheet(stylesheet2)
        self.ui.tab_2.setStyleSheet(stylesheet2)
        self.ui.label_11.setStyleSheet(stylesheet3)

    def menu_black(self):  # 设置黑色主题样式
        stylesheet1 = "QMainWindow{background-color: rgb(50,50,50);}"
        stylesheet2 = "QWidget{background-color: rgb(50,50,50);}"
        stylesheet3 = "QLabel{color: rgb(40, 240, 160); font: 18pt '黑体'; font-weight: bold;}"
        self.ui.setStyleSheet(stylesheet1)
        self.ui.centralwidget.setStyleSheet(stylesheet2)
        self.ui.tab.setStyleSheet(stylesheet2)
        self.ui.tab_2.setStyleSheet(stylesheet2)
        self.ui.label_11.setStyleSheet(stylesheet3)

    def menu_blue(self):  # 设置蓝色主题样式
        stylesheet1 = "QMainWindow{background-color: rgb(230,245,255);}"
        stylesheet2 = "QWidget{background-color: rgb(230,245,255);}"
        stylesheet3 = "QLabel{color: rgb(20, 120, 80); font: 18pt '黑体'; font-weight: bold;}"
        self.ui.setStyleSheet(stylesheet1)
        self.ui.centralwidget.setStyleSheet(stylesheet2)
        self.ui.tab.setStyleSheet(stylesheet2)
        self.ui.tab_2.setStyleSheet(stylesheet2)
        self.ui.label_11.setStyleSheet(stylesheet3)

    def menu_intro(self):  # 显示介绍对话框
        dialog = QDialog()  # 创建对话框
        dialog.setWindowTitle('介绍')  # 设置标题
        dialog.setFixedSize(800, 600)  # 设置固定大小
        layout = QHBoxLayout(dialog)  # 创建水平布局
        image_label = QLabel()  # 创建图像标签
        pixmap = QPixmap('image/2.png').scaled(300, 250)  # 加载并缩放图像
        image_label.setPixmap(pixmap)  # 设置图像
        image_label.setAlignment(Qt.AlignCenter)  # 居中对齐
        layout.addWidget(image_label)  # 添加到布局

        font = QFont()  # 创建字体
        font.setPointSize(16)  # 设置字体大小
        font.setBold(True)  # 设置粗体
        font1 = QFont()
        font1.setPointSize(10)
        font1.setBold(True)

        palette_title = QPalette()  # 创建标题调色板
        palette_text = QPalette()  # 创建文本调色板
        palette_title.setColor(QPalette.WindowText, QColor(10, 80, 50))  # 设置标题颜色
        palette_text.setColor(QPalette.WindowText, QColor(30, 180, 80))  # 设置文本颜色

        text_layout = QVBoxLayout()  # 创建垂直布局
        label1 = QLabel("基于YOLOv8的车辆检测")  # 创建标题标签
        label1.setAlignment(Qt.AlignCenter)
        label1.setFont(font)
        label1.setPalette(palette_title)
        label2 = QLabel("支持图像和视频的物体检测")  # 创建描述标签
        label2.setAlignment(Qt.AlignCenter)
        label2.setFont(font1)
        label2.setPalette(palette_text)

        text_layout.addSpacing(100)  # 添加间距
        text_layout.addWidget(label1)
        text_layout.addWidget(label2)
        text_layout.addSpacing(100)
        btn = QPushButton('关闭', dialog)  # 创建关闭按钮
        btn.setFixedSize(100, 40)
        btn.clicked.connect(dialog.close)  # 绑定关闭事件
        text_layout.addWidget(btn, alignment=Qt.AlignCenter)
        layout.addLayout(text_layout)

        dialog.setWindowIcon(QIcon("image/2.png"))  # 设置图标
        dialog.exec_()  # 显示对话框

    def menu_version(self):  # 显示版本对话框
        dialog = QDialog()
        dialog.setWindowTitle('版本')
        dialog.setFixedSize(800, 600)
        layout = QHBoxLayout(dialog)
        image_label = QLabel()
        pixmap = QPixmap("image/2.png").scaled(300, 250)
        image_label.setPixmap(pixmap)
        image_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(image_label)

        font = QFont()
        font.setPointSize(16)
        font.setBold(True)
        font1 = QFont()
        font1.setPointSize(12)
        font1.setBold(True)

        palette_title = QPalette()
        palette_text = QPalette()
        palette_title.setColor(QPalette.WindowText, QColor(10, 80, 50))
        palette_text.setColor(QPalette.WindowText, QColor(30, 180, 80))

        text_layout = QVBoxLayout()
        label1 = QLabel("基于YOLOv8的车辆检测")
        label1.setAlignment(Qt.AlignCenter)
        label1.setFont(font)
        label1.setPalette(palette_title)
        label2 = QLabel("版本: V 1.0")
        label2.setAlignment(Qt.AlignCenter)
        label2.setFont(font1)
        label2.setPalette(palette_text)
        label3 = QLabel("时间: 2024年4月11日")
        label3.setAlignment(Qt.AlignCenter)
        label3.setFont(font1)
        label3.setPalette(palette_text)

        text_layout.addSpacing(100)
        text_layout.addWidget(label1)
        text_layout.addWidget(label2)
        text_layout.addWidget(label3)
        text_layout.addSpacing(100)
        btn = QPushButton('关闭', dialog)
        btn.setFixedSize(100, 40)
        btn.clicked.connect(dialog.close)
        text_layout.addWidget(btn, alignment=Qt.AlignCenter)
        layout.addLayout(text_layout)

        dialog.setWindowIcon(QIcon("image/2.png"))
        dialog.exec_()

    def open_image_tab(self):  # 打开图像处理选项卡
        self.ui.tabWidget.setCurrentIndex(0)  # 设置当前选项卡为图像

    def open_video_tab(self):  # 打开视频处理选项卡
        self.ui.tabWidget.setCurrentIndex(1)  # 设置当前选项卡为视频

    def combo1_change(self, index):  # 模型选择下拉框变化回调
        print(f"你选择了：{self.model1.currentText()}")  # 打印选择的模型

    def select_image(self):  # 选择图像文件
        try:
            file_dialog = QFileDialog()  # 创建文件选择对话框
            file_path, _ = file_dialog.getOpenFileName(self, '打开图像文件', '', 'Images (*.png *.jpg *.bmp)')  # 过滤图像文件
            if file_path:  # 如果选择了文件
                self.image_path = file_path  # 保存路径
                self.load_image()  # 加载图像
        except Exception as e:
            print(f"选择图像文件出错: {e}")

    def load_image(self):  # 加载图像到原始图像显示区域
        try:
            img = QPixmap(self.image_path)  # 加载图像
            if img.isNull():  # 检查图像是否有效
                print(f"无法加载图像: {self.image_path}")
                return
            scene = QGraphicsScene()  # 创建图形场景
            view_size = self.raw_img.size()  # 获取显示区域大小
            scaled_img = img.scaled(view_size.width(), view_size.height(), Qt.KeepAspectRatio)  # 缩放图像
            scene.addPixmap(scaled_img)  # 添加图像到场景
            self.raw_img.setScene(scene)  # 设置场景
            self.raw_img.setSceneRect(QRectF(scaled_img.rect()))  # 使用 QRectF 设置场景边界
        except Exception as e:
            print(f"加载图像出错: {e}")

    def show_image(self):  # 显示检测结果图像
        try:
            img = QPixmap(self.image_path)  # 加载图像
            if img.isNull():  # 检查图像是否有效
                print(f"无法加载检测结果图像: {self.image_path}")
                return
            scene = QGraphicsScene()
            view_size = self.res_img.size()
            scaled_img = img.scaled(view_size.width(), view_size.height(), Qt.KeepAspectRatio)
            scene.addPixmap(scaled_img)
            self.res_img.setScene(scene)
            self.res_img.setSceneRect(QRectF(scaled_img.rect()))  # 使用 QRectF 设置场景边界
        except Exception as e:
            print(f"显示图像出错: {e}")

    def detect_objects(self):
        if not hasattr(self, 'image_path'):
            print('请先选择图像！')
            return
        try:
            conf1 = float("{:.2f}".format(self.conf1.value()))
            IOU1 = float("{:.2f}".format(self.IOU1.value()))
            class1 = int(self.class1.text()) if self.class1.text() else None

            # 加载模型
            model = YOLO('yolo11n.pt') if self.model1.currentText() == '物体检测' else YOLO('yolo11n-seg.pt')
            # 根据模型类型选择预测目录
            predict_base_dir = "runs/detect" if self.model1.currentText() == '物体检测' else "runs/segment"

            # 预测
            results = model.predict(self.image_path, save=True, imgsz=320, conf=conf1, iou=IOU1,
                                    classes=class1, max_det=100)
            file_name = os.path.basename(self.image_path)

            # 动态查找最新预测目录
            predict_dirs = glob.glob(os.path.join(predict_base_dir, "predict*"))
            if not predict_dirs:
                print(f"未找到任何预测目录！路径: {predict_base_dir}")
                return
            latest_predict_dir = max(predict_dirs, key=os.path.getmtime)

            # 检查支持的图像格式
            base_name = os.path.splitext(file_name)[0]
            for ext in ['.jpg', '.png', '.jpeg']:
                file_path = os.path.join(latest_predict_dir, base_name + ext)
                if os.path.exists(file_path):
                    break
            else:
                print(f"检测结果图像未找到: {base_name}（，支持 .jpg, .png, .jpeg）")
                print(f"最新预测目录内容: {os.listdir(latest_predict_dir)}")
                return

            print(f"检测结果图像路径: {file_path}")
            self.image_path = file_path
            self.show_image()
        except Exception as e:
            print(f"图像检测出错: {e}")

    def select_video_file(self):  # 选择视频文件
        try:
            file_dialog = QFileDialog()
            file_path, _ = file_dialog.getOpenFileName(self, '打开视频文件', '', 'Videos (*.mp4 *.avi)')
            if file_path:
                self.video_path = file_path
                self.ui.video_path.setText(file_path)
                self.display_raw_video()  # 显示并播放原始视频
        except Exception as e:
            print(f"选择视频文件出错: {e}")

    def display_raw_video(self):  # 显示并播放原始视频
        try:
            if self.raw_video_cap and self.raw_video_cap.isOpened():
                self.raw_video_cap.release()
                self.raw_video_timer.stop()
            self.raw_video_cap = cv2.VideoCapture(self.video_path)  # 打开视频
            if not self.raw_video_cap.isOpened():
                print("无法打开原始视频文件！")
                return
            ret, frame = self.raw_video_cap.read()  # 读取第一帧
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 转换颜色空间
                h, w, ch = frame.shape
                qimg = QImage(frame.data, w, h, w * ch, QImage.Format_RGB888)  # 转换为 QImage
                pixmap = QPixmap.fromImage(qimg)  # 转换为 QPixmap
                scene = QGraphicsScene()
                view_size = self.raw_video.size()
                scaled_pixmap = pixmap.scaled(view_size.width(), view_size.height(), Qt.KeepAspectRatio)
                scene.addPixmap(scaled_pixmap)
                self.raw_video.setScene(scene)
                self.raw_video.setSceneRect(QRectF(scaled_pixmap.rect()))  # 使用 QRectF 设置场景边界
                # 启动播放
                fps = self.raw_video_cap.get(cv2.CAP_PROP_FPS)
                if fps <= 0:
                    fps = 30  # 默认帧率
                self.raw_video_timer.start(int(1000 / fps))  # 按照帧率启动定时器
            else:
                self.raw_video_cap.release()
                self.raw_video_cap = None
        except Exception as e:
            print(f"显示原始视频出错: {e}")
            if self.raw_video_cap and self.raw_video_cap.isOpened():
                self.raw_video_cap.release()
                self.raw_video_cap = None
            self.raw_video_timer.stop()

    def update_raw_video_frame(self):  # 更新原始视频帧
        try:
            if self.raw_video_cap and self.raw_video_cap.isOpened():
                ret, frame = self.raw_video_cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    h, w, ch = frame.shape
                    qimg = QImage(frame.data, w, h, w * ch, QImage.Format_RGB888)
                    pixmap = QPixmap.fromImage(qimg)
                    scene = QGraphicsScene()
                    view_size = self.raw_video.size()
                    scaled_pixmap = pixmap.scaled(view_size.width(), view_size.height(), Qt.KeepAspectRatio)
                    scene.addPixmap(scaled_pixmap)
                    self.raw_video.setScene(scene)
                    self.raw_video.setSceneRect(QRectF(scaled_pixmap.rect()))
                else:
                    self.raw_video_timer.stop()
                    self.raw_video_cap.release()
                    self.raw_video_cap = None
                    print("原始视频播放结束")
            else:
                self.raw_video_timer.stop()
        except Exception as e:
            print(f"更新原始视频帧出错: {e}")
            self.raw_video_timer.stop()
            if self.raw_video_cap and self.raw_video_cap.isOpened():
                self.raw_video_cap.release()
                self.raw_video_cap = None

    def process_and_play_video(self):  # 处理并播放视频
        if not hasattr(self, 'video_path'):
            print('请先选择视频！')
            return

        try:
            # 停止原始视频播放
            if self.raw_video_cap and self.raw_video_cap.isOpened():
                self.raw_video_cap.release()
                self.raw_video_cap = None
            self.raw_video_timer.stop()

            conf2 = float("{:.2f}".format(self.conf2.value()))  # 获取置信度
            IOU2 = float("{:.2f}".format(self.IOU2.value()))  # 获取 IOU
            class2 = int(self.class2.text()) if self.class2.text() else None  # 获取类别

            model = YOLO('yolo11n.pt') if self.model2.currentText() == '物体检测' else YOLO('yolo11n-seg.pt')  # 加载模型
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                print("无法打开视频文件！")
                return

            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                fps = 30  # 默认帧率，防止除零错误
                print("视频帧率无效，使用默认值 30 FPS")

            file_name = os.path.basename(self.video_path)
            os.makedirs("video_output", exist_ok=True)
            self.output_video_path = f"video_output/{file_name}"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(self.output_video_path, fourcc, fps, (frame_width, frame_height))

            if not out.isOpened():
                print("无法创建输出视频文件！")
                cap.release()
                return

            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break
                results = model.predict(frame, conf=conf2, iou=IOU2, classes=class2, max_det=100)  # 预测
                annotated_frame = results[0].plot(labels=True, conf=True)  # 绘制结果
                out.write(annotated_frame)

            cap.release()
            out.release()

            # 更新视频路径显示为输出路径
            self.ui.video_path.setText(f"检测后的视频保存在 {self.output_video_path}")

            # 显示处理后的视频
            self.video_cap = cv2.VideoCapture(self.output_video_path)
            if not self.video_cap.isOpened():
                print("无法打开处理后的视频文件！")
                return

            # 使用整数时间间隔启动定时器
            self.video_timer.start(int(1000 / fps))  # 按照视频帧率启动定时器，转换为整数
        except Exception as e:
            print(f"视频处理出错: {e}")
            if 'cap' in locals() and cap.isOpened():
                cap.release()
            if 'out' in locals() and out.isOpened():
                out.release()
            if self.video_cap and self.video_cap.isOpened():
                self.video_cap.release()
                self.video_cap = None
            self.video_timer.stop()

    def update_video_frame(self):  # 更新视频帧
        try:
            if self.video_cap and self.video_cap.isOpened():
                ret, frame = self.video_cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    h, w, ch = frame.shape
                    qimg = QImage(frame.data, w, h, w * ch, QImage.Format_RGB888)
                    pixmap = QPixmap.fromImage(qimg)
                    scene = QGraphicsScene()
                    view_size = self.res_video.size()
                    scaled_pixmap = pixmap.scaled(view_size.width(), view_size.height(), Qt.KeepAspectRatio)
                    scene.addPixmap(scaled_pixmap)
                    self.res_video.setScene(scene)
                    self.res_video.setSceneRect(QRectF(scaled_pixmap.rect()))  # 使用 QRectF 设置场景边界
                else:
                    self.video_timer.stop()  # 停止定时器
                    self.video_cap.release()  # 释放视频捕获
                    self.video_cap = None
                    print("视频播放结束")
            else:
                self.video_timer.stop()  # 确保定时器停止
        except Exception as e:
            print(f"更新视频帧出错: {e}")
            self.video_timer.stop()
            if self.video_cap and self.video_cap.isOpened():
                self.video_cap.release()
                self.video_cap = None

    def myexit(self):  # 退出程序
        try:
            self.video_timer.stop()  # 停止定时器
            self.raw_video_timer.stop()  # 停止原始视频定时器
            if hasattr(self, 'video_cap') and self.video_cap and self.video_cap.isOpened():
                self.video_cap.release()  # 释放视频捕获
            if hasattr(self, 'raw_video_cap') and self.raw_video_cap and self.raw_video_cap.isOpened():
                self.raw_video_cap.release()  # 释放原始视频捕获
            cv2.destroyAllWindows()  # 关闭所有 OpenCV 窗口
            sys.exit()  # 退出程序
        except Exception as e:
            print(f"退出程序出错: {e}")
            sys.exit()


if __name__ == "__main__":  # 主程序入口
    app = QApplication(sys.argv)  # 创建应用实例
    win = MyWindow()  # 创建主窗口实例
    win.ui.show()  # 显示主窗口
    app.exec_()  # 进入事件循环