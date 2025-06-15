import sys
import os
import cv2
import json
import xml.etree.ElementTree as ET
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QFileDialog)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget


class DataVisualizationWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("可视化显示--lflyjmyh")
        self.setGeometry(100, 100, 800, 600)

        # Main widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)

        # Display area
        self.display_label = QLabel()
        self.display_label.setAlignment(Qt.AlignCenter)
        self.display_label.setMinimumSize(640, 480)
        self.display_label.setStyleSheet("background-color: white;")
        self.main_layout.addWidget(self.display_label)

        # Buttons layout
        self.button_layout = QHBoxLayout()
        self.select_button = QPushButton("选择")
        self.select_button.clicked.connect(self.select_file)
        self.play_button = QPushButton("播放")
        self.play_button.clicked.connect(self.play_media)
        self.play_button.setEnabled(False)

        self.button_layout.addWidget(self.select_button)
        self.button_layout.addStretch()
        self.button_layout.addWidget(self.play_button)
        self.main_layout.addLayout(self.button_layout)

        # Media player
        self.media_player = QMediaPlayer()
        self.video_widget = QVideoWidget()
        self.media_player.setVideoOutput(self.video_widget)

        # Variables
        self.current_file = None
        self.is_video = False
        self.labels = None
        self.annotation_format = None  # 'json', 'voc', or 'yolo'
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.cap = None

    def select_file(self):
        file_dialog = QFileDialog(self)
        file_dialog.setNameFilters([
            "Images (*.png *.jpg *.jpeg)",
            "Videos (*.mp4 *.avi)",
            "JSON Labels (*.json)",
            "PASCAL VOC (*.xml)",
            "YOLO Labels (*.txt)"
        ])
        if file_dialog.exec_():
            self.current_file = file_dialog.selectedFiles()[0]
            self.play_button.setEnabled(True)

            # Check file type
            ext = os.path.splitext(self.current_file)[1].lower()
            if ext in ['.png', '.jpg', '.jpeg']:
                self.is_video = False
                self.annotation_format = None
                self.load_image()
            elif ext in ['.mp4', '.avi']:
                self.is_video = True
                self.annotation_format = None
            elif ext == '.json':
                self.annotation_format = 'json'
                self.load_labels()
            elif ext == '.xml':
                self.annotation_format = 'voc'
                self.load_labels()
            elif ext == '.txt':
                self.annotation_format = 'yolo'
                self.load_labels()

    def load_image(self):
        if self.current_file:
            if not os.path.exists(self.current_file):
                print(f"Error: Image file not found at {self.current_file}")
                self.display_label.setText("Error: Image file not found")
                return
            try:
                # Handle non-ASCII paths
                with open(self.current_file, 'rb') as f:
                    img_data = np.frombuffer(f.read(), np.uint8)
                    image = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError("Failed to load image")
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # Load corresponding label if exists
                base_name = os.path.splitext(self.current_file)[0]
                label_files = [
                    (base_name + '.json', 'json'),
                    (base_name + '.xml', 'voc'),
                    (base_name + '.txt', 'yolo')
                ]

                for label_file, fmt in label_files:
                    if os.path.exists(label_file):
                        self.annotation_format = fmt
                        try:
                            if fmt == 'json':
                                with open(label_file, 'r', encoding='utf-8') as f:
                                    self.labels = json.load(f)
                            elif fmt == 'voc':
                                self.labels = self.parse_voc_xml(label_file)
                            elif fmt == 'yolo':
                                self.labels = self.parse_yolo_txt(label_file, image.shape[:2])
                            self.draw_labels(image)
                            break
                        except Exception as e:
                            print(f"Error loading labels from {label_file}: {e}")
                            self.labels = None

                height, width, channel = image.shape
                bytes_per_line = 3 * width
                q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(q_image)
                self.display_label.setPixmap(pixmap.scaled(self.display_label.size(),
                                                           Qt.KeepAspectRatio,
                                                           Qt.SmoothTransformation))
            except Exception as e:
                print(f"Error loading image: {e}")
                self.display_label.setText(f"Error loading image: {str(e)}")
                return

    def parse_voc_xml(self, xml_file):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        objects = []

        for obj in root.findall('object'):
            name = obj.find('name').text
            bbox = obj.find('bndbox')
            box = {
                'class': name,
                'bbox': [
                    int(bbox.find('xmin').text),
                    int(bbox.find('ymin').text),
                    int(bbox.find('xmax').text) - int(bbox.find('xmin').text),
                    int(bbox.find('ymax').text) - int(bbox.find('ymin').text)
                ]
            }
            objects.append(box)

        return {'objects': objects}

    def parse_yolo_txt(self, txt_file, image_shape):
        height, width = image_shape
        objects = []

        with open(txt_file, 'r') as f:
            for line in f:
                values = line.strip().split()
                if len(values) >= 5:
                    class_id = values[0]
                    x_center = float(values[1]) * width
                    y_center = float(values[2]) * height
                    w = float(values[3]) * width
                    h = float(values[4]) * height

                    # Convert to top-left corner format
                    x = int(x_center - w / 2)
                    y = int(y_center - h / 2)
                    w = int(w)
                    h = int(h)

                    objects.append({
                        'class': f'class_{class_id}',
                        'bbox': [x, y, w, h]
                    })

        return {'objects': objects}

    def load_labels(self):
        if self.current_file:
            base_name = os.path.splitext(self.current_file)[0]
            image_file = base_name + '.jpg'
            if not os.path.exists(image_file):
                image_file = base_name + '.png'
            if not os.path.exists(image_file):
                image_file = base_name + '.jpeg'

            if os.path.exists(image_file):
                self.current_file = image_file
                self.is_video = False
                if self.annotation_format == 'json':
                    with open(self.current_file.replace(os.path.splitext(self.current_file)[1], '.json'), 'r') as f:
                        self.labels = json.load(f)
                elif self.annotation_format == 'voc':
                    self.labels = self.parse_voc_xml(
                        self.current_file.replace(os.path.splitext(self.current_file)[1], '.xml'))
                elif self.annotation_format == 'yolo':
                    image = cv2.imread(image_file)
                    self.labels = self.parse_yolo_txt(
                        self.current_file.replace(os.path.splitext(self.current_file)[1], '.txt'), image.shape[:2])
                self.load_image()

    def draw_labels(self, image):
        if self.labels and 'objects' in self.labels:
            for label in self.labels['objects']:
                if 'bbox' in label:
                    x, y, w, h = label['bbox']
                    color = (0, 255, 0)  # Green for bounding box
                    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                    if 'class' in label:
                        cv2.putText(image, label['class'], (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def play_media(self):
        if self.is_video and self.current_file:
            self.cap = cv2.VideoCapture(self.current_file)
            self.timer.start(1000 // 30)  # 30 FPS
        elif self.is_video:
            self.media_player.setMedia(QMediaContent(QUrl.fromLocalFile(self.current_file)))
            self.display_label.setVisible(False)
            self.main_layout.insertWidget(0, self.video_widget)
            self.media_player.play()

    def update_frame(self):
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Apply labels if available
                base_name = os.path.splitext(self.current_file)[0]
                label_files = [
                    (base_name + '.json', 'json'),
                    (base_name + '.xml', 'voc'),
                    (base_name + '.txt', 'yolo')
                ]

                for label_file, fmt in label_files:
                    if os.path.exists(label_file):
                        self.annotation_format = fmt
                        if fmt == 'json':
                            with open(label_file, 'r') as f:
                                self.labels = json.load(f)
                        elif fmt == 'voc':
                            self.labels = self.parse_voc_xml(label_file)
                        elif fmt == 'yolo':
                            self.labels = self.parse_yolo_txt(label_file, frame.shape[:2])
                        self.draw_labels(frame)
                        break

                height, width, channel = frame.shape
                bytes_per_line = 3 * width
                q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(q_image)
                self.display_label.setPixmap(pixmap.scaled(self.display_label.size(),
                                                           Qt.KeepAspectRatio,
                                                           Qt.SmoothTransformation))
            else:
                self.timer.stop()
                self.cap.release()

    def closeEvent(self, event):
        if self.cap:
            self.cap.release()
        self.media_player.stop()
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = DataVisualizationWindow()
    window.show()
    sys.exit(app.exec_())