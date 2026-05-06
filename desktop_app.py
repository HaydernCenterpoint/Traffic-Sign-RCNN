import sys
import os
import cv2
import torch
import torchvision
import numpy as np
import subprocess
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QTabWidget, QTextEdit, 
    QProgressBar, QSlider, QGroupBox, QFormLayout, QLineEdit
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap, QFont, QColor, QPalette

# Configuration imports
from src.config import DEVICE, NUM_CLASSES
from src.model import create_faster_rcnn_model

class TrainingThread(QThread):
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()

    def __init__(self, dataset_dir):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.process = None

    def run(self):
        self.log_signal.emit(f"🚀 Bắt đầu huấn luyện Faster R-CNN với CUDA RTX 4060...")
        
        # Override DATASET_DIR in a temporary way or update config
        # Here we just run train.py. Ensure dataset path is correct before running.
        # Ideally, we should set env var, but train.py uses hardcoded DATASET_DIR
        # To avoid breaking existing code, we run it directly. 
        # (Assuming the user selected the actual dataset that is inside workspace)
        
        env = os.environ.copy()
        # Ensure optimal CUDA settings are visible for RTX 4060
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        env["TRAFFIC_DATASET_DIR"] = self.dataset_dir

        self.process = subprocess.Popen(
            [sys.executable, "train.py"], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env
        )
        
        for line in self.process.stdout:
            self.log_signal.emit(line.strip())
            
        self.process.wait()
        self.finished_signal.emit()

    def stop(self):
        if self.process:
            self.process.terminate()

class InferenceThread(QThread):
    progress_signal = pyqtSignal(QPixmap)
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()

    def __init__(self, media_path, conf_thresh, nms_iou):
        super().__init__()
        self.media_path = media_path
        self.conf_thresh = conf_thresh
        self.nms_iou = nms_iou

    def get_color_by_id(self, class_id):
        np.random.seed(class_id)
        return tuple(np.random.randint(0, 255, 3).tolist())

    def run(self):
        self.log_signal.emit(f"⏳ Đang nạp model Faster R-CNN lên {DEVICE}...")
        model = create_faster_rcnn_model(NUM_CLASSES)
        
        # Load weights
        weight_path = 'model_checkpoints/best_model.pth'
        if not os.path.exists(weight_path):
            self.log_signal.emit("❌ Lỗi: Không tìm thấy model_checkpoints/best_model.pth. Hãy Train trước!")
            self.finished_signal.emit()
            return
            
        model.load_state_dict(torch.load(weight_path, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()

        self.log_signal.emit("✅ Load weights thành công. Bắt đầu dự đoán...")
        
        # Handle Image
        if self.media_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            frame = cv2.imread(self.media_path)
            if frame is None:
                self.log_signal.emit("❌ Lỗi: Không đọc được ảnh.")
                return
                
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_tensor = torch.from_numpy(frame_rgb / 255.0).permute(2, 0, 1).float().unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                outputs = model(img_tensor)[0]
                
            boxes = outputs['boxes'].cpu()
            scores = outputs['scores'].cpu()
            labels = outputs['labels'].cpu()
            
            keep = torchvision.ops.nms(boxes, scores, self.nms_iou)
            
            for idx in keep:
                score = scores[idx].item()
                if score < self.conf_thresh:
                    continue
                    
                box = boxes[idx].numpy().astype(int)
                class_id = labels[idx].item()
                x_min, y_min, x_max, y_max = box
                color = self.get_color_by_id(class_id)
                
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, thickness=3)
                text = f"Class {class_id} | {score*100:.1f}%"
                cv2.putText(frame, text, (x_min, y_min-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Convert frame back to display on PyQt
            frame_display = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame_display.shape
            bytes_per_line = ch * w
            qt_image = QImage(frame_display.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            self.progress_signal.emit(QPixmap.fromImage(qt_image))
            self.log_signal.emit("🎉 Phân tích ảnh hoàn tất.")
        else:
            self.log_signal.emit("⚠️ Tính năng đang xử lý video. Xem lại inference.py cho Output Video...")
            # For brevity, implementing image primarily for GUI feedback.
            # You can adapt the video loop from inference.py here similar to image processing if needed.

        self.finished_signal.emit()

class TrafficSignApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Traffic Sign R-CNN Detector (RTX 4060 Optimized)")
        self.setGeometry(100, 100, 900, 700)
        self.apply_stylesheet()

        # Check CUDA
        cuda_status = f"✅ CUDA Khả dụng ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else "⚠️ Không có CUDA (Chạy bằng CPU)"

        # Main Layout
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        # Tab 1: Huấn luyện
        self.tab_train = QWidget()
        self.setup_training_tab(cuda_status)
        self.tabs.addTab(self.tab_train, "🎯 Huấn luyện Model")

        # Tab 2: Dự đoán
        self.tab_infer = QWidget()
        self.setup_inference_tab()
        self.tabs.addTab(self.tab_infer, "👁️ Nhận diện Biển báo")

    def apply_stylesheet(self):
        # Modern Dark Theme
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e2e;
                color: #cdd6f4;
            }
            QTabWidget::pane {
                border: 1px solid #313244;
                border-radius: 8px;
                background: #181825;
            }
            QTabBar::tab {
                background: #313244;
                color: #a6adc8;
                padding: 10px 20px;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                margin-right: 2px;
                font-weight: bold;
            }
            QTabBar::tab:selected {
                background: #89b4fa;
                color: #11111b;
            }
            QPushButton {
                background-color: #89b4fa;
                color: #11111b;
                border-radius: 6px;
                padding: 8px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #b4befe;
            }
            QPushButton:disabled {
                background-color: #45475a;
                color: #6c7086;
            }
            QTextEdit {
                background-color: #11111b;
                color: #a6e3a1;
                font-family: Consolas, monospace;
                border: 1px solid #313244;
                border-radius: 4px;
                padding: 5px;
            }
            QLabel {
                color: #cdd6f4;
                font-size: 14px;
            }
            QLineEdit {
                background-color: #11111b;
                color: #cdd6f4;
                border: 1px solid #313244;
                border-radius: 4px;
                padding: 5px;
            }
            QGroupBox {
                font-weight: bold;
                border: 1px solid #313244;
                border-radius: 6px;
                margin-top: 10px;
                color: #89b4fa;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)

    def setup_training_tab(self, cuda_status):
        layout = QVBoxLayout()
        
        lbl_status = QLabel(f"💻 Trạng thái GPU: {cuda_status}")
        lbl_status.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        layout.addWidget(lbl_status)
        
        # Dataset Selection
        group_data = QGroupBox("Cấu hình Dữ liệu (Dataset)")
        form_data = QFormLayout()
        
        self.txt_dataset = QLineEdit()
        self.txt_dataset.setText("dataset") # Thư mục mặc định
        btn_browse = QPushButton("📁 Chọn Thư mục")
        btn_browse.clicked.connect(self.choose_dataset_dir)
        
        hbox = QHBoxLayout()
        hbox.addWidget(self.txt_dataset)
        hbox.addWidget(btn_browse)
        form_data.addRow("Thư mục Dataset:", hbox)
        group_data.setLayout(form_data)
        layout.addWidget(group_data)
        
        # Train Action
        self.btn_train = QPushButton("🚀 Bắt đầu Huấn luyện (Train Faster R-CNN)")
        self.btn_train.setMinimumHeight(40)
        self.btn_train.clicked.connect(self.start_training)
        layout.addWidget(self.btn_train)
        
        # Terminal Log
        self.log_console = QTextEdit()
        self.log_console.setReadOnly(True)
        layout.addWidget(QLabel("📝 Terminal Log:"))
        layout.addWidget(self.log_console)
        
        self.tab_train.setLayout(layout)

    def choose_dataset_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Chọn thư mục Dataset chứa 'train' và 'val'")
        if dir_path:
            self.txt_dataset.setText(dir_path)

    def start_training(self):
        self.log_console.clear()
        dataset_path = self.txt_dataset.text()
        
        # Cập nhật đường dẫn trong config.py nếu cần (Ở đây tạm chạy bằng đường dẫn trong project)
        # Vì config.py được import trong code training
        self.log_console.append(f"Chuẩn bị dữ liệu từ: {dataset_path}")
        
        self.btn_train.setEnabled(False)
        self.btn_train.setText("⏳ Đang huấn luyện...")
        
        self.train_thread = TrainingThread(dataset_path)
        self.train_thread.log_signal.connect(self.update_log)
        self.train_thread.finished_signal.connect(self.training_finished)
        self.train_thread.start()

    def update_log(self, text):
        self.log_console.append(text)
        self.log_console.ensureCursorVisible()

    def training_finished(self):
        self.btn_train.setEnabled(True)
        self.btn_train.setText("🚀 Bắt đầu Huấn luyện (Train Faster R-CNN)")
        self.update_log("✅ Hoàn thành quy trình.")

    def setup_inference_tab(self):
        layout = QHBoxLayout()
        
        # Cột điều khiển (Bên trái)
        ctrl_layout = QVBoxLayout()
        
        grp_input = QGroupBox("Load Dữ Liệu Hình Ảnh")
        frm_input = QVBoxLayout()
        self.txt_media = QLineEdit()
        self.txt_media.setPlaceholderText("Đường dẫn ảnh/video...")
        btn_browse_media = QPushButton("📸 Chọn Ảnh")
        btn_browse_media.clicked.connect(self.choose_media_file)
        
        frm_input.addWidget(self.txt_media)
        frm_input.addWidget(btn_browse_media)
        grp_input.setLayout(frm_input)
        
        grp_params = QGroupBox("Bộ Lọc Nhận Diện")
        frm_params = QFormLayout()
        
        self.slider_conf = QSlider(Qt.Orientation.Horizontal)
        self.slider_conf.setRange(10, 100)
        self.slider_conf.setValue(50)
        self.lbl_conf = QLabel("50%")
        self.slider_conf.valueChanged.connect(lambda v: self.lbl_conf.setText(f"{v}%"))
        
        frm_params.addRow("Độ tin cậy (Conf):", self.lbl_conf)
        frm_params.addRow("", self.slider_conf)
        grp_params.setLayout(frm_params)
        
        self.btn_infer = QPushButton("🎯 Phân tích Ngay")
        self.btn_infer.setMinimumHeight(45)
        self.btn_infer.clicked.connect(self.start_inference)
        
        self.infer_log = QTextEdit()
        self.infer_log.setReadOnly(True)
        self.infer_log.setMaximumHeight(150)
        
        ctrl_layout.addWidget(grp_input)
        ctrl_layout.addWidget(grp_params)
        ctrl_layout.addWidget(self.btn_infer)
        ctrl_layout.addWidget(QLabel("📝 Lịch sử:"))
        ctrl_layout.addWidget(self.infer_log)
        ctrl_layout.addStretch()
        
        # Vùng Hiển thị ảnh (Bên phải)
        self.image_display = QLabel("Vùng Ảnh Hiển Thị")
        self.image_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_display.setStyleSheet("background-color: #11111b; border: 2px dashed #45475a; border-radius: 8px;")
        self.image_display.setMinimumSize(500, 400)
        
        layout.addLayout(ctrl_layout, 1)
        layout.addWidget(self.image_display, 3)
        self.tab_infer.setLayout(layout)

    def choose_media_file(self):
        file, _ = QFileDialog.getOpenFileName(self, "Chon hình ảnh", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if file:
            self.txt_media.setText(file)
            pixmap = QPixmap(file)
            self.display_pixmap(pixmap)

    def display_pixmap(self, pixmap):
        scaled_pixmap = pixmap.scaled(self.image_display.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.image_display.setPixmap(scaled_pixmap)

    def start_inference(self):
        media_path = self.txt_media.text()
        if not os.path.exists(media_path):
            self.infer_log.append("❌ File không tồn tại!")
            return
            
        conf = self.slider_conf.value() / 100.0
        
        self.btn_infer.setEnabled(False)
        self.infer_log.append("Đang xử lý mảng Tensor...")
        
        self.infer_thread = InferenceThread(media_path, conf_thresh=conf, nms_iou=0.45)
        self.infer_thread.progress_signal.connect(self.display_pixmap)
        self.infer_thread.log_signal.connect(self.infer_log.append)
        self.infer_thread.finished_signal.connect(lambda: self.btn_infer.setEnabled(True))
        self.infer_thread.start()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = TrafficSignApp()
    window.show()
    sys.exit(app.exec())
