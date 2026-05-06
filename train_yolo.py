import os
import yaml
from ultralytics import YOLO
from src.config import NUM_CLASSES_REAL

def create_yolo_yaml():
    """ 
    YOLOv8 ko yêu cầu viết DataLoader, nó chỉ cần DUY NHẤT 1 file dataset.yaml 
    trỏ đến thư mục ảnh và nhãn của chúng ta. Tôi sẽ generate tự động file này.
    """
    data = {
        'path': os.path.abspath('dataset'), # Tự lấy trọn vẹn đường dẫn absolute
        'train': 'train/images',
        'val': 'val/images',
        'nc': NUM_CLASSES_REAL,
        # Tự động render tên class
        'names': [f'Class_Biển_{i}' for i in range(NUM_CLASSES_REAL)]
    }
    
    with open('dataset.yaml', 'w', encoding='utf-8') as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
    print("\n[INFO] Đạt chuẩn! Đã phát sinh cấu hình 'dataset.yaml' cực chuẩn cho YOLO.")

def train_yolov8():
    create_yolo_yaml()
    
    print("\n[INFO] Kéo não bộ pretrained YOLOv8n (Nano) từ Cloud xuống (nhẹ nhất, quét siêu nhanh)...")
    # Bạn có thể thay lỏm 'yolov8n.pt' bằng 'yolov8s.pt' hoặc 'yolov8m.pt' để đánh đổi Tốc độ lấy Độ chính xác.
    model = YOLO('yolov8n.pt') 
    
    print("\n🚀 BẮT ĐẦU HUẤN LUYỆN YOLOv8 EXPERIMENT...")
    
    """
    1 Hàm duy nhất, YOLO tự Wrap:
    + DataLoader (Kèm Mosaic Augmentation cực xịn của riêng bản thân nó)
    + TensorBoard / Logging
    + Optimizer & LR Scheduler
    + Vẽ biểu đồ kết quả (F1-Curve, PR-Curve, Validation loss)
    """
    results = model.train(
        data='dataset.yaml',
        epochs=50,                  # Test thử 50 Epoch
        imgsz=640,                  # Size default YOLO
        batch=16,                   # Batch Size
        name='traffic_sign_yolov8', # Khởi tạo checkpoint name
        amp=True                    # Khởi động Mixed Precision Training tăng tốc VRAM
    )
    print("\n✅ HUẤN LUYỆN YOLOv8 HOÀN TẤT TRONG CHỚP MẮT!")
    print("Mọi lịch sử đào tạo, đồ thị loss và Weights tốt nhất được ĐÓNG GÓI TỰ ĐỘNG ở đường dẫn:")
    print("=> runs/detect/traffic_sign_yolov8/weights/best.pt")

if __name__ == '__main__':
    train_yolov8()
