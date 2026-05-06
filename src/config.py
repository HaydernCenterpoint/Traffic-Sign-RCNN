import os
import torch

# Số lượng class thực tế chưa tính background (Giả định YOLO TXT, ví dụ 43 class cho GTSRB)
NUM_CLASSES_REAL = 43  
# Cộng 1 cho class Background theo cấu trúc của Faster R-CNN
NUM_CLASSES = NUM_CLASSES_REAL + 1

# Các tham số huấn luyện
BATCH_SIZE = 4
EPOCHS = 20
LEARNING_RATE = 1e-4
STEP_SIZE = 5
GAMMA = 0.1

# Đường dẫn dữ liệu (Hỗ trợ nạp động bằng biến môi trường từ Desktop App)
DATASET_DIR = os.environ.get("TRAFFIC_DATASET_DIR", "dataset")
TRAIN_IMG_DIR = os.path.join(DATASET_DIR, "train", "images")
TRAIN_LBL_DIR = os.path.join(DATASET_DIR, "train", "labels")
VAL_IMG_DIR = os.path.join(DATASET_DIR, "val", "images")
VAL_LBL_DIR = os.path.join(DATASET_DIR, "val", "labels")

# Device configuration (Hỗ trợ CUDA và CPU tự động)
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

if DEVICE.type == 'cuda':
    # Kích hoạt CUDNN benchmark để tối ưu tốc độ convolution kernels trên RTX 4060
    torch.backends.cudnn.benchmark = True
    # RTX 40 series (Ada Lovelace) hỗ trợ TensorFloat-32 (TF32) rất tốt
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
