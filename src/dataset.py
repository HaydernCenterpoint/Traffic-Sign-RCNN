import os
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

class TrafficSignDataset(Dataset):
    def __init__(self, img_dir, lbl_dir, transforms=None):
        self.img_dir = img_dir
        self.lbl_dir = lbl_dir
        self.transforms = transforms
        
        # Lọc danh sách file ảnh hợp lệ (tránh file rác)
        valid_ext = ('.jpg', '.jpeg', '.png', '.bmp')
        if not os.path.exists(self.img_dir):
            self.imgs = []
        else:
            self.imgs = [img for img in os.listdir(self.img_dir) if img.lower().endswith(valid_ext)]
        
    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_name = self.imgs[idx]
        img_path = os.path.join(self.img_dir, img_name)
        
        # Đọc ảnh OpenCV (BGR -> RGB)
        image = cv2.imread(img_path)
        if image is None:
            # Bắt lỗi đọc ảnh hỏng, tránh crash hệ thống DataLoader
            print(f"CẢNH BÁO: Không thể đọc ảnh {img_path}. Thay thế bằng ảnh trống.")
            image = np.zeros((640, 640, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        height, width, _ = image.shape
        
        # Mapping tên ảnh với tên label TXT tương ứng (định dạng YOLO)
        lbl_name = os.path.splitext(img_name)[0] + '.txt'
        lbl_path = os.path.join(self.lbl_dir, lbl_name)
        
        boxes = []
        labels = []
        
        try:
            if os.path.exists(lbl_path):
                with open(lbl_path, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) != 5:
                            continue  # Bỏ qua dòng lỗi format
                        
                        class_id = int(parts[0])
                        class_id += 1 # Faster R-CNN quy định id=0 là Background
                        
                        x_center, y_center, w, h = map(float, parts[1:])
                        
                        # Cực kỳ quan trọng: Transform YOLO normalize sang Pascal VOC absolute [x_min, y_min, x_max, y_max]
                        x_min = (x_center - w / 2) * width
                        y_min = (y_center - h / 2) * height
                        x_max = (x_center + w / 2) * width
                        y_max = (y_center + h / 2) * height
                        
                        # Clip để box không bị ra ngoài ảnh
                        x_min, x_max = max(0, x_min), min(width, x_max)
                        y_min, y_max = max(0, y_min), min(height, y_max)
                        
                        if x_max > x_min and y_max > y_min:
                            boxes.append([x_min, y_min, x_max, y_max])
                            labels.append(class_id)
        except Exception as e:
            print(f"CẢNH BÁO: Lỗi đọc/nhận dạng file label {lbl_path} - {e}")
            
        # Áp dụng Albumentations transformations
        if self.transforms:
            try:
                # ToFloat(max_value=255) trong transform đã quy mô ảnh về [0, 1]
                transformed = self.transforms(image=image, bboxes=boxes, class_labels=labels)
                image = transformed['image'] # Dạng tensor [C, H, W]
                boxes = transformed['bboxes']
                labels = transformed['class_labels']
            except Exception as e:
                print(f"CẢNH BÁO: Lỗi transform trên ảnh {img_path} - {e}")
                # Fallback scale thủ công nếu transform thất bại
                image = image.astype(np.float32) / 255.0
                image = torch.from_numpy(image).permute(2, 0, 1)

        else:
            image = image.astype(np.float32) / 255.0
            image = torch.from_numpy(image).permute(2, 0, 1)

        # Chuyển đổi định dạng chuẩn bị Tensor
        if len(boxes) == 0:
            # Fallback nếu không có nhãn hoặc bị xóa sau data augmentation
            boxes_tensor = torch.empty((0, 4), dtype=torch.float32)
            labels_tensor = torch.empty((0,), dtype=torch.int64)
            area = torch.empty((0,), dtype=torch.float32)
            iscrowd = torch.empty((0,), dtype=torch.uint8)
        else:
            boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32)
            labels_tensor = torch.as_tensor(labels, dtype=torch.int64)
            # Area giúp đánh giá kích thước metrics
            area = (boxes_tensor[:, 3] - boxes_tensor[:, 1]) * (boxes_tensor[:, 2] - boxes_tensor[:, 0])
            iscrowd = torch.zeros((len(boxes),), dtype=torch.uint8)

        image_id = torch.tensor([idx])

        # Tạo format theo đúng yêu cầu torchvision (Dict)
        target = {}
        target["boxes"] = boxes_tensor
        target["labels"] = labels_tensor
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        return image, target

def get_train_transforms():
    """ 
    Hàm Augmentation cho tập Training.
    CHÚ Ý: TUYỆT ĐỐI KHÔNG dùng HorizontalFlip (Lật ngang) vì sẽ làm hỏng ý nghĩa của biển báo
    ví dụ: Rẽ trái chuyển thành rẽ phải, cấm đi ngược chiều bị đảo lộn.
    """
    return A.Compose([
        A.RandomBrightnessContrast(p=0.5), # Tăng giảm sáng/tuơng phản để quen với điều kiện thời tiết
        A.GaussNoise(p=0.3),               # Giả lập nhiễu camera kém
        A.ToFloat(max_value=255.0),        # Scale ảnh từ [0,255] sang [0.0, 1.0] (float32) yêu cầu của model
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'], min_area=0.0, min_visibility=0.0))

def get_val_transforms():
    """ Dành cho vòng lặp Validation / Test """
    return A.Compose([
        A.ToFloat(max_value=255.0),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))
