import os
import torch
import cv2
from src.dataset import TrafficSignDataset, get_train_transforms
from src.utils import custom_collate_fn
from src.model import create_faster_rcnn_model
from src.config import NUM_CLASSES
from torch.utils.data import DataLoader

def auto_test_pipeline():
    os.makedirs('dataset/train/images', exist_ok=True)
    os.makedirs('dataset/train/labels', exist_ok=True)
    
    # Tạo 1 file ảnh giả
    dummy_img = "dataset/train/images/test_001.jpg"
    cv2.imwrite(dummy_img, torch.randint(0, 255, (480, 640, 3)).numpy())
    
    # Tạo 1 file YOLO label giả
    dummy_lbl = "dataset/train/labels/test_001.txt"
    with open(dummy_lbl, 'w') as f:
        # class_id, x_center, y_center, w, h
        f.write("0 0.5 0.5 0.1 0.1\n")
        f.write("14 0.2 0.3 0.05 0.08\n")
        
    print("[1] Data dummy đã tạo.")
    
    # Test Dataset
    dataset = TrafficSignDataset('dataset/train/images', 'dataset/train/labels', transforms=get_train_transforms())
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=custom_collate_fn)
    
    for images, targets in dataloader:
        print("[2] Dataset shape check:")
        print(f" - Image shape: {images[0].shape}")
        print(f" - BBox shape: {targets[0]['boxes'].shape}")
        print(f" - Labels: {targets[0]['labels']}")
        print(f" - Area check OK? {targets[0]['area'].shape[0] > 0}")
        
        # OOM/Tensor Mismatch check với Model
        model = create_faster_rcnn_model(num_classes=NUM_CLASSES)
        model.train()
        
        # Test pass forward
        try:
            # Model expect list of images tensors, list of targets dictionaries
            loss_dict = model(list(images), list(targets))
            print("\n[3] Model Pass OK. Loss dictionary thu được:")
            for k, v in loss_dict.items():
                print(f" - {k}: {v.item():.4f}")
            print("\n✅ KIỂM TRA PIPELINE THÀNH CÔNG! KHÔNG CÓ LỖI SHAPE!")
        except Exception as e:
            print("\n❌ LỖI TRONG MODEL PASS:")
            import traceback
            traceback.print_exc()
        
        break

if __name__ == '__main__':
    auto_test_pipeline()
