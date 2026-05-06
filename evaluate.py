import torch
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm

from src.config import *
from src.dataset import TrafficSignDataset, get_val_transforms
from src.utils import custom_collate_fn
from src.model import create_faster_rcnn_model

def evaluate_metrics():
    print(f"[INFO] Scanning Dataset Test & Bắt đầu tính mAP trên VRAM: {DEVICE}")
    
    val_dataset = TrafficSignDataset(VAL_IMG_DIR, VAL_LBL_DIR, transforms=get_val_transforms())
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=custom_collate_fn)
    
    # Trích xuất kiến trúc Network trống
    model = create_faster_rcnn_model(NUM_CLASSES)
    
    # Lấy thông số não bộ từ Checkpoints
    try:
        model.load_state_dict(torch.load('model_checkpoints/best_model.pth', map_location=DEVICE))
        print("💡 Đã load checkpoints 'best_model.pth' thành công!")
    except Exception as e:
        print(f"❌ LỖI QUAN TRỌNG: File Model không tồn tại hoặc bị lỗi cấu trúc! {e}")
        return
        
    model.to(DEVICE)
    model.eval() # Bắt buộc phải là EVAL() để engine nhả ra output hộp Bounding Box
    
    # Chuẩn bị Metrics đánh giá chuẩn quốc tế mAP
    metric = MeanAveragePrecision(iou_type="bbox")
    
    pbar = tqdm(val_loader, desc="Testing")
    for images, targets in pbar:
        # CPU/GPU align
        images = list(image.to(DEVICE) for image in images)
        
        # No_grad là cốt tử để không làm RAM nổ trong Inference Phase
        with torch.no_grad():
            outputs = model(images)
            
        # TorchMetrics mong đợi cấu trúc Dictionary List phân cực rành rọt
        preds = []
        for out in outputs:
            preds.append({
                'boxes': out['boxes'].cpu(),
                'scores': out['scores'].cpu(),
                'labels': out['labels'].cpu()
            })
            
        targs = []
        for t in targets:
            targs.append({
                'boxes': t['boxes'].cpu(),
                'labels': t['labels'].cpu()
            })
            
        metric.update(preds, targs)
        
    # In báo cáo bảng Metrics ra màn hình Terminal
    print("\n" + "="*40)
    print(" KẾT QUẢ ĐÁNH GIÁ METRICS - GIAO THÔNG")
    print("="*40)
    
    results = metric.compute()
    print(f"📊 mAP (IoU=0.50:0.95) : {results['map'].item():.5f}")
    print(f"🎯 mAP@0.5 (Tốt nhất)   : {results['map_50'].item():.5f}")
    print(f"💥 mAP@0.75             : {results['map_75'].item():.5f}")
    print(f"👀 mAR (IoU=0.50:0.95)   : {results['mar_100'].item():.5f}")
    print("="*40)

if __name__ == '__main__':
    evaluate_metrics()
