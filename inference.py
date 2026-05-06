import cv2
import torch
import torchvision
import numpy as np
import time

from src.config import *
from src.model import create_faster_rcnn_model

def get_color_by_id(class_id):
    """ Map màu cố định cho từng Class BIỂN BÁO với random seed siêu tiện lợi """
    np.random.seed(class_id)
    return tuple(np.random.randint(0, 255, 3).tolist())

def run_prediction_pipeline(video_in, video_out, conf_thresh=0.6, nms_iou=0.45):
    """
    Video Inference Pipeline với Non-Maximum Suppression (NMS)
    Giúp dọn dẹp các hộp bao trùng lặp một cách gọn gàng nhất.
    """
    print(f"📽️ Đang khởi động R-CNN Predictor, Load video [{video_in}]...")
    
    # Setup Engine
    model = create_faster_rcnn_model(NUM_CLASSES)
    try:
        model.load_state_dict(torch.load('model_checkpoints/best_model.pth', map_location=DEVICE))
    except Exception as e:
        print(f"❌ AI FAILS: Không tìm thấy Weights! {e}")
        return
        
    model.to(DEVICE)
    model.eval()

    # Nhúng OpenCV đọc stream File/Webcam
    cap = cv2.VideoCapture(video_in)
    if not cap.isOpened():
        print(f"❌ LỖI VÀO: Path {video_in} sai hoặc lỗi định dạng.")
        return
        
    # Tính toán thông số metadata góc của Video
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Thiết lập bộ ghi luồng ra (Mp4 Codec)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(video_out, fourcc, fps, (width, height))

    frame_count = 0
    start_global_time = time.time()
    
    print("🚀 Bắt đầu Stream Xử lý hình ảnh Real-time...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Chuẩn bị Tensor (Faster RCNN nuốt format float [0., 1.])
        # Và shape phải là [Batch=1, C, H, W]
        img_tensor = torch.from_numpy(frame_rgb / 255.0).permute(2, 0, 1).float().unsqueeze(0).to(DEVICE)
        
        t0 = time.time()
        with torch.no_grad(): # Bắt buộc phải có trong Inference!
            outputs = model(img_tensor)[0]
        t1 = time.time()
        
        # Bóc tách
        boxes = outputs['boxes'].cpu()
        scores = outputs['scores'].cpu()
        labels = outputs['labels'].cpu()
        
        # 👑 BƯỚC NĂNG LƯỢNG CAO (NMS)
        # Faster RCNN thi thoảng vẽ đống Hộp chồng lên nhau, đây là chìa khóa.
        keep_indices = torchvision.ops.nms(boxes, scores, nms_iou)
        
        # Filter Render Graph
        for idx in keep_indices:
            score = scores[idx].item()
            if score < conf_thresh:
                continue
                
            # Trích xuất tọa độ an toàn (Ép kiểu sang viền Int màn hình)
            box = boxes[idx].numpy().astype(int)
            class_id = labels[idx].item()
            
            x_min, y_min, x_max, y_max = box
            color = get_color_by_id(class_id)
            
            # Vẽ HCN
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, thickness=2)
            
            # Vẽ Biển chứa Text
            text = f"Class {class_id} | {score*100:.1f}%"
            # Tạo nền đen mờ cho chữ dễ đọc
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.5, 1)
            cv2.rectangle(frame, (x_min, y_min-20), (x_min+tw+5, y_min), color, -1)
            
            cv2.putText(frame, text, (x_min+3, y_min-5), 
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)
                        
        # Đóng mộc FPS thời gian thực
        fps_read = int(1.0 / (t1 - t0))
        cv2.putText(frame, f"FPS: {fps_read}", (20, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 3)

        out_video.write(frame)
        frame_count += 1
        
        if frame_count % 30 == 0:
            print(f"[Real-time] Đã quét {frame_count} frames | Frame rate nội mảng: {fps_read} FPS")

    # Ký quỹ bộ nhớ
    cap.release()
    out_video.release()
    total_time = time.time() - start_global_time
    print(f"\n🎉 XUẤT XƯỞNG! Đã cứu sống thuật toán ra File: [{video_out}]!")
    print(f"Tổng thời gian hoạt động: {total_time:.2f} giây.")

if __name__ == '__main__':
    # Hướng dẫn chạy nhanh: Thay 'test.mp4' bằng video giao thông của bạn 
    run_prediction_pipeline("test_video.mp4", "output_rendered.mp4", conf_thresh=0.5, nms_iou=0.45)
