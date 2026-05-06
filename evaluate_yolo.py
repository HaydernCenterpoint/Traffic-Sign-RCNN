from ultralytics import YOLO

def evaluate_yolov8():
    print("[INFO] Bắt đầu quá trình Evaluation toàn rập cho hệ thống YOLOv8...")
    
    # 1. Tìm quả tim não bộ ngon nhất mà ta vừa train 
    model_path = 'runs/detect/traffic_sign_yolov8/weights/best.pt'
    try:
        model = YOLO(model_path)
        print(f"💡 Đang dùng YOLO Custom Checkpoints tại: {model_path}")
    except Exception as e:
        print(f"❌ LỖI KHÔNG TÌM THẤY WEIGHTS GỐC: Bạn chưa huấn luyện YOLO xong hoặc sai đường dẫn.")
        print("Sẽ dùng thử model gốc pretrained yolov8n.pt để chạy demo.")
        model = YOLO('yolov8n.pt')

    print("\n🚀 QUÉT TOÀN BỘ TẬP DỮ LIỆU ĐỂ LẤY METRICS & TỰ ĐỘNG VẼ BIỂU ĐỒ...")
    # Tính năng Val của YOLO tự động so khớp Predicted vs Target Ground Truth
    # Và quan trọng nhất: Nó tự kết xuất đồ họa!
    metrics = model.val(
        data='dataset.yaml',      # Cấu trúc file nhãn đã tạo
        split='val',              # Đánh giá trên tập Validation
        project='runs/detect',    # Lưu kết quả tại folder nền detect
        name='yolov8_evaluation', # Sinh ra thư mục yolov8_evaluation riêng để dễ tracking so sánh
        conf=0.25,                # Confidence Threshold mặc định để soi các kết quả sát mốc
        iou=0.6                   # NMS IoU threshold lúc đánh giá
    )
    
    print("\n" + "="*50)
    print(" 🏆 KẾT QUẢ ĐÁNH GIÁ ĐỘ PHỦ YOLOV8")
    print("="*50)
    
    # Các metrics chuẩn của COCO format
    print(f"📊 Thuật toán mAP (IoU=0.50:0.95) : {metrics.box.map:.5f}")
    print(f"🎯 Điểm chính xác mAP@0.5          : {metrics.box.map50:.5f}")
    print(f"💥 Điểm mAP@0.75 khắt khe         : {metrics.box.map75:.5f}")
    print("="*50)
    
    # Báo cáo đầu ra biểu đồ cho User
    print("\n🎁 ĐÃ SINH BIỂU ĐỒ TRỰC QUAN GÓI KẾT QUẢ:")
    print("Hệ thống YOLO tự động vẽ sẵn các bản báo cáo:")
    print(" > 1. F1_curve.png (Độ hiệu quả điều chỉnh Precision và Recall)")
    print(" > 2. PR_curve.png (Precision-Recall trade-off giống bài báo khoa học)")
    print(" > 3. confusion_matrix.png (Ma trận nhầm lẫn phát hiện lỗi chéo giữa các biển báo)")
    print(" > 4. Các frame ảnh minh họa trực quan box detect vs box thực.")
    print("=> Vui lòng nhấp vào thư mục: [runs/detect/yolov8_evaluation] để xem biểu đồ và dùng nó cho báo cáo khóa luận/dự án của bạn!")

if __name__ == '__main__':
    evaluate_yolov8()
