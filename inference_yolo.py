import os
import cv2
import urllib.request
from ultralytics import YOLO

try:
    # Thư viện ma thuật của Jupyter/Colab để xoá và in đè ảnh tạo hiệu ứng Video Player
    from IPython.display import display, clear_output, Image
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

def download_sample_video(video_path="test_video.mp4"):
    """ Tải video mẫu giao thông tự động từ Internet (Github Intel) nếu chưa có """
    if not os.path.exists(video_path):
        print(f"📥 Hệ thống đang tự tải video giao thông mẫu về đường dẫn: {video_path}...")
        url = "https://github.com/intel-iot-devkit/sample-videos/raw/master/car-detection.mp4"
        try:
            urllib.request.urlretrieve(url, video_path)
            print("✅ Tải video hoàn tất!")
        except Exception as e:
            print(f"❌ Lỗi tải video: {e}")

def stream_inference_colab(video_path):
    # 1. Pipeline tự download test data
    download_sample_video(video_path)
    
    # 2. Khởi tạo Neural Network
    model_path = 'runs/detect/traffic_sign_yolov8/weights/best.pt'
    if os.path.exists(model_path):
        print(f"💡 Đang nạp Model Custom của bạn: {model_path}")
        model = YOLO(model_path)
    else:
        print(f"⚠️ Không tìm thấy Weights! Trở về dùng AI chung yolov8n.pt...")
        model = YOLO('yolov8n.pt')

    # 3. Đọc dữ liệu thô
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Lỗi: OpenCV không thể giải mã video {video_path}")
        return

    print("🚀 Bắt đầu Stream Bounding Box trực tiếp tại Terminal Output (Colab Mode)...")
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        # YOLOv8 Predict cho TỪNG ẢNH (Không dùng chế độ xuất video)
        # verbose=False để thanh tiến trình terminal không in rác text log dư thừa
        results = model.predict(frame, conf=0.5, iou=0.45, verbose=False)
        
        # Hàm .plot() tích hợp sẵn của YOLO vẽ luôn Bbox và Percent % lên ma trận ảnh
        annotated_frame = results[0].plot() 
        
        if IN_COLAB:
            # Trick hiển thị Video ngay trên Output Cell của Colab
            clear_output(wait=True) # Xóa ngay ảnh của Frame cũ
            
            # Encode ma trận CV2 sang chuẩn ảnh nén để chèn vào trình duyệt
            _, buffer = cv2.imencode('.png', annotated_frame)
            display(Image(data=buffer.tobytes()))
            print(f"🎥 Đang stream trực tiếp: Frame {frame_count}")
        else:
            # Fallback nếu chạy ở Local Terminal máy tính thường
            cv2.imshow("YOLOv8 Colab-Style Stream", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    if not IN_COLAB:
        cv2.destroyAllWindows()
    
    print("\n🎉 STREAM KẾT THÚC! Bạn đã theo dõi hệ thống YOLO nhận diện thời gian thực thành công.")

if __name__ == '__main__':
    stream_inference_colab('test_video.mp4')
