import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.anchor_utils import AnchorGenerator
import torchvision.models.detection.rpn as rpn

def create_faster_rcnn_model(num_classes):
    """
    Khởi tạo mô hình Faster R-CNN (ResNet50 FPN V2) kèm Transfer Learning
    Được custom để phát hiện được biển báo kích thước cực nhỏ.
    
    Args:
        num_classes: Tổng số lớp thực tế + 1 (Background class).
    Returns:
        model: PyTorch model.
    """
    # 1. Tải kiến trúc Faster RCNN mạnh mẽ nhất có sẵn trong torchvision
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights='DEFAULT')

    # 2. Xử lý giới hạn nhận diện vật thể nhỏ (Vô cùng cần thiết cho Biển báo Giao thông)
    # Default của anchor là ((32,), (64,), (128,), (256,), (512,)) sẽ lọt mất các biển báo nhỏ ở xa
    # Chúng ta thay thế RPN head với các anchor size nhỏ hơn:
    anchor_sizes = ((16,), (32,), (64,), (128,), (256,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    
    anchor_generator = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)
    model.rpn.anchor_generator = anchor_generator
    
    # Do chúng ta sử dụng FPN network, channel đầu ra của Resnet cho phần backbone là 256. 
    # Ta phải cấu hình lại RPN cho khớp với lượng anchors_per_location tạo ra
    out_channels = model.backbone.out_channels
    num_anchors_per_location = anchor_generator.num_anchors_per_location()[0]
    
    model.rpn.head = rpn.RPNHead(
        in_channels=out_channels, 
        num_anchors=num_anchors_per_location
    )

    # 3. Tùy chỉnh khối Classification Predictor (Phân nhóm cuối cùng)
    # Lấy ra số lượng channel in của module phân loại
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # Thay mới hoàn toàn khối classification để hợp với số class chúng ra yêu cầu (num_classes)
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model
