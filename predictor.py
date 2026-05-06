import cv2
import torch
import torchvision
import numpy as np
import time
import os

from src.config import *
from src.model import create_faster_rcnn_model

model = None

def load_system_model(model_path='model_checkpoints/best_model.pth'):
    global model
    print(f"Loading model from {model_path}...")
    model = create_faster_rcnn_model(NUM_CLASSES)
    try:
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        print(f"Model loaded successfully from {model_path}.")
    except Exception as e:
        print(f"Warning: Could not load model from {model_path}: {e}")
    model.to(DEVICE)
    model.eval()
    return True

# Load initially
load_system_model()

def get_color_by_id(class_id):
    np.random.seed(class_id)
    return tuple(np.random.randint(0, 255, 3).tolist())

def process_frame(frame, conf_thresh=0.6, nms_iou=0.45):
    """Processes a single BGR frame and returns the annotated frame."""
    if model is None:
        return frame
        
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(frame_rgb / 255.0).permute(2, 0, 1).float().unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        outputs = model(img_tensor)[0]
        
    boxes = outputs['boxes'].cpu()
    scores = outputs['scores'].cpu()
    labels = outputs['labels'].cpu()
    
    keep_indices = torchvision.ops.nms(boxes, scores, nms_iou)
    
    for idx in keep_indices:
        score = scores[idx].item()
        if score < conf_thresh:
            continue
            
        box = boxes[idx].numpy().astype(int)
        class_id = labels[idx].item()
        
        x_min, y_min, x_max, y_max = box
        color = get_color_by_id(class_id)
        
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, thickness=2)
        
        text = f"Class {class_id} | {score*100:.1f}%"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.5, 1)
        cv2.rectangle(frame, (x_min, max(0, y_min-20)), (x_min+tw+5, y_min), color, -1)
        cv2.putText(frame, text, (x_min+3, y_min-5), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)
                    
    return frame

def predict_image(image_path, output_path, conf_thresh=0.6):
    img = cv2.imread(image_path)
    if img is None:
        return False
    
    result_img = process_frame(img, conf_thresh)
    cv2.imwrite(output_path, result_img)
    return True

def predict_video(video_path, output_path, conf_thresh=0.6):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # H264 for web
    out_video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        result_frame = process_frame(frame, conf_thresh)
        out_video.write(result_frame)
        
    cap.release()
    out_video.release()
    return True
