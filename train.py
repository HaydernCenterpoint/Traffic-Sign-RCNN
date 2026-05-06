import os
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import torch.cuda.amp as amp
from tqdm import tqdm

from src.config import *
from src.dataset import TrafficSignDataset, get_train_transforms, get_val_transforms
from src.model import create_faster_rcnn_model
from src.utils import custom_collate_fn

def train_one_epoch(model, optimizer, dataloader, device, scaler):
    model.train()  # Bật chế độ huấn luyện
    running_loss = 0.0
    
    # Thanh tiến trình chuyên nghiệp
    pbar = tqdm(dataloader, desc="Training")
    
    for images, targets in pbar:
        # Chuyển dữ liệu lên GPU/CPU
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # Đưa dạo hàm về 0
        optimizer.zero_grad()
        
        # Mixed Precision Training: Tối ưu sức mạnh VRAM, chống OOM hiệu quả
        with amp.autocast():
             # PyTorch Faster R-CNN ở chế độ train() trả về dictionary của 4 loss component
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
        
        # Kích hoạt tính đạo hàm và Backpropagation qua scaler
        scaler.scale(losses).backward()
        
        # Gradient Update an toàn
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += losses.item()
        # Cập nhật real-time loss trên thanh bar
        pbar.set_postfix(loss=f"{losses.item():.4f}")
        
    return running_loss / len(dataloader)

@torch.no_grad()
def evaluate_loss(model, dataloader, device):
    """
    Tính trung bình Loss trên tập Validation.
    CHÚ Ý CỐT LÕI: Mọi mạng RCNN trong torchvision sẽ không sinh ra loss 
    nếu chuyển tắt `model.eval()`. Vậy nên phải "lừa" nó bằng `model.train()`
    kèm theo context manager tắt tính gradient `torch.no_grad()`
    """
    model.train() # Hack để nhả ra list các hàm loss thay vì boxes prediction
    
    running_loss = 0.0
    for images, targets in dataloader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        loss_dict = model(images, targets) # Rất gọn, ko lo rò rỉ RAM!
        losses = sum(loss for loss in loss_dict.values())
        running_loss += losses.item()
        
    return running_loss / len(dataloader)

def main():
    print(f"[INFO] Bắt đầu huấn luyện AI với: {DEVICE}")
    print("====== PIPELINE INITIALIZATION ======")
    
    # 1. Bảo vệ sinh dữ liệu: Khai báo check-point folder
    os.makedirs('model_checkpoints', exist_ok=True)
    
    # 2. Chuẩn bị DataLoader
    train_dataset = TrafficSignDataset(TRAIN_IMG_DIR, TRAIN_LBL_DIR, transforms=get_train_transforms())
    val_dataset = TrafficSignDataset(VAL_IMG_DIR, VAL_LBL_DIR, transforms=get_val_transforms())
    
    if len(train_dataset) == 0:
        print("❌ LỖI NGHIÊM TRỌNG: Thư mục train trống trơn! Xin hãy cho database vào /dataset/train/images")
        return
        
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                              collate_fn=custom_collate_fn, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                            collate_fn=custom_collate_fn, num_workers=0, pin_memory=True)
                            
    # 3. Phân bổ Network Model
    model = create_faster_rcnn_model(NUM_CLASSES).to(DEVICE)
    
    # Chỉ bám theo Layer CẦN cập nhật (Loại bỏ các layers bị Freeze)
    params = [p for p in model.parameters() if p.requires_grad]
    
    # 4. Định hình Optimizer & Scheduler
    optimizer = torch.optim.AdamW(params, lr=LEARNING_RATE, weight_decay=1e-4) # AdamW thông minh
    scheduler = StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA) # Giảm LR đứt đoạn để hội tụ sâu
    scaler = amp.GradScaler() # Amp Engine
    
    best_val_loss = float('inf')
    
    # 5. Khởi động kỷ nguyên Training
    for epoch in range(EPOCHS):
        print(f"\n🌊 --- EPOCH [{epoch+1}/{EPOCHS}] ---")
        train_loss = train_one_epoch(model, optimizer, train_loader, DEVICE, scaler)
        
        # Scheduler update (MỖI 1 epoch chứ không dùng cho vòng iterations)
        scheduler.step()
        
        val_loss = evaluate_loss(model, val_loader, DEVICE)
        
        print(f"✅ TRAIN_LOSS: {train_loss:.4f}  |  VAL_LOSS: {val_loss:.4f}  |  LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # 6. Bảo vệ Trí Thông Minh: Auto-Save
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'model_checkpoints/best_model.pth')
            print(f"🏆 CHÚC MỪNG! Đã lưu Best Model với Val Loss mới = {val_loss:.4f}")

if __name__ == '__main__':
    main()
