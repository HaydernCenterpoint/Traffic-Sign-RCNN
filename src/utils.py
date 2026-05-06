import torch

def custom_collate_fn(batch):
    """
    Hàm custom_collate_fn dùng cho DataLoader.
    Mặc định DataLoader sẽ stack (xếp chồng) các tensor lại với nhau. Tuy nhiên, 
    ảnh object detection có số lượng bounding box khác nhau (shapes khác nhau),
    gây ra lỗi stack. Hàm này sẽ gói chúng thành một tuple để tránh lỗi.
    """
    return tuple(zip(*batch))
