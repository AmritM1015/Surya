# downstream_examples/common/resize.py
import torch
import torch.nn.functional as F

def resize_ts(ts: torch.Tensor, size: int):
    #  ts: (C, T, H, W)
    C, T, H, W = ts.shape
    x = ts.view(C*T, 1, H,W)

    x = F.interpolate(
        x,
        size = (size, size),
        mode = "bilinear",
        align_corners = False
    )
    return x.view(C, T, size, size)

def resize_mask(mask: torch.Tensor, size:int ):
    # mask: (H, W) or (1, H, W)
    if mask.dim() == 2:
        mask = mask.unsqueeze(0)
    x = mask.unsqueeze(0).float() # (1, 1, H, W)
    x = F.interpolate(x, size=(size, size), mode="nearest")
    return x.squeeze(0)