"""Method-neutral patch removal, shared by the faithfulness metrics and the closed-loop test.

`inpaint` (default) fills removed pixels by push-pull interpolation (Gortler et al.
1996) from the pixels that remain: smooth, uses nothing from the removed pixels (unlike
blurring, which leaks them and is BlurIG's baseline), and shares no baseline with any
method (grey is IG (bg)'s baseline). `gray` pastes RDT's "missing camera" colour, the
pilot 2 protocol. The fill changes the deletion ranking of methods but not the curve
overshoot (D/D_full peaks at ~2 under both; see metrics.py).

With every pixel removed there is nothing to interpolate from and the result is 0, so
callers work relative to the grey image and full removal stays RDT's "missing camera".
"""
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

FILLS = ("inpaint", "gray")


def inpaint(x, remove):
    """x: (N, C, H, W); remove: bool (N or 1, 1, H, W), True = remove.
    Returns x with removed pixels interpolated from kept ones; fully removed images become 0."""
    dtype = x.dtype
    w = (~remove).float().expand(x.shape[0], 1, *x.shape[-2:])
    c = x.float() * w
    levels = []
    while min(c.shape[-2:]) > 1:                                   # push: weighted means down the pyramid
        levels.append((c, w))
        ws = F.avg_pool2d(w, 2, ceil_mode=True)
        c = F.avg_pool2d(c * w, 2, ceil_mode=True) / ws.clamp_min(1e-12)
        w = (4 * ws).clamp(max=1)
    c = c * (w > 0)
    for cl, wl in reversed(levels):                                # pull: fill gaps from the coarser level
        c = wl * cl + (1 - wl) * F.interpolate(c, size=cl.shape[-2:], mode="bilinear", align_corners=False)
    return c.to(dtype)


def remove_pixels(x, remove, bg, fill="inpaint", chunk=32):
    """x: (C, H, W) image, remove: (J, 1, H, W) bool, bg: (C, H, W) grey image in the same space.
    Returns (J, C, H, W): x with each mask's pixels removed."""
    if fill == "gray":
        return torch.where(remove, bg, x)
    if fill != "inpaint":
        raise ValueError(f"fill must be one of {FILLS}, got {fill!r}")
    out = []
    for i in range(0, remove.shape[0], chunk):
        r = remove[i:i + chunk]
        out.append(inpaint((x - bg).expand(r.shape[0], -1, -1, -1), r) + bg)
    return torch.cat(out)


def remove_from_image(img, mask_img, color, fill="inpaint"):
    """PIL frame with the white pixels of `mask_img` removed (grey colour or inpainted)."""
    if img is None:
        return None
    img = img.convert("RGB")
    m = mask_img.resize(img.size, Image.NEAREST)
    if fill == "gray":
        out = img.copy()
        out.paste(Image.new("RGB", out.size, color), mask=m)
        return out
    x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)
    bg = torch.tensor(color, dtype=torch.float32)[:, None, None].expand_as(x) / 255.0
    r = torch.from_numpy(np.asarray(m) > 127)[None, None]
    y = remove_pixels(x, r, bg, fill)[0]
    return Image.fromarray((y.permute(1, 2, 0).clamp(0, 1).numpy() * 255).round().astype(np.uint8))
