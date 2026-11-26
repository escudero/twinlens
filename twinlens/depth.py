"""Step 3: depth per photo with MoGe-2 (CPU is fine, ~15 s per photo).

Silhouettes cannot see concavities; a depth map says which voxels a camera
looked through. Step 4 uses it to carve them away.
"""
from pathlib import Path

import cv2
import numpy as np
import torch

WEIGHTS = "Ruicheng/moge-2-vitl-normal"


def run(cfg, focal_px):
    from moge.model.v2 import MoGeModel
    cfg.depth.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    model = MoGeModel.from_pretrained(WEIGHTS).eval()
    for p in sorted(cfg.images.glob("*.jpg")):
        dst = cfg.depth / (p.stem + ".npz")
        if dst.exists():
            continue
        img = cv2.imread(str(p))[:, :, ::-1]
        m = cv2.imread(str(cfg.masks / (p.stem + ".png")), 0)
        h, w = m.shape
        ys, xs = np.nonzero(m)
        y0, x0 = max(ys.min() - 60, 0), max(xs.min() - 60, 0)
        y1, x1 = min(ys.max() + 60, h), min(xs.max() + 60, w)
        crop = np.ascontiguousarray(img[y0:y1, x0:x1])
        x = torch.tensor(crop / 255.0, dtype=torch.float32).permute(2, 0, 1)
        fov = float(np.degrees(2 * np.arctan(crop.shape[1] / 2 / focal_px)))
        with torch.inference_mode():
            out = model.infer(x, fov_x=fov, use_fp16=False)
        np.savez_compressed(dst, depth=out["depth"].numpy().astype(np.float32), box=np.array([x0, y0, x1, y1]))
        print(f"  {p.name}")
