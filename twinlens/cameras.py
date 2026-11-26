"""Step 2: where each photo was taken from.

ALIKED keypoints on the object crop, LightGlue matches between photos that are
neighbours around the circle, COLMAP's incremental mapper (CPU). Classic SIFT
registered only 8 of 20 photos of a smooth plastic figurine; this registers all.
"""
import shutil
from pathlib import Path

import cv2
import numpy as np
import pycolmap
import torch
from lightglue import ALIKED, LightGlue
from lightglue.utils import numpy_image_to_torch, rbd
from PIL import Image

from .prepare import photos


def exif_focal(path, fallback):
    """35 mm-equivalent focal and digital zoom. Phones record the optical 35 mm
    figure even when digital zoom crops the frame, so both are needed."""
    ex = Image.open(path).getexif().get_ifd(0x8769)
    f35 = float(ex.get(0xA405) or fallback)   # FocalLengthIn35mmFilm
    zoom = float(ex.get(0xA404) or 1.0)       # DigitalZoomRatio
    return f35, (zoom if zoom > 0 else 1.0)


def run(cfg):
    work = cfg.cameras.parent
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    names = sorted(p.name for p in cfg.images.glob("*.jpg"))
    h, w = cv2.imread(str(cfg.images / names[0])).shape[:2]
    f35, zoom = exif_focal(photos(cfg)[0], cfg.focal_35mm_fallback)
    f = f35 * zoom / 43.27 * np.hypot(w, h)
    print(f"  focal prior: {f35:g} mm (35 mm equiv.) x {zoom:g} digital zoom = {f:.0f} px")

    db_path = work / "database.db"
    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = "SIMPLE_RADIAL"
    reader.camera_params = f"{f:.1f},{w / 2},{h / 2},0"
    pycolmap.Database.open(db_path).close()
    pycolmap.import_images(db_path, cfg.images, camera_mode=pycolmap.CameraMode.SINGLE, options=reader)

    torch.set_num_threads(8)
    extractor = ALIKED(max_num_keypoints=cfg.max_keypoints, detection_threshold=0.005).eval()
    matcher = LightGlue(features="aliked", depth_confidence=-1, width_confidence=-1).eval()
    db = pycolmap.Database.open(db_path)
    ids = {img.name: img.image_id for img in db.read_all_images()}
    feats = {}
    for n in names:
        m = cv2.dilate(cv2.imread(str(cfg.masks / (Path(n).stem + ".png")), 0), np.ones((15, 15), np.uint8))
        ys, xs = np.nonzero(m)
        x0, y0 = max(xs.min() - 60, 0), max(ys.min() - 60, 0)
        x1, y1 = min(xs.max() + 60, w), min(ys.max() + 60, h)
        rgb = cv2.imread(str(cfg.images / n))[y0:y1, x0:x1, ::-1]
        s = max(rgb.shape[:2]) / cfg.feature_long_side
        rgb = cv2.resize(rgb, (round(rgb.shape[1] / s), round(rgb.shape[0] / s)), interpolation=cv2.INTER_CUBIC)
        with torch.inference_mode():
            ft = extractor.extract(numpy_image_to_torch(rgb))
        ft["keypoints"] = ft["keypoints"] * s + torch.tensor([x0, y0])
        ft["image_size"] = torch.tensor([[w, h]], dtype=torch.float32)
        kp = ft["keypoints"][0].numpy()
        u, v = np.clip(kp.round().astype(int), 0, [w - 1, h - 1]).T
        keep = torch.from_numpy(m[v, u] > 0)  # only the object votes, never the cloth
        ft = {k: (t[:, keep] if t.ndim > 1 and t.shape[1] == keep.numel() else t) for k, t in ft.items()}
        feats[n] = ft
        db.write_keypoints(ids[n], (ft["keypoints"][0].numpy() + 0.5).astype(np.float32))

    pairs = [(names[i], names[(i + d) % len(names)])
             for i in range(len(names)) for d in range(1, min(cfg.neighbours, len(names) - 1) + 1)]
    pairs = list(dict.fromkeys(tuple(sorted(p)) for p in pairs))
    for a, b in pairs:
        with torch.inference_mode():
            out = rbd(matcher({"image0": feats[a], "image1": feats[b]}))
        db.write_matches(ids[a], ids[b], out["matches"].numpy().astype(np.uint32))
    db.close()
    (work / "pairs.txt").write_text("\n".join(f"{a} {b}" for a, b in pairs))
    pycolmap.verify_matches(db_path, work / "pairs.txt")

    opts = pycolmap.IncrementalPipelineOptions()
    opts.ba_refine_focal_length = True
    recs = pycolmap.incremental_mapping(db_path, cfg.images, work, options=opts)
    best = max(recs.values(), key=lambda r: r.num_reg_images())
    cfg.cameras.mkdir(parents=True, exist_ok=True)
    best.write(cfg.cameras)
    n_reg = best.num_reg_images()
    print(f"  registered {n_reg} / {len(names)} photos, {best.num_points3D()} points, "
          f"reprojection error {best.compute_mean_reprojection_error():.2f} px")
    if n_reg < 0.8 * len(names):
        print("  warning: many photos were not placed. More overlap helps (smaller steps around the object).")
