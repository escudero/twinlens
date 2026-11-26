"""Step 8: the honest check. Render the model in Blender through the cameras of
some photos and put each render next to its photo; score how alike they are
(SSIM inside the object's mask)."""
import json
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import trimesh
from skimage.metrics import structural_similarity

from .geometry import Frame, load_views

SCENE = Path(__file__).parent / "scene.py"


def run(cfg, model=None, n=4):
    rec, views = load_views(cfg)
    frame = Frame(np.load(cfg.frame)["M"])
    model = Path(model or cfg.out / "model.ply")
    height = trimesh.load(model, process=False).vertices[:, 2].max()
    work = cfg.out / "compare"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    pick = [round(i * len(views) / n) for i in range(n)]
    cams = []
    for i in pick:
        v = views[i]
        Ru = v["R"] @ frame.Minv[:3, :3]            # model (upright) -> camera
        tu = v["R"] @ frame.Minv[:3, 3] + v["t"]
        f, cx, cy = v["cam"].params[:3]
        cams.append(dict(name=v["name"], R=Ru.tolist(), C=(-Ru.T @ tu).tolist(), f=float(f), cx=float(cx),
                         cy=float(cy), w=v["cam"].width, h=v["cam"].height))
    spec = work / "cameras.json"
    spec.write_text(json.dumps(dict(height=float(height), cameras=cams)))
    subprocess.run(["blender", "-b", "-P", str(SCENE), "--", "--model", str(model), "--out", str(work),
                    "--photo-views", str(spec)], check=True, capture_output=True)
    rows, scores = [], []
    for i, c in zip(pick, cams):
        v = views[i]
        photo = v["rgb"][:, :, ::-1].copy()
        render = cv2.imread(str(work / "photo_views" / (Path(c["name"]).stem + ".png")))
        render = cv2.resize(render, (photo.shape[1], photo.shape[0]))
        m = v["mask_raw"]
        ys, xs = np.nonzero(m)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        g = [cv2.GaussianBlur(cv2.cvtColor(im[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY), (5, 5), 0) for im in (photo, render)]
        _, smap = structural_similarity(g[0], g[1], data_range=255, full=True)
        scores.append(float(smap[m[y0:y1, x0:x1]].mean()))
        pad = int(0.08 * (y1 - y0))
        box = (slice(max(y0 - pad, 0), y1 + pad), slice(max(x0 - pad, 0), x1 + pad))
        pair = np.hstack([photo[box], render[box]])
        rows.append(cv2.resize(pair, (900, round(900 * pair.shape[0] / pair.shape[1]))))
    h = max(r.shape[0] for r in rows)
    rows = [cv2.copyMakeBorder(r, 0, h - r.shape[0], 0, 0, cv2.BORDER_CONSTANT) for r in rows]
    if len(rows) % 2:
        rows.append(np.zeros_like(rows[0]))
    cv2.imwrite(str(cfg.out / "compare.jpg"), np.vstack([np.hstack(rows[i:i + 2]) for i in range(0, len(rows), 2)]))
    print(f"  photo | render pairs in {cfg.out / 'compare.jpg'}; SSIM on the object {np.round(scores, 3).tolist()}, "
          f"mean {np.mean(scores):.3f}")
    return float(np.mean(scores))
