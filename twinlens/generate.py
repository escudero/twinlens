"""Step 5: a clean shape from 4 of the photos with Hunyuan3D-2mv.

The carved model follows the photos' silhouettes but cannot see concavities and
looks lumpy. Hunyuan3D-2mv (Tencent) turns a front, two sides and a back view
into a smooth, closed mesh. It needs a GPU: this step builds the 4 views, sends
them to a free Kaggle T4 as a private dataset + private kernel, waits and
downloads one mesh per seed. Needs `~/.kaggle` credentials (kaggle.com > Settings
> API) and a phone-verified account (GPU + internet in kernels).
"""
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from .geometry import azimuths, load_views
from .paint import despill

SIZE = 1024
MARGIN = 0.6
POLL_S = 30


def make_views(cfg, views):
    """Photos closest to 0, 90, 180 and 270 deg (0 = the first photo), background
    removed with the mask, green spill cleaned, centred 1024 px RGBA squares."""
    cfg.mv_views.mkdir(parents=True, exist_ok=True)
    az = azimuths(views)
    for role, target in (("front", 0), ("side090", 90), ("back", 180), ("side270", 270)):
        i = int(np.argmin(np.abs((az - target + 180) % 360 - 180)))
        v = views[i]
        rgb = v["rgb"].astype(float)
        m = v["mask_raw"].astype(np.uint8) * 255
        clean = despill(rgb.reshape(-1, 3), v["cloth"]).reshape(rgb.shape)
        alpha = cv2.GaussianBlur(cv2.erode(m, np.ones((3, 3), np.uint8)), (5, 5), 0)
        ys, xs = np.nonzero(m)
        cy, cx = (ys.min() + ys.max()) // 2, (xs.min() + xs.max()) // 2
        half = int(max(np.ptp(ys), np.ptp(xs)) * MARGIN)
        pad = half + 10
        big = np.zeros((rgb.shape[0] + 2 * pad, rgb.shape[1] + 2 * pad, 4), np.uint8)
        big[pad:-pad, pad:-pad, :3] = clean.astype(np.uint8)
        big[pad:-pad, pad:-pad, 3] = alpha
        crop = big[cy + pad - half:cy + pad + half, cx + pad - half:cx + pad + half]
        crop = cv2.resize(crop, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(cfg.mv_views / f"{role}.png"), cv2.cvtColor(crop, cv2.COLOR_RGBA2BGRA))
        print(f"  {role}: {v['name']} ({az[i]:.0f} deg)")


def kaggle(*args, check=True):
    exe = shutil.which("kaggle") or str(Path(sys.executable).parent / "kaggle")
    r = subprocess.run([exe, *args], capture_output=True, text=True)
    if check and r.returncode:
        raise SystemExit(f"kaggle {' '.join(args)} failed:\n{r.stdout}\n{r.stderr}")
    return r.stdout + r.stderr


def run(cfg):
    if cfg.gpu == "none":
        print("  --gpu none: skipping; the carved model will be used")
        return
    _, views = load_views(cfg)
    make_views(cfg, views)
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    user = api.get_config_value("username")
    slug = re.sub(r"[^a-z0-9]+", "-", cfg.work.resolve().name.lower()).strip("-")[:40] or "object"
    ds_id, kn_id = f"{user}/twinlens-{slug}-views", f"{user}/twinlens-{slug}-gen"
    stage = cfg.work / "kaggle"
    ds_dir, kn_dir = stage / "dataset", stage / "kernel"
    # a fingerprint of what is sent: if a run was interrupted while waiting, the
    # next one resumes (polls and downloads) instead of pushing everything again
    import hashlib
    h = hashlib.sha1()
    for p in sorted(cfg.mv_views.glob("*.png")):
        h.update(p.read_bytes())
    h.update(repr((list(cfg.seeds), cfg.octree)).encode())
    marker = stage / "pushed.json"
    resume = marker.exists() and json.loads(marker.read_text()).get("hash") == h.hexdigest() and \
        "Status." in kaggle("kernels", "status", kn_id, check=False)
    if resume:
        print(f"  resuming https://www.kaggle.com/code/{kn_id}")
    else:
        for d in (ds_dir, kn_dir):
            shutil.rmtree(d, ignore_errors=True)
            d.mkdir(parents=True)
        for p in cfg.mv_views.glob("*.png"):
            shutil.copy(p, ds_dir)
        (ds_dir / "dataset-metadata.json").write_text(json.dumps(
            {"title": f"twinlens-{slug}-views", "id": ds_id, "licenses": [{"name": "CC0-1.0"}]}))
        if "ready" in kaggle("datasets", "status", ds_id, check=False):
            kaggle("datasets", "version", "-p", str(ds_dir), "-m", "new views")
        else:
            kaggle("datasets", "create", "-p", str(ds_dir))
        for _ in range(40):  # the dataset must be ready before the kernel mounts it
            if "ready" in kaggle("datasets", "status", ds_id, check=False):
                break
            time.sleep(15)
        code = (Path(__file__).parent / "kaggle_kernel.py").read_text()
        code = code.replace("__SEEDS__", repr(list(cfg.seeds))).replace("__OCTREE__", str(cfg.octree))
        (kn_dir / "kernel.py").write_text(code)
        (kn_dir / "kernel-metadata.json").write_text(json.dumps({
            "id": kn_id, "title": f"twinlens-{slug}-gen", "code_file": "kernel.py", "language": "python",
            "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_tpu": False,
            "enable_internet": True, "dataset_sources": [ds_id], "kernel_sources": [],
            "competition_sources": [], "model_sources": [], "machine_shape": "NvidiaTeslaT4"}))
        kaggle("kernels", "push", "-p", str(kn_dir))
        print(f"  pushed https://www.kaggle.com/code/{kn_id} ({len(cfg.seeds)} seeds, ~6 min each + queue)")
        marker.write_text(json.dumps({"kernel": kn_id, "hash": h.hexdigest()}))
    last = ""
    while True:
        time.sleep(POLL_S)
        st = kaggle("kernels", "status", kn_id, check=False)
        state = st.split('"')[-2] if '"' in st else st.strip()
        if state != last:
            print(f"  {time.strftime('%H:%M')} {state}")
            last = state
        if any(k in state for k in ("COMPLETE", "ERROR", "CANCEL")):
            break
    out = stage / "output"
    kaggle("kernels", "output", kn_id, "-p", str(out))
    cfg.generated.mkdir(parents=True, exist_ok=True)
    got = sorted(out.glob("mv_s*.ply"))
    for p in got:
        shutil.copy(p, cfg.generated / p.name)
    print(f"  {len(got)} meshes -> {cfg.generated}")
    if not got:
        log = out / "log.txt"
        raise SystemExit(f"no mesh came back; see {log if log.exists() else out}")
