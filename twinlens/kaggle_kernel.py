# TwinLens: Hunyuan3D-2mv shape from 4 photos (front, both sides, back), run on a Kaggle T4.
# This file is a template: generate.py fills in SEEDS and OCTREE and pushes it as a private kernel.
# Input: a private dataset with front.png, side090.png, back.png, side270.png (RGBA, no background).
# Output (/kaggle/working): mv_s<seed>.ply per seed + log.txt
import glob, os, subprocess, sys, time

SEEDS = __SEEDS__
OCTREE = __OCTREE__
T0 = time.time()
LOG = open("/kaggle/working/log.txt", "w")


def log(*a):
    msg = f"[{(time.time() - T0) / 60:6.1f} min] " + " ".join(str(x) for x in a)
    print(msg, flush=True)
    LOG.write(msg + "\n")
    LOG.flush()


def sh(cmd):
    log("$", cmd)
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    LOG.write(r.stdout[-4000:] + "\n" + r.stderr[-4000:] + "\n")
    log("exit", r.returncode)
    return r.returncode


sh("nvidia-smi --query-gpu=name,memory.total --format=csv")
sh("git clone -q --depth 1 https://github.com/Tencent-Hunyuan/Hunyuan3D-2 /tmp/hy3d")
sh(f"{sys.executable} -m pip install -q accelerate einops omegaconf trimesh pymeshlab pygltflib "
   "opencv-python-headless scikit-image ninja")
# Hunyuan3D-2 (Jan 2025) predates the DINOv2 key renames in newer transformers; pin its era
sh(f'{sys.executable} -m pip install -q "transformers<4.50" "diffusers==0.32.2" "huggingface_hub<1.0"')
sh(f"cd /tmp/hy3d && {sys.executable} -m pip install -q --no-deps -e .")
sys.path.insert(0, "/tmp/hy3d")

import torch
from PIL import Image
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

src = os.path.dirname(glob.glob("/kaggle/input/**/front.png", recursive=True)[0])
img = {k: Image.open(f"{src}/{k}.png").convert("RGBA") for k in ("front", "side090", "back", "side270")}
views = {"front": img["front"], "left": img["side090"], "back": img["back"], "right": img["side270"]}
pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
    "tencent/Hunyuan3D-2mv", subfolder="hunyuan3d-dit-v2-mv", variant="fp16")
log("pipeline loaded")
for seed in SEEDS:
    try:
        mesh = pipe(image=views, num_inference_steps=50, octree_resolution=OCTREE,
                    num_chunks=20000, generator=torch.manual_seed(seed), output_type="trimesh")[0]
        mesh.export(f"/kaggle/working/mv_s{seed}.ply")
        log("seed", seed, "verts", len(mesh.vertices), "faces", len(mesh.faces))
    except Exception as e:
        import traceback
        log("FAILED seed", seed, repr(e))
        LOG.write(traceback.format_exc())
    torch.cuda.empty_cache()
log("done")
