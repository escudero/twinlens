"""TwinLens: photos of an object on a green cloth -> a painted 3D model in Blender.

    python -m twinlens run PHOTOS --work WORK [options]     # everything, resuming where it stopped
    python -m twinlens STEP PHOTOS --work WORK [options]    # one step: prepare cameras depth carve
                                                                 # generate align scene compare

`run` stops once after the masks and asks to go on, so a bad key is caught
before the long steps (--yes skips the question).
"""
import argparse
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from .config import Config

STEPS = ["prepare", "cameras", "depth", "carve", "generate", "align", "scene", "compare"]


def load_profile(path):
    if not path:
        return None
    spec = importlib.util.spec_from_file_location("twinlens_profile", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    print(f"profile: {path}")
    return mod


def done(cfg, step):
    """What each step leaves behind, so `run` can skip it next time."""
    return {
        "prepare": lambda: any(cfg.masks.glob("*.png")),
        "cameras": lambda: (cfg.cameras / "images.bin").exists(),
        "depth": lambda: len(list(cfg.depth.glob("*.npz"))) >= len(list(cfg.images.glob("*.jpg"))) > 0,
        "carve": lambda: cfg.frame.exists() and (cfg.out / "model_carved.ply").exists(),
        "generate": lambda: cfg.gpu == "none" or any(cfg.generated.glob("mv_s*.ply")),
        "align": lambda: (cfg.out / "model.ply").exists(),
        "scene": lambda: (cfg.out / "model.blend").exists(),
        "compare": lambda: (cfg.out / "compare.jpg").exists(),
    }[step]()


def masks_checked(cfg, args):
    """Before the first step after the masks, `run` asks once whether they are
    right. The answer is kept in the work folder; new masks ask again."""
    if cfg.masks_ok.exists():
        return
    if not args.yes:
        print(f"== check the masks in {cfg.out / 'masks_preview.jpg'}")
        print("   the yellow outline should hug the object; dimmed parts were taken as cloth")
        try:
            answer = input("   continue with these masks? [y/N] ").strip().lower()
        except EOFError:  # no terminal to ask in
            print()
            answer = ""
        if answer not in ("y", "yes", "s", "sim"):
            raise SystemExit("stopped after the masks. Run again to continue, or add --yes to skip the question.\n"
                             "To change the key: exg_frac / shade_tol in twinlens/config.py, "
                             "then `twinlens prepare` again.")
    cfg.masks_ok.write_text("masks checked\n")


def do(step, cfg, profile, args):
    print(f"== {step}")
    if step == "prepare":
        from . import prepare
        prepare.run(cfg)
    elif step == "cameras":
        from . import cameras
        cameras.run(cfg)
    elif step == "depth":
        import pycolmap
        from . import depth
        cam = next(iter(pycolmap.Reconstruction(str(cfg.cameras)).cameras.values()))
        depth.run(cfg, cam.params[0])
    elif step == "carve":
        from . import carve
        carve.run(cfg, profile)
    elif step == "generate":
        from . import generate
        generate.run(cfg)
    elif step == "align":
        from . import align
        if align.run(cfg, profile) is None:
            shutil.copy(cfg.out / "model_carved.ply", cfg.out / "model.ply")
            print("  out/model.ply = the carved model")
    elif step == "scene":
        if not shutil.which("blender"):
            raise SystemExit("Blender not found on PATH (needs Blender 4.2+; tested with 5.3)")
        cmd = ["blender", "-b", "-P", str(Path(__file__).parent / "scene.py"), "--",
               "--model", str(cfg.out / "model.ply"), "--out", str(cfg.out), "--stills",
               "--seconds", str(cfg.seconds), "--fps", str(cfg.fps),
               "--width", str(cfg.resolution[0]), "--height", str(cfg.resolution[1])]
        if not args.no_video:
            cmd.append("--video")
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode:
            print(r.stdout[-3000:], r.stderr[-3000:])
            raise SystemExit("Blender failed")
        print(f"  {cfg.out / 'model.blend'}" + ("" if args.no_video else f", {cfg.out / 'turntable.mp4'}")
              + f", stills in {cfg.out / 'stills'}")
    elif step == "compare":
        from . import compare
        compare.run(cfg)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="twinlens", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["run", *STEPS])
    ap.add_argument("photos", type=Path, help="folder with the photos (15-40, all around the object)")
    ap.add_argument("--work", type=Path, required=True, help="work folder for this object")
    ap.add_argument("--gpu", choices=["kaggle", "none"], default="kaggle",
                    help="where Hunyuan3D-2mv runs; 'none' keeps the carved model")
    ap.add_argument("--seeds", default="1,2,3,4", help="generated candidates (comma-separated seeds)")
    ap.add_argument("--pedestal", action="store_true", help="the object stands on a round base: redraw it exactly")
    ap.add_argument("--height-cm", type=float, help="real height, for a model in metres")
    ap.add_argument("--profile", type=Path, help="extra rules for one object (see examples/)")
    ap.add_argument("--no-video", action="store_true", help="scene: stills only (the video takes ~8 s per frame)")
    ap.add_argument("--force", action="store_true", help="run: redo steps that already have results")
    ap.add_argument("--yes", action="store_true", help="run: go on after the masks without asking")
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(line_buffering=True)  # progress shows up even when redirected to a file
    cfg = Config(photos=args.photos, work=args.work, gpu=args.gpu, pedestal=args.pedestal,
                 seeds=tuple(int(s) for s in args.seeds.split(",")), height_cm=args.height_cm,
                 profile=args.profile)
    cfg.work.mkdir(parents=True, exist_ok=True)
    profile = load_profile(args.profile)
    steps = STEPS if args.step == "run" else [args.step]
    for step in steps:
        if args.step == "run" and not args.force and done(cfg, step):
            print(f"== {step}: done (use --force to redo)")
            continue
        if args.step == "run" and step != "prepare":
            masks_checked(cfg, args)
        do(step, cfg, profile, args)


if __name__ == "__main__":
    main()
