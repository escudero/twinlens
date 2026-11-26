"""Settings and the layout of a work folder.

One work folder per object. Every step reads what the previous one wrote
there, so a run can stop and resume at any step.
"""
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    photos: Path                      # the original photos (JPEG/PNG, EXIF kept)
    work: Path                        # everything the pipeline writes

    # 1. prepare
    long_side: int = 2040             # px; photos are resized so the long side is this
    cloth_border: float = 0.06        # the cloth is measured on this strip along each photo's border
    exg_frac: float = 0.4             # background: excess green G - max(R, B) at least this fraction of the cloth's...
    shade_tol: float = 6.0            # ...and shade within this many of the cloth's spreads (its own varies up to ~4)
    min_cloth_exg: float = 10.0       # a border less green than this is not a green cloth: warn
    min_blob_frac: float = 0.002      # drop mask specks smaller than this fraction of the photo

    # 2. cameras
    max_keypoints: int = 4096
    feature_long_side: int = 1600     # px of the object crop fed to ALIKED
    neighbours: int = 9               # match each photo with the next N around the circle
    focal_35mm_fallback: float = 26.0 # when the photos have no EXIF

    # 4. carve
    grid: int = 448                   # voxels along the longest side
    mask_grow_px: int = 3
    smooth_sigma: float = 1.3
    depth_margin_sigma: float = 2.5
    pedestal: bool = False            # replace the bottom with a fitted round pedestal (figurines)

    # 5. generate (Hunyuan3D-2mv)
    gpu: str = "kaggle"               # "kaggle" or "none" (keep the carved model)
    seeds: tuple = (1, 2, 3, 4)
    octree: int = 512

    # 6. align + paint
    relief_smooth: int = 120          # Taubin iterations on the generated mesh (fine relief off)
    top_views: int = 5
    facing_power: int = 6
    color_inset_px: int = 3
    max_disagreement: float = 90.0
    height_cm: float | None = None    # real height, for a model in metres

    # 7. scene
    seconds: int = 8
    fps: int = 30
    resolution: tuple = (1080, 1350)

    profile: Path | None = None       # optional extra rules for one object (see examples/)
    extra: dict = field(default_factory=dict)

    # work folder layout
    @property
    def images(self): return self.work / "images"
    @property
    def masks(self): return self.work / "masks"
    @property
    def masks_ok(self): return self.work / "masks_ok"   # written once the masks were checked
    @property
    def cameras(self): return self.work / "cameras" / "model"
    @property
    def depth(self): return self.work / "depth"
    @property
    def hull_cam(self): return self.work / "hull_cam.ply"
    @property
    def frame(self): return self.work / "frame.npz"
    @property
    def mv_views(self): return self.work / "mv_views"
    @property
    def generated(self): return self.work / "generated"
    @property
    def out(self): return self.work / "out"
