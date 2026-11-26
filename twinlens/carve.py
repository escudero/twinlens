"""Step 4: carve the object out of a voxel grid.

Silhouettes first (a voxel outside the green-key mask in any photo is not
object), then depth (a voxel clearly in front of the surface a photo sees is
air). The result is a complete painted model on its own (out/model_carved.ply)
and the reference the generated model is aligned to (step 6). This step also
fixes the upright frame and finds the table.
"""
import cv2
import numpy as np
import trimesh
from scipy import ndimage
from skimage import measure

from . import paint, pedestal
from .geometry import Frame, floor_height, load_views, project

BOX_MARGIN = 0.25
DEPTH_MIN_VOTES = 2
DEPTH_MARGIN_VOX = 3
DEPTH_EDGE_PX = 10
DEPTH_SCALE = 0.5


def initial_box(rec, views):
    pts = np.array([p.xyz for p in rec.points3D.values()])
    hits = np.zeros(len(pts))
    for v in views:
        uv, z = project(v, pts)
        h, w = v["mask"].shape
        u, vv = np.round(uv).astype(int).T
        ok = (z > 0) & (u >= 0) & (u < w) & (vv >= 0) & (vv < h)
        hits[ok] += v["mask"][vv[ok], u[ok]]
    lo, hi = np.percentile(pts[hits >= 0.6 * len(views)], [1, 99], axis=0)
    pad = (hi - lo) * BOX_MARGIN
    return lo - pad, hi + pad


def silhouette_carve(views, lo, hi, grid):
    step = (hi - lo).max() / grid
    dims = np.ceil((hi - lo) / step).astype(int)
    axes = [lo[i] + step * (np.arange(dims[i]) + 0.5) for i in range(3)]
    occ = np.zeros(dims, bool)
    gx, gy = np.meshgrid(axes[0], axes[1], indexing="ij")
    for k, z in enumerate(axes[2]):  # one z-slab at a time keeps RAM low
        X = np.stack([gx.ravel(), gy.ravel(), np.full(gx.size, z)], 1)
        keep = np.ones(gx.size, bool)
        for v in views:
            uv, d = project(v, X)
            h, w = v["mask"].shape
            u, vv = np.round(uv).astype(int).T
            inside = (d > 0) & (u >= 0) & (u < w) & (vv >= 0) & (vv < h)
            keep[inside] &= v["mask"][vv[inside], u[inside]]
        occ[:, :, k] = keep.reshape(gx.shape)
    return largest(ndimage.binary_opening(occ)), step


def largest(occ):
    lab, n = ndimage.label(occ)
    return lab == (np.bincount(lab.ravel())[1:].argmax() + 1) if n > 1 else occ


def to_mesh(occ, lo, step, sigma):
    vol = ndimage.gaussian_filter(np.pad(occ, 2).astype(np.float32), sigma)
    verts, faces, _, _ = measure.marching_cubes(vol, 0.5)
    mesh = trimesh.Trimesh((verts - 1.5) * step + lo, faces[:, ::-1], process=True)
    trimesh.smoothing.filter_taubin(mesh, iterations=25)
    if mesh.volume < 0:
        mesh.invert()
    return mesh


def hull_depth(mesh, v, scale):
    uv, z = project(v, mesh.vertices)
    h, w = v["mask"].shape
    H, W = int(h * scale), int(w * scale)
    su, sv = (uv * scale).astype(int).T
    ok = (z > 0) & (su >= 0) & (su < W) & (sv >= 0) & (sv < H)
    zb = np.full(H * W, np.inf)
    np.minimum.at(zb, sv[ok] * W + su[ok], z[ok])
    return ndimage.grey_erosion(zb.reshape(H, W), size=3)


def depth_carve(occ, lo, step, views, cfg):
    """MoGe-2 depth is relative: fit it to the hull's depth (robust scale +
    shift; concavities are the outliers), then cut the voxels a photo saw through."""
    mesh = to_mesh(occ, lo, step, cfg.smooth_sigma)
    idx = np.argwhere(occ)
    X = lo + (idx + 0.5) * step
    votes = np.zeros(len(idx), np.uint8)
    for v in views:
        f = cfg.depth / (v["name"].rsplit(".", 1)[0] + ".npz")
        if not f.exists():
            continue
        d = np.load(f)
        x0, y0, x1, y1 = d["box"]
        h, w = v["mask"].shape
        full = np.full((h, w), np.nan, np.float32)
        full[y0:y1, x0:x1] = d["depth"]
        valid = cv2.erode(v["mask_raw"].astype(np.uint8), np.ones((2 * DEPTH_EDGE_PX + 1,) * 2, np.uint8)) > 0
        valid &= np.isfinite(full)
        H, W = int(h * DEPTH_SCALE), int(w * DEPTH_SCALE)
        zh = hull_depth(mesh, v, DEPTH_SCALE)
        dm = cv2.resize(np.nan_to_num(full, nan=0), (W, H), interpolation=cv2.INTER_NEAREST)
        vm = cv2.resize(valid.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
        sel = vm & np.isfinite(zh)
        a_, b_ = dm[sel].astype(np.float64), zh[sel]
        keep = np.ones(len(a_), bool)
        for _ in range(6):
            (s_, t_), *_ = np.linalg.lstsq(np.c_[a_[keep], np.ones(keep.sum())], b_[keep], rcond=None)
            r = b_ - (s_ * a_ + t_)
            mad = np.median(np.abs(r[keep] - np.median(r[keep]))) + 1e-9
            keep = np.abs(r) < 2.5 * 1.4826 * mad
        seen = np.where(valid, s_ * full + t_, np.nan)
        uv, z = project(v, X)
        u, vv = np.round(uv).astype(int).T
        ins = (u >= 0) & (u < w) & (vv >= 0) & (vv < h)
        D = np.full(len(X), np.nan)
        D[ins] = seen[vv[ins], u[ins]]
        margin = max(DEPTH_MARGIN_VOX * step, cfg.depth_margin_sigma * 1.4826 * mad)
        votes += np.isfinite(D) & (z < D - margin)
    cut = votes >= DEPTH_MIN_VOTES
    occ = occ.copy()
    occ[tuple(idx[cut].T)] = False
    print(f"  depth carving removed {cut.mean() * 100:.1f}% of the voxels")
    return largest(ndimage.binary_opening(occ))


def finish(mesh_cam, views, frame, cfg, ped=None):
    """Cameras' frame mesh -> upright, table cut, optional pedestal, painted."""
    mesh = mesh_cam.copy()
    mesh.vertices = frame.to_up(mesh.vertices)
    mesh = mesh.slice_plane([0, 0, 0], [0, 0, 1], cap=True)
    if ped is not None:
        mesh = pedestal.build(mesh, ped)
    mesh.vertices = frame.from_up(mesh.vertices)
    paint.paint(mesh, views, cfg)
    mesh.vertices = frame.to_up(mesh.vertices)
    if ped is not None:
        pedestal.paint_top(mesh, ped)
    return mesh


def run(cfg, profile=None):
    rec, views = load_views(cfg)
    lo, hi = initial_box(rec, views)
    occ, step = silhouette_carve(views, lo, hi, cfg.grid)
    if any(cfg.depth.glob("*.npz")):
        occ = depth_carve(occ, lo, step, views, cfg)
    hull = to_mesh(occ, lo, step, cfg.smooth_sigma)
    hull.export(cfg.hull_cam)

    frame = Frame.from_hull(hull, views)
    up = hull.copy()
    up.vertices = frame.to_up(up.vertices)
    z_floor = floor_height(up, views, frame)
    frame = frame.lowered(z_floor)
    up.vertices -= [0, 0, z_floor]
    print(f"  table found; the {z_floor / up.vertices[:, 2].max() * 100:.0f}% (of the object's height) "
          f"cone the hull grows below it is cut off")
    up = up.slice_plane([0, 0, 0], [0, 0, 1], cap=True)
    ped = pedestal.measure(up) if cfg.pedestal else None
    ctx = dict(cfg=cfg, views=views, frame=frame, pedestal=ped, lo=lo, step=step)
    if profile is not None and hasattr(profile, "refine_voxels"):
        occ = profile.refine_voxels(occ, ctx)
        hull = to_mesh(occ, lo, step, cfg.smooth_sigma)
    np.savez(cfg.frame, M=frame.M, **({f"ped_{k}": v for k, v in ped.items()} if ped else {}))

    cfg.out.mkdir(parents=True, exist_ok=True)
    ious = paint.silhouette_check(hull, views, cfg.out / "check_carved.jpg")
    model = finish(hull, views, frame, cfg, ped)
    model.export(cfg.out / "model_carved.ply")
    print(f"  carved model: silhouette IoU {ious.mean():.3f} (min {ious.min():.3f}), "
          f"{len(model.vertices)} vertices -> {cfg.out / 'model_carved.ply'}")
    return ious.mean()
