"""Step 6: place each generated mesh where the photos say, paint it, keep the best.

The generated shape comes in its own frame and scale. Stand it up, scale it to
the carved model's height, then search rotation, translation and per-axis scale
that maximise silhouette IoU over all photos (yaw sweep + Nelder-Mead). Fine
relief the generator sculpted a little off the real one (eyes, printed letters)
is smoothed away so the photos' paint, not misplaced relief, draws those details.

Candidates (one per seed) are ranked by silhouette IoU and photo-consistency.
"""
import json

import numpy as np
import trimesh
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from . import paint, pedestal
from .geometry import Frame, iou_views, load_views, polar_grid, raster, silhouettes

ALIGN_FACES = 40000
ALIGN_SCALE = 0.25
YAW_STEP = 10


def load_generated(path, height):
    """Y-up generated mesh -> Z up, standing at z = 0, centred on its bottom, scaled."""
    g = trimesh.load(path, force="mesh", process=True)
    g.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    v = g.vertices
    low = v[:, 2] < v[:, 2].min() + 0.05 * np.ptp(v[:, 2])
    centre = (v[low, :2].max(0) + v[low, :2].min(0)) / 2
    g.vertices = (v - [*centre, v[:, 2].min()]) * (height / np.ptp(v[:, 2]))
    return g


def place_up(V, p, origin):
    """p = yaw, pitch, roll (deg), tx, ty, tz, log scale x, y, z; origin = where the
    carved model's bottom centre is (the search starts there)."""
    R = Rotation.from_euler("zyx", p[:3], degrees=True).as_matrix()
    return (V * np.exp(p[6:9])) @ R.T + p[3:6] + origin


def search(gen, views, frame, ped, height, origin, cache):
    if cache.exists():
        return np.array(json.loads(cache.read_text()))
    small = gen.simplify_quadric_decimation(face_count=ALIGN_FACES) if len(gen.faces) > ALIGN_FACES else gen
    Vs, Fs = small.vertices, small.faces
    sils = silhouettes(views, ALIGN_SCALE)
    sub = list(range(0, len(views), 2))
    zt = ped["z_top"] if ped else 0.0
    if ped:  # compare "generated object + fitted pedestal" to the photos
        cone = trimesh.util.concatenate([
            polar_grid(lambda t: ped["r"] + (ped["r_top"] - ped["r"]) * t, lambda t: zt * t, 4, 90),
            polar_grid(lambda t: ped["r_top"] * t, lambda t: zt + 0 * t, 4, 90)])
        cone.apply_translation([ped["cx"], ped["cy"], 0])
        base_r = [raster(v, frame.from_up(cone.vertices), cone.faces, m.shape, ALIGN_SCALE)
                  for v, m in zip(views, sils)]
    else:
        base_r = None

    def loss(p, idx=None):
        Y = place_up(Vs, p, origin)
        F = Fs[Y[Fs][:, :, 2].max(1) > zt] if ped else Fs
        vs = [views[i] for i in idx] if idx else views
        ss = [sils[i] for i in idx] if idx else sils
        br = ([base_r[i] for i in idx] if idx else base_r) if base_r else None
        return 1 - iou_views(frame.from_up(Y), F, vs, ss, ALIGN_SCALE, br).mean()

    best = min((loss([yaw, 0, 0, 0, 0, 0, 0, 0, 0], sub), yaw) for yaw in range(0, 360, YAW_STEP))
    p = np.array([best[1], 0, 0, 0, 0, 0, 0, 0, 0], float)
    d = 0.02 * height
    steps = np.array([4, 2, 2, d, d, d, 0.03, 0.03, 0.03])
    for it, div in ((800, 1), (500, 3)):  # the restart matters: Nelder-Mead often stalls once
        res = minimize(loss, p, method="Nelder-Mead",
                       options=dict(maxiter=it, xatol=1e-4, fatol=1e-6,
                                    initial_simplex=np.vstack([p, p + np.diag(steps / div)])))
        p = res.x
    print(f"    yaw {p[0]:.1f} pitch {p[1]:.1f} roll {p[2]:.1f}, scale {np.round(np.exp(p[6:9]), 3).tolist()}, "
          f"IoU {1 - res.fun:.3f} at {ALIGN_SCALE}x")
    cache.write_text(json.dumps(p.tolist()))
    return p


def build(cfg, path, views, frame, ped, height, origin, profile=None):
    """One candidate, end to end. Returns (mesh upright and painted, IoU, disagreement)."""
    gen = load_generated(path, height)
    cache = path.with_suffix(".align_ped.json" if ped else ".align.json")  # the search differs with a pedestal
    p = search(gen, views, frame, ped, height, origin, cache)
    mesh = gen.copy()
    if cfg.relief_smooth:
        trimesh.smoothing.filter_taubin(mesh, lamb=0.5, nu=-0.53, iterations=cfg.relief_smooth)
    mesh.vertices = place_up(mesh.vertices, p, origin)
    mesh = mesh.slice_plane([0, 0, 0], [0, 0, 1], cap=True)
    if ped:
        mesh = pedestal.build(mesh, ped)
    parts = {}
    ctx = dict(cfg=cfg, views=views, frame=frame, pedestal=ped)
    if profile is not None and hasattr(profile, "parts"):
        mesh.vertices = frame.from_up(mesh.vertices)        # profiles measure on a painted model
        paint.paint(mesh, views, cfg)
        mesh.vertices = frame.to_up(mesh.vertices)
        mesh, parts, ctx["profile_state"] = profile.parts(mesh, ctx)
    pieces = [mesh] + list(parts.values())
    out = trimesh.util.concatenate(pieces) if parts else mesh
    ctx["sizes"] = dict(zip(["object", *parts], [len(q.vertices) for q in pieces]))
    out.vertices = frame.from_up(out.vertices)
    ious = paint.silhouette_check(out, views, cfg.out / f"check_{path.stem}.jpg")
    spread = paint.paint(out, views, cfg)
    out.vertices = frame.to_up(out.vertices)
    if ped:
        pedestal.paint_top(out, ped)
    if profile is not None and hasattr(profile, "clean_colours"):
        profile.clean_colours(out, ctx)
    return out, float(ious.mean()), spread


def run(cfg, profile=None):
    cands = sorted(p for p in cfg.generated.glob("mv_*.ply") if ".painted" not in p.name)
    if not cands:
        print("  no generated meshes: using the carved model")
        return None
    _, views = load_views(cfg)
    fz = np.load(cfg.frame)
    frame = Frame(fz["M"])
    ped = {k[4:]: float(fz[k]) for k in fz.files if k.startswith("ped_")} or None
    hull = trimesh.load(cfg.hull_cam, process=False)
    hull.vertices = frame.to_up(hull.vertices)
    height = hull.vertices[:, 2].max()
    hv = hull.vertices[(hull.vertices[:, 2] > 0) & (hull.vertices[:, 2] < 0.05 * height)]
    origin = np.r_[(hv[:, :2].max(0) + hv[:, :2].min(0)) / 2, 0.0]
    results = []
    for path in cands:
        print(f"  {path.name}")
        mesh, iou, spread = build(cfg, path, views, frame, ped, height, origin, profile)
        mesh.export(cfg.generated / f"{path.stem}.painted.ply")
        results.append((path.stem, iou, spread))
        print(f"    silhouette IoU {iou:.3f}, photo disagreement {spread:.1f}")
    # rank: better IoU and lower disagreement; average the two ranks
    r_iou = np.argsort(np.argsort([-r[1] for r in results]))
    r_sp = np.argsort(np.argsort([r[2] for r in results]))
    best = results[int(np.argmin(r_iou + r_sp + 1e-3 * r_iou))]
    print(f"  best: {best[0]} (IoU {best[1]:.3f}, disagreement {best[2]:.1f})")
    model = trimesh.load(cfg.generated / f"{best[0]}.painted.ply", process=False)
    if cfg.height_cm:
        model.vertices *= cfg.height_cm / 100 / model.vertices[:, 2].max()
    model.export(cfg.out / "model.ply")
    (cfg.out / "candidates.json").write_text(json.dumps(
        [dict(name=n, iou=i, disagreement=s) for n, i, s in results], indent=1))
    return best
