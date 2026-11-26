"""Colour a mesh with the photos.

For every vertex: the photos that see it (z-buffer), weighted by how head-on;
the colour of the photo that agrees most with the others (weighted medoid, so a
photo that lands on the wrong part is outvoted and no colour is invented);
green spill removed; points the photos disagree on, or none sees, take their
neighbours' colour.
"""
import cv2
import numpy as np
from scipy import ndimage

from .geometry import project

DESPILL_MAX = 0.5          # never remove more than half the cloth colour
WEAK_CHROMA = 85           # below this chroma a pixel is black/white/grey + reflection
NEUTRAL_RESIDUAL = 25      # after despill, a spread below this is a grey residue
ZBUF_SCALE = 0.5
ZBUF_TOL = 0.003


def despill(c, cloth):
    """Glossy surfaces mirror the green cloth. Subtract as much cloth colour as
    the pixel's G and B excess over R allows (never past grey, so no magenta),
    only for greenish or weakly coloured pixels (strong blues and reds stay)."""
    chroma = c[:, 1:] - c[:, :1]
    k = cloth[1:] - cloth[0]
    a = np.clip((chroma / k).min(1), 0, DESPILL_MAX)
    a[(c[:, 1] < c[:, 2]) & (np.linalg.norm(chroma, axis=1) > WEAK_CHROMA)] = 0
    out = np.clip(c - a[:, None] * cloth, 0, 255)
    grey = (a > 0) & (np.ptp(out, axis=1) < NEUTRAL_RESIDUAL)
    cyan = (a > 0) & (np.abs(out[:, 1] - out[:, 0]) < 10) & (out[:, 2] - out[:, 0] < 45)
    grey |= cyan  # the cloth's blue part can outlive its green on glossy black/white
    out[grey] = out[grey].mean(1, keepdims=True)
    return out


def observe(mesh, views):
    """Per photo: how squarely each vertex is seen (0 = hidden) and its colour."""
    V, N = mesh.vertices, mesh.vertex_normals
    score = np.zeros((len(views), len(V)))
    cols = np.zeros((len(views), len(V), 3))
    for i, v in enumerate(views):
        uv, z = project(v, V)
        h, w = v["mask"].shape
        su, sv = (uv * ZBUF_SCALE).astype(int).T
        H, W = int(h * ZBUF_SCALE), int(w * ZBUF_SCALE)
        ok = (z > 0) & (su >= 0) & (su < W) & (sv >= 0) & (sv < H)
        zb = np.full(H * W, np.inf)
        np.minimum.at(zb, sv[ok] * W + su[ok], z[ok])
        zb = ndimage.grey_erosion(zb.reshape(H, W), size=5).ravel()
        vis = np.zeros(len(V), bool)
        vis[ok] = z[ok] <= zb[sv[ok] * W + su[ok]] * (1 + ZBUF_TOL) + 1e-9
        to_cam = v["center"] - V
        to_cam /= np.linalg.norm(to_cam, axis=1, keepdims=True)
        facing = np.clip((N * to_cam).sum(1), 0, 1)
        u, vv = np.clip(np.round(uv), 0, [w - 1, h - 1]).astype(int).T
        vis &= v["mask_inner"][vv, u]   # silhouette edges mix object and cloth
        score[i] = np.where(vis, facing, 0)
        cols[i] = despill(v["rgb"][vv, u].astype(np.float64), v["cloth"])
    return score, cols


def fill_from_neighbours(mesh, rgb, known, iters=200):
    rgb, known = rgb.copy(), known.copy()
    e = mesh.edges_unique
    for _ in range(iters):
        if known.all():
            break
        acc = np.zeros_like(rgb)
        cnt = np.zeros(len(rgb))
        for a, b in ((e[:, 0], e[:, 1]), (e[:, 1], e[:, 0])):
            k = known[b]
            np.add.at(acc, a[k], rgb[b[k]])
            np.add.at(cnt, a[k], 1)
        new = ~known & (cnt > 0)
        rgb[new] = acc[new] / cnt[new, None]
        known |= new
    return rgb


def paint(mesh, views, cfg):
    score, cols = observe(mesh, views)
    score = score ** cfg.facing_power
    best = np.argsort(-score, axis=0)[:cfg.top_views]
    wts = np.take_along_axis(score, best, 0)
    c = np.take_along_axis(cols, best[..., None], 0)
    dist = np.linalg.norm(c[:, None] - c[None], axis=-1)
    cost = (dist * wts[None]).sum(1) + (wts == 0) * 1e9
    pick = cost.argmin(0)
    rgb = np.take_along_axis(c, pick[None, :, None], 0)[0]
    unseen = wts.sum(0) == 0
    spread = np.take_along_axis(cost, pick[None], 0)[0] / np.maximum(wts.sum(0), 1e-9)
    unsure = ((wts > 0).sum(0) >= 2) & (spread > cfg.max_disagreement)
    print(f"  paint: {unseen.mean() * 100:.1f}% of vertices unseen, {unsure.mean() * 100:.1f}% disputed")
    rgb = fill_from_neighbours(mesh, rgb, ~unseen & ~unsure)
    mesh.visual.vertex_colors = np.c_[rgb.clip(0, 255), np.full(len(rgb), 255)].astype(np.uint8)
    seen = (wts > 0).sum(0) >= 2
    # photo-consistency: how much the photos disagree on a point's colour. A shape
    # that is off makes each photo see something else at the same point.
    return float(np.median(spread[seen])) if seen.any() else float("nan")


def silhouette_check(mesh, views, path):
    """Mesh rasterised into every photo vs. the green-key mask: red = extra
    volume, blue = missing. Returns the IoU per photo."""
    tiles, ious = [], []
    for v in views:
        uv, _ = project(v, mesh.vertices)
        sil = np.zeros(v["mask"].shape, np.uint8)
        cv2.fillPoly(sil, list(np.round(uv[mesh.faces] * 4).astype(np.int32)), 255, shift=2)
        sil = cv2.morphologyEx(sil, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        m, s = v["mask_raw"], sil > 0
        ious.append((m & s).sum() / (m | s).sum())
        img = (v["rgb"][:, :, ::-1] * 0.6).astype(np.uint8)
        img[s & m] = (img[s & m] * 0.4 + 120).astype(np.uint8)
        img[s & ~m] = (0, 0, 255)
        img[m & ~s] = (255, 80, 0)
        ys, xs = np.nonzero(m)
        crop = img[max(ys.min() - 40, 0):ys.max() + 40, max(xs.min() - 40, 0):xs.max() + 40]
        crop = cv2.resize(crop, (300, round(300 * crop.shape[0] / crop.shape[1])))
        crop = cv2.copyMakeBorder(crop, 0, max(0, 560 - crop.shape[0]), 0, 0, cv2.BORDER_CONSTANT)[:560]
        cv2.putText(crop, f"IoU {ious[-1]:.3f}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        tiles.append(crop)
    while len(tiles) % 5:
        tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(str(path), np.vstack([np.hstack(tiles[i:i + 5]) for i in range(0, len(tiles), 5)]))
    return np.array(ious)
