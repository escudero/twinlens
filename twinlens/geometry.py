"""Cameras, projection and the object's upright frame."""
from pathlib import Path

import cv2
import numpy as np
import pycolmap
import trimesh


def load_views(cfg, grow_px=None):
    """One dict per registered photo: camera, pose, image, masks, cloth colour."""
    rec = pycolmap.Reconstruction(str(cfg.cameras))
    grow = cfg.mask_grow_px if grow_px is None else grow_px
    views = []
    for img in sorted(rec.images.values(), key=lambda i: i.name):
        cam = rec.cameras[img.camera_id]
        T = img.cam_from_world() if callable(img.cam_from_world) else img.cam_from_world
        R, t = T.rotation.matrix(), T.translation
        raw = cv2.imread(str(cfg.masks / (Path(img.name).stem + ".png")), 0)
        rgb = cv2.imread(str(cfg.images / img.name))[:, :, ::-1]
        far = cv2.dilate(raw, np.ones((61, 61), np.uint8)) == 0
        views.append(dict(
            name=img.name, cam=cam, R=R, t=t, center=-R.T @ t, rgb=rgb,
            mask=cv2.dilate(raw, np.ones((2 * grow + 1,) * 2, np.uint8)) > 0,
            mask_raw=raw > 0,
            mask_inner=cv2.erode(raw, np.ones((2 * cfg.color_inset_px + 1,) * 2, np.uint8)) > 0,
            cloth=np.median(rgb[far], 0).astype(np.float64)))
    return rec, views


def project(v, X):
    """World points -> pixel coordinates and depth in view v."""
    Xc = X @ v["R"].T + v["t"]
    return np.asarray(v["cam"].img_from_cam(Xc)), Xc[:, 2]


def up_axis(views):
    """The photos were taken around the object: their centres span a plane whose
    normal is the vertical. The photos' own "up" picks its sign."""
    C = np.array([v["center"] for v in views])
    c0 = C.mean(0)
    _, _, vt = np.linalg.svd(C - c0)
    up = vt[2]
    cam_up = np.mean([-v["R"][1] for v in views], 0)
    return (up if up @ cam_up > 0 else -up), c0, cam_up


def azimuths(views, front=0):
    """Angle of each camera around the vertical, 0 at photo `front`."""
    up, c0, _ = up_axis(views)
    C = np.array([v["center"] for v in views]) - c0
    x = C[front] - (C[front] @ up) * up
    x /= np.linalg.norm(x)
    y = np.cross(up, x)
    return np.degrees(np.arctan2(C @ y, C @ x)) % 360


class Frame:
    """Affine map between the cameras' frame and the upright frame (Z up, the
    table at z = 0, the object centred on the Z axis)."""

    def __init__(self, M):
        self.M = np.asarray(M, float)
        self.Minv = np.linalg.inv(self.M)

    def to_up(self, X):
        return X @ self.M[:3, :3].T + self.M[:3, 3]

    def from_up(self, Y):
        return Y @ self.Minv[:3, :3].T + self.Minv[:3, 3]

    @classmethod
    def from_hull(cls, hull, views):
        up, c0, cam_up = up_axis(views)
        x = np.cross(cam_up if abs(cam_up @ up) < 0.99 else [1, 0, 0], up)
        x /= np.linalg.norm(x)
        y = np.cross(up, x)
        R = np.stack([x, y, up])
        v = (hull.vertices - c0) @ R.T
        shift = np.array([np.median(v[:, 0]), np.median(v[:, 1]), v[:, 2].min()])
        M = np.eye(4)
        M[:3, :3] = R
        M[:3, 3] = -R @ c0 - shift
        return cls(M)

    def lowered(self, dz):
        """Same frame with z shifted down by dz (puts the table at z = 0)."""
        M = self.M.copy()
        M[2, 3] -= dz
        return Frame(M)


def floor_height(hull_up, views, frame, offsets=(4, 8, 12, 16)):
    """Height of the table in the upright frame. In every photo the lowest pixel
    of the silhouette is where the object meets the table on the camera's side.
    A ray through that exact pixel only grazes the hull (it often misses), so
    rays a few pixels above it are cast, where they hit the near face squarely,
    and their height is extrapolated back to the silhouette's edge. Median over
    photos. Makes no assumption about the object's shape (the hull itself grows
    a cone below the object, because no photo sees underneath)."""
    per_offset = []
    for k in offsets:
        zs = []
        for v in views:
            m = v["mask_raw"]
            rows = np.nonzero(m.any(1))[0]
            if not len(rows):
                continue
            r = rows.max() - k
            c = np.median(np.nonzero(m[r])[0])
            xy = np.asarray(v["cam"].cam_from_img(np.array([[c, r]], float)))[0]
            o = frame.to_up(v["center"][None])[0]
            d = frame.to_up((v["center"] + v["R"].T @ np.r_[xy, 1.0])[None])[0] - o
            loc, _, _ = hull_up.ray.intersects_location([o], [d / np.linalg.norm(d)], multiple_hits=False)
            if len(loc):
                zs.append(loc[0, 2])
        per_offset.append(np.median(zs) if zs else np.nan)
    k, z = np.array(offsets, float), np.array(per_offset)
    ok = np.isfinite(z)
    if ok.sum() >= 2:
        return float(np.polyval(np.polyfit(k[ok], z[ok], 1), 0.0))
    return float(z[ok][0]) if ok.any() else 0.0


def polar_grid(r_of_t, z_of_t, n_t, n_theta=360):
    """Quad strip around Z: t in [0, 1] runs along the profile (radius, height)."""
    t = np.linspace(0, 1, n_t)
    th = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    T, TH = np.meshgrid(t, th, indexing="ij")
    R = r_of_t(T)
    V = np.stack([R * np.cos(TH), R * np.sin(TH), z_of_t(T)], -1).reshape(-1, 3)
    i, j = np.meshgrid(np.arange(n_t - 1), np.arange(n_theta), indexing="ij")
    a = i * n_theta + j
    b = i * n_theta + (j + 1) % n_theta
    c, d = a + n_theta, b + n_theta
    F = np.concatenate([np.stack([a, b, d], -1).reshape(-1, 3), np.stack([a, d, c], -1).reshape(-1, 3)])
    return trimesh.Trimesh(V, F, process=False)


def silhouettes(views, scale):
    return [cv2.resize(v["mask_raw"].astype(np.uint8), None, fx=scale, fy=scale,
                       interpolation=cv2.INTER_AREA) > 0 for v in views]


def raster(v, V, F, shape, scale):
    """Filled silhouette of mesh (V, F) in view v at `scale` of the photo size."""
    uv, _ = project(v, V)
    img = np.zeros(shape, np.uint8)
    cv2.fillPoly(img, list(np.round(uv[F] * scale * 4).astype(np.int32)), 1, shift=2)
    return img


def iou_views(V, F, views, sils, scale, base_rasters=None):
    out = []
    for i, (v, m) in enumerate(zip(views, sils)):
        img = raster(v, V, F, m.shape, scale)
        if base_rasters is not None:
            img |= base_rasters[i]
        s = img > 0
        out.append((m & s).sum() / max((m | s).sum(), 1))
    return np.array(out)
