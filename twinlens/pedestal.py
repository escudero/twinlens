"""Optional (--pedestal): figurines often stand on a round, plain base. Photos
taken at one height leave it lumpy; measure it and swap in the exact shape (a
tapered cylinder). Works in the upright frame, table at z = 0."""
import numpy as np
import trimesh

from .geometry import polar_grid

TOP_LIFT = 0.005           # fraction of height: keep the top clear of the carved/generated one (no z-fighting)


def slice_area(mesh, z):
    sec = mesh.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
    if sec is None:
        return 0.0
    planar, _ = sec.to_2D()
    return sum(p.area for p in planar.polygons_full)


def measure(mesh):
    """Where the pedestal ends (the slice area drops) and a circle fit to its side."""
    h = mesh.vertices[:, 2].max()
    zs = np.linspace(0.005 * h, 0.4 * h, 160)
    areas = np.array([slice_area(mesh, z) for z in zs])
    a0 = np.median(areas[zs < 0.04 * h])
    z_top = zs[np.argmax(areas < 0.6 * a0)]
    pts, zz = [], np.linspace(0.1, 0.8, 10) * z_top
    for z in zz:
        sec = mesh.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
        pts.append(sec.vertices[:, :2])
    allp = np.concatenate(pts)
    A = np.c_[2 * allp, np.ones(len(allp))]
    (cx, cy, _), *_ = np.linalg.lstsq(A, (allp ** 2).sum(1), rcond=None)
    radii = [np.median(np.linalg.norm(p - [cx, cy], axis=1)) for p in pts]
    slope, r0 = np.polyfit(zz, radii, 1)
    z_top += TOP_LIFT * h
    b = dict(z_top=z_top, cx=cx, cy=cy, r=r0, r_top=r0 + slope * z_top, h=h)
    print(f"  pedestal: radius {b['r'] / h:.3f} -> {b['r_top'] / h:.3f}, height {z_top / h:.3f} of the object")
    return b


def build(mesh, b):
    """Object above the pedestal + side tube + flat top disc (separate pieces:
    crisp edge; no cap on the cut, it would z-fight with the disc)."""
    top = mesh.slice_plane([0, 0, b["z_top"]], [0, 0, 1], cap=False)
    zt, r0, r1 = b["z_top"], b["r"], b["r_top"]
    side = polar_grid(lambda t: r0 + (r1 - r0) * t, lambda t: zt * t, 60)
    disc = polar_grid(lambda t: r1 * t, lambda t: zt + 0 * t, 120)
    for part in (side, disc):
        part.apply_translation([b["cx"], b["cy"], 0])
    if disc.face_normals[:, 2].mean() < 0:
        disc.invert()
    radial = side.triangles_center[:, :2] - [b["cx"], b["cy"]]
    if (np.einsum("ij,ij->i", side.face_normals[:, :2], radial) < 0).mean() > 0.5:
        side.invert()
    return trimesh.util.concatenate([top, side, disc])


def paint_top(mesh, b):
    """The top is seen only at a grazing angle, where gloss mirrors the cloth:
    give it the colour of the pedestal's side."""
    v, n = mesh.vertices, mesh.vertex_normals
    col = mesh.visual.vertex_colors
    top = (np.abs(v[:, 2] - b["z_top"]) < 1e-6 * b["h"] + 1e-9) & (n[:, 2] > 0.9)
    band = (v[:, 2] > 0.5 * b["z_top"]) & (v[:, 2] < 0.95 * b["z_top"]) & (np.abs(n[:, 2]) < 0.3)
    if band.any():
        col[top, :3] = np.median(col[band, :3], 0).astype(np.uint8)
        mesh.visual.vertex_colors = col
