"""TwinLens profile for a Mickey Mouse Club figurine (the study this tool
came from). An example of how to add object-specific rules without touching the
core; run with  --profile examples/mickey/profile.py --pedestal

Hooks (all optional) called by the core:
  refine_voxels(occ, ctx)      step 4, extra carving rules
  parts(mesh, ctx)             step 6, replace pieces with exact shapes; returns
                               (mesh without them, {name: part mesh}, state)
  clean_colours(mesh, ctx)     step 6, colour rules after painting

Here: the ring, the glass "100 Years of Wonder" tag and the wire hook are drawn
as vectors (measured on the model, the ring fitted to the photos' silhouettes);
red left on the figure is a stain (only the tongue is red); shoes and gloves are
flat colours; the tag is glass in a bronze frame (vertex alpha 80 / 160).
Upright frame throughout: Z up, the table at z = 0.
"""
import numpy as np
import trimesh

RING_SEGMENTS = 1440      # around the ring: ~2 photo pixels per vertex on the front face
RING_RADIAL = 70          # across the ring's front face
TAG_GRID = (440, 320)     # glass face resolution (across, down): finer than the print strokes in the full photo
METAL, GLASS = 160, 80
TAG_MIN_HEIGHT = 0.12     # fraction of the figure's height
MIN_PIECE = 0.002         # drop mesh pieces smaller than this fraction of the figure


def ring_frame(mesh):
    """Find the ring from the painted, aligned model: red vertices high up span
    its plane; a circle through them gives centre and radii."""
    v = mesh.vertices
    c = mesh.visual.vertex_colors[:, :3].astype(int)
    h = v[:, 2].max()
    red = (c[:, 0] - c[:, 1] > 50) & (v[:, 2] > 0.5 * h)
    o = v[red][:, :2].mean(0)
    _, _, vt = np.linalg.svd(v[red][:, :2] - o, full_matrices=False)
    e_u, e_w = vt[0], vt[1]
    u = (v[:, :2] - o) @ e_u
    w = (v[:, :2] - o) @ e_w
    P = np.c_[u[red], v[red, 2]]
    A = np.c_[2 * P, np.ones(len(P))]
    (u0, z0, _), *_ = np.linalg.lstsq(A, (P ** 2).sum(1), rcond=None)
    r = np.hypot(u - u0, v[:, 2] - z0)
    r_in, r_out = np.percentile(r[red], [2, 98])
    ring = red & (r > r_in) & (r < r_out)
    # the two faces are the two density peaks along the normal (raised letters
    # spread beyond them and must not count as thickness)
    hist, edges = np.histogram(w[ring], bins=60)
    mid = np.median(w[ring])
    lo_side, hi_side = edges[:-1] < mid, edges[:-1] >= mid
    w0 = edges[np.argmax(np.where(lo_side, hist, -1))] + (edges[1] - edges[0]) / 2
    w1 = edges[np.argmax(np.where(hi_side, hist, -1))] + (edges[1] - edges[0]) / 2
    return dict(o=o, e_u=e_u, e_w=e_w, u0=u0, z0=z0, r_in=r_in, r_out=r_out, w0=w0, w1=w1, h=h,
                u=u, w=w, r=r)


def to_world(f, U, Z, W):
    """Ring-plane coordinates (u along the ring, z up, w along its normal) -> upright XYZ."""
    xy = f["o"] + np.outer(U, f["e_u"]) + np.outer(W, f["e_w"])
    return np.c_[xy, Z]


def grid(n_a, n_b, closed_a=False):
    """Faces of an n_a x n_b vertex grid (optionally wrapping along a)."""
    ia = np.arange(n_a if closed_a else n_a - 1)
    ib = np.arange(n_b - 1)
    A, B = np.meshgrid(ia, ib, indexing="ij")
    a0 = A * n_b + B
    a1 = ((A + 1) % n_a) * n_b + B
    return np.concatenate([np.stack([a0, a1, a1 + 1], -1).reshape(-1, 3),
                           np.stack([a0, a1 + 1, a0 + 1], -1).reshape(-1, 3)])


def ring_mesh(f, segments=RING_SEGMENTS, radial=RING_RADIAL):
    """Flat annulus: front and back faces plus inner and outer walls."""
    th = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    parts = []
    for kind in ("front", "back", "outer", "inner"):
        if kind in ("front", "back"):
            rad = np.linspace(f["r_in"], f["r_out"], radial)
            T, R = np.meshgrid(th, rad, indexing="ij")
            W = np.full_like(T, f["w1"] if kind == "front" else f["w0"])
        else:
            ws = np.linspace(f["w0"], f["w1"], 12 if radial > 4 else 2)
            T, W = np.meshgrid(th, ws, indexing="ij")
            R = np.full_like(T, f["r_out"] if kind == "outer" else f["r_in"])
        U = f["u0"] + R * np.cos(T)
        Z = f["z0"] + R * np.sin(T)
        V = to_world(f, U.ravel(), Z.ravel(), W.ravel())
        m = trimesh.Trimesh(V, grid(*T.shape, closed_a=True), process=False)
        # outward normals: front/back along +-w, walls radially out (outer) or in (inner)
        cen = m.triangles_center
        if kind == "front":
            want = np.tile(np.r_[f["e_w"], 0], (len(cen), 1))
        elif kind == "back":
            want = -np.tile(np.r_[f["e_w"], 0], (len(cen), 1))
        else:
            ctr = to_world(f, np.array([f["u0"]]), np.array([f["z0"]]), np.array([0.0]))[0]
            out_dir = cen - ctr
            out_dir -= np.outer(out_dir @ np.r_[f["e_w"], 0], np.r_[f["e_w"], 0])
            want = out_dir if kind == "outer" else -out_dir
        orient(m, want)
        parts.append(m)
    return trimesh.util.concatenate(parts)


def orient(m, want):
    """Flip the faces whose normal points against the wanted direction."""
    flip = np.einsum("ij,ij->i", m.face_normals, want) < 0
    F = m.faces.copy()
    F[flip] = F[flip][:, ::-1]
    m.faces = F


def tag_box(f):
    """The tag hangs inside the hole at the top; its extent is read from the
    model's non-red vertices there, inside the ring's depth band."""
    v = f["_v"]
    c = f["_c"]
    h = f["h"]
    red = (c[:, 0] - c[:, 1] > 50)
    band = (f["w"] > f["w0"] - 0.01 * h) & (f["w"] < f["w1"] + 0.01 * h)
    inside = (f["r"] < f["r_in"] - 0.005 * h) & (v[:, 2] > f["z0"]) & (np.abs(f["u"] - f["u0"]) < 0.085 * h)
    t = band & inside & ~red
    ul, uh = np.percentile(f["u"][t], [3, 97])
    zl, zh = np.percentile(v[t, 2], [3, 97])
    return dict(ul=ul, uh=uh, zl=zl, zh=zh)


def box_mesh(f, b, wl, wh, n_u, n_z):
    """Axis-aligned box in ring coordinates with dense front/back faces."""
    U, Z = np.meshgrid(np.linspace(b["ul"], b["uh"], n_u), np.linspace(b["zh"], b["zl"], n_z), indexing="ij")
    faces = []
    for wv in (wl, wh):
        V = to_world(f, U.ravel(), Z.ravel(), np.full(U.size, wv))
        faces.append(trimesh.Trimesh(V, grid(n_u, n_z), process=False))
    # four thin sides
    corners_u = [b["ul"], b["uh"]]
    corners_z = [b["zl"], b["zh"]]
    for side in range(4):  # quad corners ordered (a0,w0) (a0,w1) (a1,w0) (a1,w1)
        if side < 2:
            pts = [(corners_u[side], z, wv) for z in corners_z for wv in (wl, wh)]
        else:
            pts = [(u, corners_z[side - 2], wv) for u in corners_u for wv in (wl, wh)]
        P = np.array(pts)
        V = to_world(f, P[:, 0], P[:, 1], P[:, 2])
        faces.append(trimesh.Trimesh(V, [[0, 1, 3], [0, 3, 2]], process=False))
    m = trimesh.util.concatenate(faces)
    ctr = to_world(f, np.array([(b["ul"] + b["uh"]) / 2]), np.array([(b["zl"] + b["zh"]) / 2]),
                   np.array([(wl + wh) / 2]))[0]
    orient(m, m.triangles_center - ctr)  # convex box: outward = away from its centre
    return m


def wire_loop(f, uc, zc, wc, R, r, plane="ring"):
    """Torus of wire. plane='ring' lies in the ring's plane, 'cross' turned 90 deg."""
    tor = trimesh.creation.torus(major_radius=R, minor_radius=r, major_sections=48, minor_sections=12)
    V = tor.vertices  # torus around Z: x, y in its plane
    if plane == "ring":   # loop axis along w: (x, y, z) -> (u, z, w)
        U, Z, W = V[:, 0], V[:, 1], V[:, 2]
    else:                 # loop axis along u
        U, Z, W = V[:, 2], V[:, 1], V[:, 0]
    return trimesh.Trimesh(to_world(f, uc + U, zc + Z, wc + W), tor.faces, process=False)


def frame_bars(f, b, wl, wh, bar):
    """Metal frame: four bars around the glass, not a box that would hide it."""
    bars = [dict(ul=b["ul"] - bar, uh=b["uh"] + bar, zl=b["zh"], zh=b["zh"] + bar),
            dict(ul=b["ul"] - bar, uh=b["uh"] + bar, zl=b["zl"] - bar, zh=b["zl"]),
            dict(ul=b["ul"] - bar, uh=b["ul"], zl=b["zl"], zh=b["zh"]),
            dict(ul=b["uh"], uh=b["uh"] + bar, zl=b["zl"], zh=b["zh"])]
    return trimesh.util.concatenate([box_mesh(f, bb, wl, wh, 4, 4) for bb in bars])


def build(mesh, fit_ring=None):
    """Return (figure without ring/tag/hook, dict of parts drawn as vectors).
    fit_ring(f, figure) may refine the ring's measures against the photos."""
    f = ring_frame(mesh)
    v = mesh.vertices
    c = mesh.visual.vertex_colors[:, :3].astype(int)
    f["_v"], f["_c"] = v, c
    h = f["h"]
    b = tag_box(f)
    wm = (f["w0"] + f["w1"]) / 2
    # remove the generated ring (but not the white hands that hold it), the tag and the hook
    red = (c[:, 0] - c[:, 1] > 50)
    band = (f["w"] > f["w0"] - 0.012 * h) & (f["w"] < f["w1"] + 0.012 * h)
    on_ring = band & (f["r"] > f["r_in"] - 0.006 * h) & (f["r"] < f["r_out"] + 0.006 * h)
    hands = (c.min(1) > 140) & (np.ptp(c, 1) < 40)
    tag_zone = (np.abs(f["u"] - f["u0"]) < 0.09 * h) & (v[:, 2] > b["zl"] - 0.01 * h) & band
    # which side of the ring Mickey is on (his black head is behind the ring)
    black = c.max(1) < 45
    head = black & (v[:, 2] > 0.55 * h) & (np.abs(f["u"] - f["u0"]) < 0.3 * h)
    side = np.sign(np.median(f["w"][head]) - wm)
    # generated tag and hook remnants: top centre, from the ring to just before the ears
    toward = (f["w"] - wm) * side
    hook_zone = ~red & ~hands & (v[:, 2] > f["z0"] + f["r_in"] - TAG_MIN_HEIGHT * h - 0.04 * h) & \
        (np.abs(f["u"] - f["u0"]) < 0.11 * h) & \
        (toward < 0.085 * h) & (toward > -0.06 * h)
    drop = (on_ring & ~hands) | (tag_zone & (f["r"] < f["r_in"] + 0.01 * h)) | \
           (tag_zone & (f["r"] > f["r_out"] - 0.01 * h)) | hook_zone
    keep_faces = ~drop[mesh.faces].any(1)
    fig = mesh.submesh([np.nonzero(keep_faces)[0]], append=True)
    fig.remove_unreferenced_vertices()
    # cutting leaves floating chips of the old tag and hook: keep only real pieces
    pieces = fig.split(only_watertight=False)
    big = [p for p in pieces if len(p.faces) > MIN_PIECE * len(fig.faces)]
    fig = trimesh.util.concatenate(big)
    if fit_ring is not None:
        f = fit_ring(f, fig)
        wm = (f["w0"] + f["w1"]) / 2

    ring = ring_mesh(f)
    # the tag hangs just behind the ring, on Mickey's side; its top hides behind the ring band
    thick = 0.012 * h                                     # glass plate thickness
    bar = 0.008 * h                                       # frame bar width
    frame_depth = 0.026 * h                               # the bronze frame is chunky (side photos)
    face = f["w1"] if side > 0 else f["w0"]
    wt = face + side * (0.004 * h + frame_depth / 2)      # tag centre along the normal
    top_out = f["z0"] + f["r_out"]
    top_in = f["z0"] + f["r_in"]
    # the tag's top sits just behind the ring's inner edge (back photo); its height
    # comes from the model but never less than the real tag's (~0.12 of the figure)
    zh = top_in + 0.15 * (f["r_out"] - f["r_in"])
    b = dict(b, zh=zh, zl=min(b["zl"], zh - TAG_MIN_HEIGHT * h))
    gl = box_mesh(f, b, wt - thick / 2, wt + thick / 2, *TAG_GRID)
    frame = frame_bars(f, b, wt - frame_depth / 2, wt + frame_depth / 2, bar)
    um = (b["ul"] + b["uh"]) / 2
    wire = 0.0035 * h
    R = 0.024 * h
    hook = wire_loop(f, um, top_out + R * 0.8, wt, R, wire * 1.2, "cross")   # big loop, across the ring
    link_r = max((top_out - b["zh"] - bar) / 2, 0.008 * h)
    link = wire_loop(f, um, top_out - link_r, wt, link_r, wire, "ring")       # small loop down to the tag
    drop_loop = wire_loop(f, um, b["zl"] - bar - 0.011 * h, wt, 0.011 * h, wire * 0.8, "ring")
    print(f"parts: ring r {f['r_in'] / h:.3f}-{f['r_out'] / h:.3f} of height, thickness {(f['w1'] - f['w0']) / h:.3f}; "
          f"tag {(b['uh'] - b['ul']) / h:.3f} x {(b['zh'] - b['zl']) / h:.3f}; removed {drop.sum()} model vertices")
    f["tag"] = dict(b, wt=wt)
    return fig, dict(ring=ring, glass=gl, frame=frame, hook=hook, link=link, loop=drop_loop), f


# ---------------------------------------------------------------- hooks

from scipy.optimize import minimize                                    # noqa: E402

from twinlens.geometry import iou_views, project, raster, silhouettes  # noqa: E402
from twinlens.paint import despill, fill_from_neighbours            # noqa: E402

SHOE_LAYER = 0.07          # fraction of the height above the pedestal where the shoes stand
BLACK_MAX = 70
SHOE_VOTES = 2
FIT_SCALE = 0.25
PRINT_MAX_EXG = 25         # the print is light and barely green; the cloth seen through the glass is green
LEGS_TOP = 0.36            # fraction of the height where the shirt starts
PRINT_MIN = 120            # glass pixels brighter than this (and grey) are the tag's print


def refine_voxels(occ, ctx):
    """Cameras all sit above the pedestal top, so right above it nothing black can
    hide a shoe (jeans are blue, shoes brown, the black head is far above). A
    voxel there that lands on a black pixel in several photos is air in front of
    the black pedestal top: remove it."""
    b = ctx["pedestal"]
    if b is None:
        return occ
    views, frame, lo, step = ctx["views"], ctx["frame"], ctx["lo"], ctx["step"]
    idx = np.argwhere(occ)
    X = lo + (idx + 0.5) * step
    z = frame.to_up(X)[:, 2]
    layer = (z > b["z_top"]) & (z < b["z_top"] + SHOE_LAYER * b["h"])
    idx, X = idx[layer], X[layer]
    votes = np.zeros(len(X), np.uint8)
    for v in views:
        uv, _ = project(v, X)
        h, w = v["mask"].shape
        u, vv = np.round(uv).astype(int).T
        ins = (u >= 0) & (u < w) & (vv >= 0) & (vv < h)
        c = np.zeros((len(X), 3))
        c[ins] = despill(v["rgb"][vv[ins], u[ins]].astype(np.float64), v["cloth"])
        dark = ins & (c.max(1) < BLACK_MAX)
        dark[ins] &= v["mask_raw"][vv[ins], u[ins]]
        votes += dark
    occ = occ.copy()
    occ[tuple(idx[votes >= SHOE_VOTES].T)] = False
    print(f"  profile: shoe layer, removed {(votes >= SHOE_VOTES).sum()} voxels")
    return occ


def parts(mesh, ctx):
    views, frame = ctx["views"], ctx["frame"]
    sils = silhouettes(views, FIT_SCALE)

    def fit_ring(f, fig):
        """Centre, radii and faces of the ring so figure + ring match the photos'
        silhouettes (the model's red pixels only give a first guess)."""
        small = fig.simplify_quadric_decimation(face_count=60000) if len(fig.faces) > 60000 else fig
        figV = frame.from_up(small.vertices)
        fig_sil = [raster(v, figV, small.faces, m.shape, FIT_SCALE) for v, m in zip(views, sils)]
        keys = ("u0", "z0", "r_in", "r_out", "w0", "w1")
        x0 = np.array([f[k] for k in keys])

        def loss(x):
            g = dict(f, **dict(zip(keys, x)))
            if not (0 < g["r_in"] < g["r_out"]) or g["w0"] >= g["w1"]:
                return 1.0
            ring = ring_mesh(g, segments=180, radial=2)
            return 1 - iou_views(frame.from_up(ring.vertices), ring.faces, views, sils, FIT_SCALE, fig_sil).mean()

        before = 1 - loss(x0)
        d = 0.01 * f["h"]
        res = minimize(loss, x0, method="Nelder-Mead",
                       options=dict(maxiter=300, xatol=1e-5, fatol=1e-6,
                                    initial_simplex=np.vstack([x0, x0 + np.diag([d] * 6)])))
        print(f"  profile: ring fitted to the photos, IoU {before:.3f} -> {1 - res.fun:.3f}")
        return dict(f, **dict(zip(keys, res.x)))

    return build(mesh, fit_ring)


def clean_colours(out, ctx):
    f, sizes, views, frame, b = ctx["profile_state"], ctx["sizes"], ctx["views"], ctx["frame"], ctx["pedestal"]
    z_top = b["z_top"] if b else 0.0
    names = list(sizes)
    start = np.cumsum([0] + list(sizes.values()))
    col = out.visual.vertex_colors.copy()
    n_fig = sizes["object"]
    rgb = col[:n_fig, :3].astype(int)
    vz = out.vertices[:n_fig, 2]
    h = vz.max()
    fu = (out.vertices[:n_fig, :2] - f["o"]) @ f["e_u"]
    # the ring is a separate part now: red left on the figure is a stain from
    # photos that saw the ring behind an ear or a glove; only the tongue is red
    red = (rgb[:, 0] - rgb[:, 1] > 45) & (rgb[:, 0] > 110)
    mouth = (vz > 0.55 * h) & (vz < 0.70 * h) & (np.abs(fu - f["u0"]) < 0.06 * h)
    reddish_high = ((vz > 0.75 * h) & (rgb[:, 0] - rgb[:, 1] > 28)) | ((vz > 0.80 * h) & (rgb[:, 0] - rgb[:, 1] > 18))
    vn = out.vertex_normals[:n_fig]
    shoe_top = (vz > z_top + 0.002 * h) & (vz < z_top + 0.09 * h) & (vn[:, 2] > 0.35)
    stain = np.zeros(len(col), bool)
    # the hanging wire passes just behind the legs; photos that see it paint black
    # streaks on the jeans (nothing else between shoes and shirt is dark)
    legs = (vz > z_top + 0.02 * h) & (vz < LEGS_TOP * h) & (rgb.max(1) < 90)  # down to the cuffs; tan shoes are brighter
    legs |= (vz > z_top + 0.012 * h) & (vz < z_top + 0.02 * h) & (rgb.max(1) < 70)  # jeans meet shoes (the black pedestal top is just above z_top)
    legs |= (vz < (LEGS_TOP + 0.06) * h) & (vz > z_top + 0.012 * h) & (np.abs(fu - f["u0"]) < 0.10 * h) & (rgb.max(1) < 90)  # shirt hem, between the legs (the hands' black cuffs are further out)
    stain[:n_fig] = (red & ~mouth) | reddish_high | (shoe_top & (rgb.max(1) < 110)) | legs
    fixed = fill_from_neighbours(out, col[:, :3].astype(float), ~stain)
    col[stain, :3] = fixed[stain].astype(np.uint8)
    # flat-painted parts keep one hue: shoes mirror the green floor, gloves pick up
    # pink from the ring. Keep each point's brightness, restore the part's hue.
    rgb = col[:n_fig, :3].astype(float)
    lum = rgb.mean(1, keepdims=True)
    shoes = (vz > z_top + 0.002 * h) & (vz < z_top + 0.10 * h) & ~(rgb[:, 2] > rgb[:, 0] + 30) & (rgb.max(1) >= 70)
    tan = shoes & (rgb[:, 0] - rgb[:, 1] > 25) & (rgb[:, 1] - rgb[:, 2] > 15)
    if tan.any():
        rgb[shoes] = lum[shoes] * np.median(rgb[tan] / rgb[tan].mean(1, keepdims=True), 0)
    pinkish = (rgb.min(1) > 110) & (rgb[:, 0] - rgb[:, 1] > 8) & (rgb[:, 0] - rgb[:, 1] < 70) & \
        (np.abs(fu - f["u0"]) > 0.12 * h) & (vz > z_top + 0.15 * h)
    rgb[pinkish] = lum[pinkish]
    col[:n_fig, :3] = np.clip(rgb, 0, 255).astype(np.uint8)
    print(f"  profile: {stain.sum()} stain vertices repainted, {shoes.sum()} shoe and {pinkish.sum()} glove recoloured")
    for k, a, e in zip(names[1:], start[1:-1], start[2:]):
        if k == "glass":
            # the print from the one photo facing the tag most squarely (photos from
            # both sides of the glass, blended, smear it)
            nrm = np.r_[f["e_w"], 0]
            mid = out.vertices[a:e].mean(0)
            best = max(views, key=lambda v: abs((frame.to_up(v["center"][None])[0] - mid) @ nrm)
                       / np.linalg.norm(frame.to_up(v["center"][None])[0] - mid))
            uv, _ = project(best, frame.from_up(out.vertices[a:e]))
            # the full-resolution original: the strokes are only a few pixels wide
            from PIL import Image, ImageOps
            full = np.asarray(ImageOps.exif_transpose(Image.open(ctx["cfg"].photos / best["name"])).convert("RGB"))
            hh, ww = full.shape[:2]
            uv = uv * (ww / best["rgb"].shape[1])
            uu = np.clip(np.round(uv[:, 0]), 0, ww - 1).astype(int)
            vv = np.clip(np.round(uv[:, 1]), 0, hh - 1).astype(int)
            # raw colours: through clear glass the photo sees the green cloth, which
            # despill would turn grey and pass for print; the print is white/silver
            g = full[vv, uu].astype(float)
            printed = (g.min(1) > PRINT_MIN) & (g[:, 1] - np.maximum(g[:, 0], g[:, 2]) < PRINT_MAX_EXG)
            ed = out.edges_unique
            ed = ed[((ed >= a) & (ed < e)).all(1)] - a
            for _ in range(2):  # majority over each point and its neighbours: strokes stay, speckle goes
                pf = printed.astype(float)
                acc, cnt = pf.copy(), np.ones(len(pf))
                for i, j in ((0, 1), (1, 0)):
                    np.add.at(acc, ed[:, i], pf[ed[:, j]])
                    np.add.at(cnt, ed[:, i], 1)
                printed = acc / cnt > 0.5
            print(f"  profile: glass print {printed.mean() * 100:.0f}% of the tag from {best['name']}")
            col[a:e, 3] = GLASS
            col[a:e][printed, 3] = 255
            col[a:e][printed, :3] = 225
        elif k != "ring":
            col[a:e, 3] = METAL
    out.visual.vertex_colors = col
