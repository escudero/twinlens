"""Step 1: upright and resize the photos, key out the cloth."""
import cv2
import numpy as np
from PIL import Image, ImageOps

EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def photos(cfg):
    return sorted(p for p in cfg.photos.iterdir() if p.suffix in EXTS)


def colour(bgr):
    """Per pixel: its shade, the chromaticity (R, B) / (R + G + B), which does
    not change with brightness, and its excess green G - max(R, B)."""
    f = cv2.GaussianBlur(bgr, (5, 5), 0).astype(np.float32)
    b, g, r = f[:, :, 0], f[:, :, 1], f[:, :, 2]
    s = f.sum(2) + 3.0
    return np.dstack([r / s, b / s]), g - np.maximum(r, b)


def border(shape, frac):
    h, w = shape
    bh, bw = max(1, int(h * frac)), max(1, int(w * frac))
    ring = np.zeros((h, w), bool)
    ring[:bh] = ring[-bh:] = True
    ring[:, :bw] = ring[:, -bw:] = True
    return ring


def cloth_colour(shade, exg, ring):
    """The cloth as this photo shows it, measured on the border strip: the mean
    and spread of its shade, and how green it is. The 10 % of border pixels
    that differ most are left out, in case something else reaches the edge."""
    x = shade[ring]
    d = np.linalg.norm(x - np.median(x, 0), axis=1)
    core = d < np.percentile(d, 90)
    cov = np.cov(x[core].T) + np.eye(2) * 1e-7
    return x[core].mean(0), np.linalg.inv(cov), float(np.median(exg[ring][core]))


def key_cloth(bgr, cfg):
    """Object mask and warnings. A pixel is background when it is about as green
    as the cloth (excess green at least `exg_frac` of the cloth's) AND has the
    cloth's own shade (within `shade_tol` of the cloth's spread). Green alone
    keys out green paint and parts that mirror the cloth; shade alone keys out
    dark parts lit by the cloth's green light. The cloth is measured on each
    photo, so nothing is set by hand."""
    shade, exg = colour(bgr)
    ring = border(exg.shape, cfg.cloth_border)
    mu, icov, exg_cloth = cloth_colour(shade, exg, ring)
    dev = shade - mu
    dist = np.sqrt(np.einsum("...i,ij,...j->...", dev, icov, dev))
    bg = (exg >= cfg.exg_frac * exg_cloth) & (dist < cfg.shade_tol)
    obj = (~bg).astype(np.uint8) * 255
    obj = cv2.morphologyEx(obj, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(obj, 8)
    keep = np.zeros_like(obj)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= cfg.min_blob_frac * obj.size:
            keep[lab == i] = 255
    inv = cv2.bitwise_not(keep)  # fill pinholes, keep real openings
    n, lab, stats, _ = cv2.connectedComponentsWithStats(inv, 4)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < 0.0005 * obj.size:
            keep[lab == i] = 255

    warnings = []
    if exg_cloth < cfg.min_cloth_exg:
        warnings.append("the photo border does not look like green cloth")
    elif (not_cloth := (keep[ring] > 0).mean()) > 0.02:
        warnings.append(f"{100 * not_cloth:.0f}% of the photo border is not cloth")
    if keep[:2].any() or keep[-2:].any() or keep[:, :2].any() or keep[:, -2:].any():
        warnings.append("the object touches the photo edge")
    return keep, warnings


def preview_tile(bgr, mask, name, warnings, width=480):
    """The photo cropped to the object, the cut-away part dimmed, the mask
    outlined: what the key took away stays visible."""
    vis = bgr.copy()
    vis[mask == 0] = (vis[mask == 0] * 0.3).astype(np.uint8)
    cnts, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(vis, cnts, -1, (0, 255, 255), 3)
    ys, xs = np.nonzero(mask)
    if len(ys):
        h, w = mask.shape
        vis = vis[max(ys.min() - 60, 0):min(ys.max() + 60, h), max(xs.min() - 60, 0):min(xs.max() + 60, w)]
    tile = cv2.resize(vis, (width, round(width * vis.shape[0] / vis.shape[1])), interpolation=cv2.INTER_AREA)
    cv2.putText(tile, name, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    for k, text in enumerate(warnings):
        cv2.putText(tile, "! " + text, (8, 52 + 24 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
    return tile


def run(cfg):
    cfg.images.mkdir(parents=True, exist_ok=True)
    cfg.masks.mkdir(parents=True, exist_ok=True)
    cfg.masks_ok.unlink(missing_ok=True)  # new masks: `run` asks for a check again
    tiles, flagged = [], 0
    files = photos(cfg)
    if len(files) < 8:
        raise SystemExit(f"found {len(files)} photos in {cfg.photos}; take 15-40 around the object")
    for p in files:
        im = ImageOps.exif_transpose(Image.open(p)).convert("RGB")
        s = cfg.long_side / max(im.size)
        if s < 1:
            im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
        name = p.stem + ".jpg"
        im.save(cfg.images / name, quality=95)
        bgr = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)
        m, warnings = key_cloth(bgr, cfg)
        cv2.imwrite(str(cfg.masks / (p.stem + ".png")), m)
        tiles.append(preview_tile(bgr, m, p.stem, warnings))
        flagged += bool(warnings)
        print(f"  {p.name}: object {100 * (m > 0).mean():.1f}% of the photo"
              + "".join(f"  ! {w}" for w in warnings))
    rows = []
    for i in range(0, len(tiles), 5):
        row = tiles[i:i + 5]
        row += [np.zeros((1, tiles[0].shape[1], 3), np.uint8)] * (5 - len(row))
        h = max(t.shape[0] for t in row)
        rows.append(np.hstack([cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 0, cv2.BORDER_CONSTANT) for t in row]))
    cfg.out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(cfg.out / "masks_preview.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 90])
    if flagged:
        print(f"  {flagged} photo(s) with warnings (marked in red in the preview)")
    print(f"  check the key in {cfg.out / 'masks_preview.jpg'}: the yellow outline should hug the object")
