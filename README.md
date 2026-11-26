# TwinLens

**Photos of an object on a green cloth → a painted 3D model in Blender.**

Put the object on a green chroma-key cloth, walk around it taking 15–40 photos with your phone, and run one command. You get a painted 3D model (`.ply`), a Blender studio scene (`.blend`), a turntable video and side-by-side checks against your own photos.

Everything runs on a laptop CPU. One optional step, the shape generator, runs on a free Kaggle GPU, sent and collected by the tool itself.

![photo vs render](docs/compare.jpg)

![turntable stills](docs/stills.jpg)

## Quick start

```bash
git clone <this repo> && cd twinlens
uv venv --python 3.12 .venv && uv pip install --python .venv -e .
# Blender 4.2+ on PATH (tested with 5.3); Kaggle API token in ~/.kaggle (kaggle.com > Settings > API)

.venv/bin/twinlens run ~/photos/my-object --work work/my-object
```

`run` stops once after the green key: check `work/my-object/out/masks_preview.jpg` (the yellow outline should hug the object) and answer `y` to go on. `--yes` skips the question.

Results land in `work/my-object/out/`:

| File | What |
|---|---|
| `model.ply` | the model: vertex colours from the photos, Z up, standing on z = 0 |
| `model.blend` | studio scene: seamless backdrop, three soft lights, turntable animation |
| `turntable.mp4`, `stills/` | 8 s turntable (1080 × 1350) and four angles |
| `compare.jpg` | your photos next to renders through the same cameras |
| `model_carved.ply` | the model built from the photos alone (no GPU step) |
| `masks_preview.jpg`, `check_*.jpg` | the green key (cut-away parts dimmed, warnings in red) and the silhouette fit, for checking |

Each step saves its results in the work folder. `run` skips finished steps, so it resumes after an interruption. Steps can also be run one at a time: `twinlens carve PHOTOS --work WORK`.

Options: `--gpu none` (photos only, no Kaggle), `--seeds 1,2,3,4` (generated candidates), `--pedestal` (the object stands on a round base, which is redrawn exactly), `--height-cm 12` (real scale), `--profile file.py` (object-specific rules, see below), `--no-video`, `--yes` (don't stop to check the masks).

## How to take the photos

- **Cloth:** a green cloth under and behind the object, filling the photo out to its edges: each photo's border is where the cloth's colour is measured. Green details on the object are fine as long as they are not the cloth's shade. Soft, even light.
- **Photos:** 15–40 of them, walking all the way around at about the same height, a bit above the object (10–20°). Keep 10–20° between shots, so neighbouring photos overlap.
- **Framing:** keep the whole object in every photo. Lock the focus and exposure if your phone allows it, and **turn digital zoom off**. Zoom is read from EXIF and corrected, but it costs resolution.
- **Top surfaces:** a second ring of photos from ~35° above helps them, and the gaps between parts.

## How it works

1. **prepare**: upright the photos (EXIF) and key the cloth, measured on each photo's border. A pixel is cloth when it is about as green as the cloth ("excess green" `G − max(R, B)`, at least 40 % of the cloth's) **and** has the cloth's own shade (chromaticity, within 6 times the cloth's own spread). Then `run` stops for a look at the masks.
2. **cameras**: where each photo was taken from. ALIKED keypoints on the object only, LightGlue matches between neighbours, COLMAP's mapper. The focal prior comes from EXIF (35 mm equivalent × digital zoom).
3. **depth**: MoGe-2 depth per photo (CPU).
4. **carve**: voxel carving from the silhouettes, then from depth (a voxel a photo saw through is air). This gives a complete painted model on its own. It also finds the upright frame: the photos circle the object, so their plane gives the vertical. The table is found from each photo's lowest silhouette pixel.
5. **generate**: Hunyuan3D-2mv turns 4 photos (front, sides, back; picked by angle, background removed, green spill cleaned) into a smooth, closed mesh. It runs on a free Kaggle T4: the tool uploads a private dataset, pushes a private kernel, waits and downloads one mesh per seed.
6. **align**: each generated mesh is placed where the photos say. Rotation, translation and per-axis scale maximise silhouette IoU over all photos. Fine relief is smoothed off, so that the photo paint draws small details, not relief sculpted slightly off. Every candidate is painted, and the best one is kept by silhouette IoU and photo-consistency (how much the photos disagree on each point's colour).
7. **scene**: Blender studio scene, stills and video.
8. **compare**: renders through the real cameras next to the photos, scored by SSIM on the object.

**Painting:** each vertex takes the colour of the photo that agrees most with the others, among the 5 that see it most squarely (z-buffer visibility). This weighted medoid lets a photo that lands on the wrong part get outvoted, and no colour is invented by mixing. Green spill is removed, never past grey. Points the photos disagree on take their neighbours' colour.

## Profiles: rules for one object

The core knows nothing about any particular object. A profile is a Python file with optional hooks:

```python
def refine_voxels(occ, ctx): ...      # step 4: extra carving rules
def parts(mesh, ctx): ...             # step 6: swap pieces for exact shapes -> (mesh, {name: part}, state)
def clean_colours(mesh, ctx): ...     # step 6: colour rules after painting
```

`ctx` holds the config, the photos with their cameras (`views`), the upright `frame` and the pedestal, if any. Profiles can tag materials in the vertex alpha: 160 = metal, 80 = glass.

[`examples/mickey/profile.py`](examples/mickey/profile.py) is the figurine this tool was built on (`--pedestal --profile examples/mickey/profile.py`). Its ring, glass tag and wire hook are drawn as vectors, measured on the model, and the ring is fitted to the photos' silhouettes. That made the ring's lettering sharp, which painting the generated ring could not do. Its colour rules know that only the tongue is red and that shoes and gloves are flat colours.

## What we learned building it

- **Classic SIFT** placed 8 of 20 photos of smooth plastic. Learned features on the object crop (ALIKED + LightGlue) placed all 20.
- **Phones record the optical focal length even with digital zoom.** Our photos were at 1.75×, which threw off the first camera solve.
- **Silhouettes cannot see concavities.** With all photos at one height, carving leaves tents between parts, and thin parts (ears) come out boxy. Depth carving helps; a generative shape aligned to the photos fixes it.
- **"Green" is not enough to find the cloth.** Excess green alone keyed out green paint and the edges of dark glossy parts that mirror the cloth: the horns of a brown figurine lost their outline. The cloth's shade alone keys out dark hollows lit only by the cloth's green light. Asking for both, with the cloth measured on each photo's border, keeps all three.
- **Glossy surfaces mirror the cloth.** Black plastic turned teal, and white turned mint. Despill bugs along the way: over-subtracting green turned whites magenta, and per-channel medians invented blue. The fix: never despill past grey, and medoid colours.
- **Generated relief is a little off the real one.** Its shading doubled the painted eyes and letters. Smoothing the fine relief fixed it.
- **Blender:** AgX washed light colours to white, so the scene uses the Standard view transform. Coplanar faces z-fight. Blender 5 dropped FFMPEG output, so the video is made from frames with ffmpeg.

## Limitations

- **Generated shape:** it is a plausible guess conditioned on 4 photos, not a measurement. Faces and fine sculpt can differ from the real object. `compare.jpg` shows how much.
- **Small print:** it comes out soft. Thin wires and transparent parts (glass) need a profile.
- **Captures:** it assumes photos taken around the object at a similar height, with the first photo being the front.
- **Licences:** Hunyuan3D-2 has its own licence (Tencent Hunyuan Community License, with territory limits). It is downloaded and run in your own Kaggle account, not shipped here.

## Credits

COLMAP / pycolmap · ALIKED and LightGlue (cvg) · MoGe-2 (Microsoft) · Hunyuan3D-2mv (Tencent) · Blender · trimesh, scikit-image, SciPy, OpenCV · Kaggle for the free GPU.
