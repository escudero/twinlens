"""Step 7: studio scene in Blender (run by Blender, not by Python):

    blender -b -P twinlens/scene.py -- --model M.ply --out DIR [--stills] [--video]
                                           [--photo-views cameras.json] [--seconds 8] [--fps 30]

Imports the painted model (vertex colours; alpha tags 160 = metal, 80 = glass),
builds a seamless backdrop with three soft area lights, a turntable animation,
and saves DIR/model.blend. --photo-views renders through the real photo cameras.
"""
import math
import sys
from pathlib import Path

import bpy
from mathutils import Matrix, Vector

args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []


def arg(name, default=None, cast=str):
    return cast(args[args.index(name) + 1]) if name in args else default


MODEL = Path(arg("--model", "out/model.ply"))
OUT = Path(arg("--out", "out"))
SECONDS = arg("--seconds", 8, int)
FPS = arg("--fps", 30, int)
RES = (arg("--width", 1080, int), arg("--height", 1350, int))   # 4:5 portrait, the feed format
CAM_ELEVATION_DEG = 12
CAM_LENS_MM = 70
BACKDROP_RGB = (0.93, 0.92, 0.90)
SAMPLES = 64
KEY_W = 450               # key light (W); fill and rim follow it
WORLD_STRENGTH = 0.6
VIEW_TRANSFORM = "Standard"   # AgX washes light, warm colours (a cream face) towards white


def reset():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def import_model():
    bpy.ops.wm.ply_import(filepath=str(MODEL.resolve()))
    obj = bpy.context.selected_objects[0]
    obj.name = "Model"
    # smooth, but keep real edges (base rim) crisp
    bpy.ops.object.shade_smooth_by_angle(angle=math.radians(35))
    # scale to a 2-unit tall object so lights and camera are model-independent
    h = obj.dimensions.z
    s = 2.0 / h
    obj.scale = (s, s, s)
    bpy.ops.object.transform_apply(scale=True)
    mat = bpy.data.materials.new("PhotoColour")
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    attr = nt.nodes.new("ShaderNodeVertexColor")
    if obj.data.color_attributes:
        attr.layer_name = obj.data.color_attributes[0].name
    else:  # uncoloured mesh (geometry preview): plain clay
        obj.data.color_attributes.new("Col", "BYTE_COLOR", "POINT")
        for c in obj.data.color_attributes[0].data:
            c.color = (0.7, 0.7, 0.7, 1)
        attr.layer_name = "Col"
    # colours come from photos that already carry soft light: keep shading gentle
    hsv = nt.nodes.new("ShaderNodeHueSaturation")
    hsv.inputs["Saturation"].default_value = 1.0
    hsv.inputs["Value"].default_value = 1.0
    nt.links.new(attr.outputs["Color"], hsv.inputs["Color"])
    nt.links.new(hsv.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.55
    bsdf.inputs["Coat Weight"].default_value = 0.08  # a hint of painted resin; more shows every bump
    # vertex alpha tags the glass tag (80) and its metal frame and hook (160)
    metal = nt.nodes.new("ShaderNodeBsdfPrincipled")
    metal.inputs["Base Color"].default_value = (0.30, 0.24, 0.17, 1)  # aged bronze
    metal.inputs["Metallic"].default_value = 1.0
    metal.inputs["Roughness"].default_value = 0.35
    # thin flat glass: see straight through it (a refracting principled BSDF in
    # EEVEE picks up the white floor and turns opaque white) plus a sharp sheen
    clear = nt.nodes.new("ShaderNodeBsdfTransparent")
    clear.inputs["Color"].default_value = (0.86, 0.92, 0.89, 1)
    sheen = nt.nodes.new("ShaderNodeBsdfGlossy")
    sheen.inputs["Roughness"].default_value = 0.05
    fres = nt.nodes.new("ShaderNodeFresnel")
    fres.inputs["IOR"].default_value = 1.5
    glass = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(fres.outputs[0], glass.inputs["Fac"])
    nt.links.new(clear.outputs["BSDF"], glass.inputs[1])
    nt.links.new(sheen.outputs["BSDF"], glass.inputs[2])
    is_tag = nt.nodes.new("ShaderNodeMath")
    is_tag.operation = "LESS_THAN"
    is_tag.inputs[1].default_value = 0.8
    is_glass = nt.nodes.new("ShaderNodeMath")
    is_glass.operation = "LESS_THAN"
    is_glass.inputs[1].default_value = 0.45
    nt.links.new(attr.outputs["Alpha"], is_tag.inputs[0])
    nt.links.new(attr.outputs["Alpha"], is_glass.inputs[0])
    mix_tag = nt.nodes.new("ShaderNodeMixShader")
    mix_glass = nt.nodes.new("ShaderNodeMixShader")
    out = nt.nodes["Material Output"]
    nt.links.new(is_glass.outputs[0], mix_glass.inputs["Fac"])
    nt.links.new(metal.outputs["BSDF"], mix_glass.inputs[1])
    nt.links.new(glass.outputs["Shader"], mix_glass.inputs[2])
    nt.links.new(is_tag.outputs[0], mix_tag.inputs["Fac"])
    nt.links.new(bsdf.outputs["BSDF"], mix_tag.inputs[1])
    nt.links.new(mix_glass.outputs["Shader"], mix_tag.inputs[2])
    nt.links.new(mix_tag.outputs["Shader"], out.inputs["Surface"])
    try:  # EEVEE: let the glass refract
        mat.use_raytrace_refraction = True
        mat.surface_render_method = "DITHERED"
    except AttributeError:
        pass
    obj.data.materials.append(mat)
    return obj


def backdrop():
    bpy.ops.mesh.primitive_plane_add(size=40, location=(0, 0, 0))
    floor = bpy.context.object
    floor.name = "Backdrop"
    # bend the back edge up into a cyclorama
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.subdivide(number_cuts=40)
    bpy.ops.object.mode_set(mode="OBJECT")
    for v in floor.data.vertices:
        y = v.co.y
        if y > 4:
            t = y - 4
            v.co.z = t * t / 6
    bpy.ops.object.shade_smooth()
    mat = bpy.data.materials.new("Backdrop")
    mat.use_nodes = True
    b = mat.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*BACKDROP_RGB, 1)
    b.inputs["Roughness"].default_value = 0.9
    floor.data.materials.append(mat)


def lights():
    def area(name, loc, energy, size, color=(1, 1, 1)):
        bpy.ops.object.light_add(type="AREA", location=loc)
        L = bpy.context.object
        L.name = name
        L.data.energy = energy
        L.data.size = size
        L.data.color = color
        d = Vector((0, 0, 1.0)) - L.location
        L.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
    # soft and fairly flat: the vertex colours already carry the photos' light
    area("Key", (-3.5, -4.0, 4.5), KEY_W, 4.0, (1.0, 0.98, 0.95))
    area("Fill", (4.5, -3.0, 2.0), KEY_W * 0.6, 5.0, (0.97, 0.98, 1.0))
    area("Rim", (0.5, 4.5, 4.0), KEY_W * 0.6, 2.5)
    world = bpy.data.worlds.new("World")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = WORLD_STRENGTH
    bpy.context.scene.world = world


def camera(obj):
    target = bpy.data.objects.new("Target", None)
    target.location = (0, 0, obj.dimensions.z * 0.5)
    bpy.context.collection.objects.link(target)
    cam_data = bpy.data.cameras.new("Cam")
    cam_data.lens = CAM_LENS_MM
    cam_data.dof.use_dof = True
    cam_data.dof.focus_object = target
    cam_data.dof.aperture_fstop = 4.0
    cam = bpy.data.objects.new("Cam", cam_data)
    bpy.context.collection.objects.link(cam)
    # distance so the object fills ~80% of the frame height
    vfov = 2 * math.atan(cam_data.sensor_width / 2 / CAM_LENS_MM) * RES[1] / RES[0]
    dist = obj.dimensions.z / 0.8 / 2 / math.tan(vfov / 2) * 1.05
    el = math.radians(CAM_ELEVATION_DEG)
    cam.location = (0, -dist * math.cos(el), target.location.z + dist * math.sin(el))
    c = cam.constraints.new("TRACK_TO")
    c.target = target
    c.track_axis = "TRACK_NEGATIVE_Z"
    c.up_axis = "UP_Y"
    bpy.context.scene.camera = cam


def setup_render():
    sc = bpy.context.scene
    sc.render.engine = "BLENDER_EEVEE"
    sc.eevee.taa_render_samples = SAMPLES
    try:
        sc.eevee.use_raytracing = True
    except AttributeError:
        pass
    sc.render.resolution_x, sc.render.resolution_y = RES
    sc.render.fps = FPS
    sc.view_settings.view_transform = VIEW_TRANSFORM  # AgX washed the cream face to white
    sc.view_settings.look = "None"


def main():
    reset()
    obj = import_model()
    backdrop()
    lights()
    camera(obj)
    setup_render()
    sc = bpy.context.scene
    sc.frame_start, sc.frame_end = 1, SECONDS * FPS
    # turntable: the object spins, camera/backdrop/lights stay put
    obj.rotation_euler = (0, 0, 0)
    obj.keyframe_insert("rotation_euler", frame=1)
    obj.rotation_euler = (0, 0, 2 * math.pi)
    obj.keyframe_insert("rotation_euler", frame=SECONDS * FPS + 1)
    _linearize(obj)
    OUT.mkdir(parents=True, exist_ok=True)
    # the default Solid view draws everything grey: open showing the vertex colours
    for scr in bpy.data.screens:
        for area in scr.areas:
            for sp in area.spaces:
                if sp.type == "VIEW_3D":
                    sp.shading.type = "SOLID"
                    sp.shading.color_type = "VERTEX"
                    sp.shading.light = "STUDIO"
    bpy.ops.wm.save_as_mainfile(filepath=str((OUT / "model.blend").resolve()))

    if "--photo-views" in args:  # render through the real photo cameras (see 05_compare.py)
        import json
        spec = json.loads(Path(arg("--photo-views")).read_text())
        k = 2.0 / spec["height"]  # import_model scaled the object to 2 units
        sc.frame_set(1)
        cam = sc.camera
        cam.constraints.clear()
        cam.data.dof.use_dof = False
        for c in spec["cameras"]:
            R = Matrix(c["R"])  # COLMAP: x right, y down, looks along +z
            C = Vector(c["C"]) * k
            rot = R.transposed() @ Matrix(((1, 0, 0), (0, -1, 0), (0, 0, -1)))
            cam.matrix_world = Matrix.Translation(C) @ rot.to_4x4()
            w, h = c["w"], c["h"]
            cam.data.sensor_fit = "HORIZONTAL"
            cam.data.sensor_width = 36
            cam.data.lens = c["f"] / w * 36
            cam.data.shift_x = (w / 2 - c["cx"]) / w
            cam.data.shift_y = (c["cy"] - h / 2) / w
            sc.render.resolution_x, sc.render.resolution_y = w // 2, h // 2
            sc.render.image_settings.file_format = "PNG"
            sc.render.filepath = str((OUT / "photo_views" / c["name"].replace(".jpg", ".png")).resolve())
            bpy.ops.render.render(write_still=True)
    if "--stills" in args:
        (OUT / "stills").mkdir(parents=True, exist_ok=True)
        sc.render.image_settings.file_format = "PNG"
        for deg in (0, 90, 180, 270):
            sc.frame_set(1 + round(deg / 360 * SECONDS * FPS))
            sc.render.filepath = str((OUT / "stills" / f"angle_{deg:03d}.png").resolve())
            bpy.ops.render.render(write_still=True)
    if "--video" in args:  # Blender 5 dropped FFMPEG as an image format: PNG frames + ffmpeg
        import shutil
        import subprocess
        frames = OUT / "frames"
        shutil.rmtree(frames, ignore_errors=True)
        frames.mkdir(parents=True)
        sc.render.image_settings.file_format = "PNG"
        sc.render.filepath = str(frames.resolve()) + "/"
        bpy.ops.render.render(animation=True)
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", str(FPS),
                        "-i", str(frames / "%04d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-crf", "18", str(OUT / "turntable.mp4")], check=True)


def _linearize(obj):
    """Blender 4.4+ layered actions hide fcurves behind slots/channelbags."""
    act = obj.animation_data.action
    curves = []
    if hasattr(act, "layers") and act.layers:
        for layer in act.layers:
            for strip in layer.strips:
                for bag in strip.channelbags:
                    curves += list(bag.fcurves)
    elif hasattr(act, "fcurves"):
        curves = list(act.fcurves)
    for fc in curves:
        for k in fc.keyframe_points:
            k.interpolation = "LINEAR"


main()
