# This script is passed to Blender from the cli and manages the rendering operaions

import json
import os
import sys

import bpy

def render(output_file,
           frame=None, # Leave current frame alone
           scene=None,
           layer=""):
    if scene is not None:
        bpy.context.window.scene = bpy.data.scenes.get(scene)
    scene = bpy.context.scene
    scene.render.filepath = output_file
    scene.render.image_settings.file_format = 'PNG' # TODO
    scene.render.use_overwrite = False
    if frame is not None:
        scene.frame_set(frame)
    if layer in scene.view_layers:
        # Disable all the layers we're not interested in
        for name, Layer in scene.view_layers.items():
            if name == layer:
                continue
            Layer.use = False
    bpy.ops.render.render(write_still=True)


print("json args: {}".format(sys.argv[-1]))
settings = json.loads(sys.argv[-1])
output_file = settings['output']
bpy.context.scene.render.filepath = output_file

frames = settings.get('frames', None)
if frames:
    if len(frames) > 1:
        output_dir, output_ext = os.path.splitext(output_file)
    for frame in frames:
        if len(frames) > 1:
            name_format = '{{0:04d}}{}'.format(output_ext)
            output_file = os.path.join(output_dir, name_format.format(frame))
        render(output_file, frame=frame, layer=settings['layer'], scene=settings['scene'])
else:
    render(output_file, scene=settings['scene'], layer=settings['layer'])
