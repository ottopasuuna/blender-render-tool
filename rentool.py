#! /bin/env python
import argparse
import textwrap
import os

from src.pipeline import run_pipeline
from src.tools import (AddOverlayTool, InterpolateTool, ScaleTool,
                   DenoiseTool, AddNoiseTool, DiffTool, BlendTool,
                   ExtractForegroundTool, BlenderRender)

def init_makefile(args):
    if os.path.exists('Makefile'):
        print('A Makefile already exists, skipping to avoid overwriting it...')
        return
    blendfile_name = args.blendfile
    if blendfile_name is None:
        # find blend files in cwd
        blendfiles = [f for f in os.listdir() if f.endswith('.blend')]
        if blendfiles:
            blendfile_name = blendfiles[0]
        else:
            blendfile_name = '<path to blendfile>'

    makefile_str = textwrap.dedent('''
        BLEND_FILE={blendfile_name}
        RENDER_OUT=Rendered_frames
        PIPELINE_FILE=pipeline.yaml
        RENTOOL_OUT=rentool_output
        FINAL_DIR=$(RENTOOL_OUT)
        VIDEO_OUT=video.avi
        FPS=24
        RENDER_START=1


        post: rentool video

        render: $(BLEND_FILE)
        \tblender -b $(BLEND_FILE) -s $(RENDER_START) -j 2 -o "$(RENDER_OUT)/####.png" -a

        $(PIPELINE_FILE):
        \techo "- load $(RENDER_OUT)" > $(PIPELINE_FILE)
        \techo "- interpolate" >> $(PIPELINE_FILE)
        \techo "- save $(RENTOOL_OUT)" >> $(PIPELINE_FILE)

        rentool: $(PIPELINE_FILE)
        \trentool pipeline $(PIPELINE_FILE)

        video:
        \tffmpeg -r $(FPS) -i $(FINAL_DIR)/%04d.png  -c:v rawvideo -pix_fmt bgr24 $(VIDEO_OUT)

        play:
        \tmpv $(VIDEO_OUT) --loop

        clean:
        \trm -rf $(FINAL_DIR)  $(VIDEO_OUT) $(RENTOOL_OUT) $(PIPELINE_FILE)
        '''.format(blendfile_name=blendfile_name))
    with open('Makefile', 'w') as makefile:
        makefile.write(makefile_str)


def parse_arguments(arguments=None):
    parser = argparse.ArgumentParser(
        description='Multitool for post processing blender renders.')
    subparsers = parser.add_subparsers()

    _tools = [AddOverlayTool, InterpolateTool, ScaleTool,
              DenoiseTool, AddNoiseTool, DiffTool, BlendTool,
              ExtractForegroundTool, BlenderRender]
    for tool in _tools:
        tool.build_standalone_parser(subparsers)

    pipeline_parser = subparsers.add_parser('pipeline',
                                            help='Read commands from a pipeline file')
    pipeline_parser.add_argument('pipeline_file', type=str)
    pipeline_parser.set_defaults(func=run_pipeline)

    makefile_parser = subparsers.add_parser('init',
                                            help='Initialize a makefile for using the rentool workflow')
    makefile_parser.add_argument('blendfile', nargs='?',
                                 help='.blend file to point to for rendering. '
                                      'Defaults to the first blend file in the current working directory')
    makefile_parser.set_defaults(func=init_makefile)

    args = parser.parse_args(arguments)
    return args, parser

if __name__ == '__main__':
    args, parser = parse_arguments()
    if args and hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()
