import argparse
import yaml

from .core import load_images, save_images

from .tools import (AddOverlayTool, InterpolateTool, ScaleTool,
                   DenoiseTool, BlenderRender)


def run_load_images(args, images):
    return load_images(args.images)


def run_save_images(args, images):
    save_images(images, args.output)


def parse_args(args):
    parser = argparse.ArgumentParser(description='Multitool for post processing blender renders.')
    subparsers = parser.add_subparsers()

    load_images_parser = subparsers.add_parser('load')
    load_images_parser.add_argument('images')
    load_images_parser.set_defaults(func=run_load_images)

    save_images_parser = subparsers.add_parser('save')
    save_images_parser.add_argument('output')
    save_images_parser.set_defaults(func=run_save_images)

    AddOverlayTool.build_pipeline_parser(subparsers)
    InterpolateTool.build_pipeline_parser(subparsers)
    ScaleTool.build_pipeline_parser(subparsers)
    DenoiseTool.build_pipeline_parser(subparsers)
    BlenderRender.build_pipeline_parser(subparsers)

    return parser.parse_args(args)


def run_pipeline(args):
    command_file = args.pipeline_file
    with open(command_file) as f:
        pipeline = yaml.load(f)
    if not isinstance(pipeline, list):
        raise RuntimeError('Improper format for YAML pipeline')
    pipeline = [line.split() for line in pipeline]
    if pipeline[0][0] not in ('load', 'render'):
        raise RuntimeError('First line must be load or render.')

    images = None
    for step in pipeline:
        args = parse_args(step)
        print(step[0])
        images = args.func(args, images)
