#! /bin/env python
import argparse

from src.pipeline import run_pipeline
from src.tools import (AddOverlayTool, InterpolateTool, ScaleTool,
                   DenoiseTool, AddNoiseTool, DiffTool, BlendTool,
                   ExtractForegroundTool, BlenderRender)


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

    args = parser.parse_args(arguments)
    return args, parser

if __name__ == '__main__':
    args, parser = parse_arguments()
    if args and hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()
