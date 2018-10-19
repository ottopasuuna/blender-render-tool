#! /bin/env python
import argparse

import pipeline
from tools import (AddOverlayTool, InterpolateTool, ScaleTool,
                   DenoiseTool, AddNoiseTool, DiffTool, BlendTool,
                   ExtractForegroundTool)


def parse_arguments(arguments=None):
    parser = argparse.ArgumentParser(
        description='Multitool for post processing blender renders.')
    subparsers = parser.add_subparsers()

    DiffTool.build_standalone_parser(subparsers)
    AddOverlayTool.build_standalone_parser(subparsers)
    InterpolateTool.build_standalone_parser(subparsers)
    ScaleTool.build_standalone_parser(subparsers)
    DenoiseTool.build_standalone_parser(subparsers)
    AddNoiseTool.build_standalone_parser(subparsers)
    BlendTool.build_standalone_parser(subparsers)
    ExtractForegroundTool.build_standalone_parser(subparsers)

    pipeline_parser = subparsers.add_parser('pipeline',
                                            help='Read commands from a pipeline file')
    pipeline_parser.add_argument('pipeline_file', type=str)
    pipeline_parser.set_defaults(func=pipeline.run_pipeline)

    args = parser.parse_args(arguments)
    return args, parser

if __name__ == '__main__':
    args, parser = parse_arguments()
    if args and hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()
