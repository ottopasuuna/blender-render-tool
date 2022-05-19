import argparse
import yaml

from .core import load_images, save_images

from .tools import (AddOverlayTool, InterpolateTool, ScaleTool,
                   DenoiseTool, BlenderRender)


class LoadImages(object):
    name = 'load'

    def __init__(self, path):
        self.path = path

    def __str__(self):
        return f"load {self.path}"

    @classmethod
    def from_dict(cls, dct):
        if 'path' not in dct:
            raise ValueError("\"load\" operation requires an argument \"path\"")
        return cls(path=dct['path'])

    def __call__(self, images):
        return load_images(self.path)

class SaveImages(object):
    name = 'save'

    def __init__(self, path):
        self.path = path

    def __str__(self):
        return f"save {self.path}"

    @classmethod
    def from_dict(cls, dct):
        if dct is None or 'path' not in dct:
            raise ValueError("\"load\" operation requires an argument \"path\"")
        return cls(path=dct['path'])

    def __call__(self, images):
        save_images(images, self.path)

def build_tool(pipeline_stage_dict):
    build_map = {
        tool.name: tool
        for tool in [LoadImages,
                     SaveImages,
                     ScaleTool,
                     InterpolateTool,
                     DenoiseTool,
                     AddOverlayTool,
                     ]
    }
    if len(pipeline_stage_dict) > 1:
        raise ValueError("Pipeline stages can only contain one operation.")
    tool_name = list(pipeline_stage_dict.keys())[0]
    if tool_name not in build_map:
        raise ValueError(f"Operation \"{tool_name}\" not recognized.")
    params_dct = pipeline_stage_dict[tool_name]
    return build_map[tool_name].from_dict(params_dct)


def run_pipeline(args):
    command_file = args.pipeline_file
    with open(command_file) as f:
        pipeline = yaml.load(f, Loader=yaml.BaseLoader)
    if not isinstance(pipeline, list):
        raise RuntimeError('Improper format for YAML pipeline. Must be a list of operations')

    print(f"Read pipeline: {pipeline}")
    images = None
    for stage in pipeline:
        tool = build_tool(stage)
        print(tool)
        images = tool(images)
