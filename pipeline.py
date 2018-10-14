import argparse
import yaml

from core import load_images, parallelize, load_image, save_images
from common_ops import (transparentOverlay, interpolate_flow, blend,
                        scale, denoise)


def run_load_images(args, images):
    return load_images(args.images)


def run_subjects_pipe(args, images):
    bg = load_image(args.background)
    params = [(subject, bg) for subject in images]
    merged = parallelize(transparentOverlay, params)
    return merged

def run_interpolate(args, images):
    if args.mode == 'flow':
        interp_func = interpolate_flow
    elif args.mode == 'blend':
        interp_func = blend
    else:
        raise ValueError('Invalid interpolation mode')

    # Grouping files to interpolate
    curr = 0
    end = len(images) - 1
    groups = []
    while curr < end:
        frame1 = images[curr]
        frame3 = images[curr+1]
        groups.append((frame1, frame3))
        curr += 1

    # Interpolation
    interp_frames = parallelize(interp_func, groups)
    all_frames = interp_frames + images
    all_frames[::2] = images
    all_frames[1::2] = interp_frames
    return all_frames


def run_save_images(args, images):
    save_images(images, args.path)


def run_scale(args, images):
    template = images[0]
    if args.percent:
        height, width = int(template.shape[0]*args.percent), int(template.shape[1]*args.percent)
    else:
        width, height = args.width, args.height
    params = [(img, width, height, args.mode) for img in images]
    scaled = parallelize(scale, params)
    return scaled


def run_denoise(args, images):
    params = [(img, args.strength, args.mode) for img in images]
    denoised = parallelize(denoise, params)
    return denoised


def parse_args(args):
    parser = argparse.ArgumentParser(description='Multitool for post processing blender renders.')
    subparsers = parser.add_subparsers()

    load_images_parser = subparsers.add_parser('load')
    load_images_parser.add_argument('images')
    load_images_parser.set_defaults(func=run_load_images)

    save_images_parser = subparsers.add_parser('save')
    save_images_parser.add_argument('path')
    save_images_parser.set_defaults(func=run_save_images)

    # Add subjects
    add_subjects_parser = subparsers.add_parser('add-subjects', help='Add subject frames to static background')
    add_subjects_parser.add_argument('background', type=str,
            help='Image of the background')
    add_subjects_parser.set_defaults(func=run_subjects_pipe)

    # Frame interpolation
    interp_parser = subparsers.add_parser('interpolate', help='Interpolate frames')
    interp_parser.add_argument('-m', '--mode', default='flow')
    interp_parser.set_defaults(func=run_interpolate)

    # Image scaling
    scale_parser = subparsers.add_parser('scale', help='Resize frames')
    scale_parser.add_argument('-m', '--mode', default='lanczos',
            help='Interpolation mode to use when resizing.')
    scale_parser.add_argument('-p', '--percent', type=float,
            help=('Percentage change as a float'))
    scale_parser.add_argument('--width', type=int)
    scale_parser.add_argument('--height', type=int)
    scale_parser.set_defaults(func=run_scale)

    # Image denoising
    denoise_parser = subparsers.add_parser('denoise', help='Denoise images')
    denoise_parser.add_argument('-s', '--strength', default=5, type=int,
            help='The strength of the filter')
    denoise_parser.add_argument('-m', '--mode', default='fastNL')
    denoise_parser.set_defaults(func=run_denoise)

    return parser.parse_args(args)


def run_pipeline(args):
    command_file = args.pipeline_file
    with open(command_file) as f:
        pipeline = yaml.load(f)
    if not isinstance(pipeline, list):
        raise RuntimeError('Improper format for YAML pipeline')
    pipeline = [line.split() for line in pipeline]
    if pipeline[0][0] != 'load':
        raise RuntimeError('First line must be load.')

    images = None
    for step in pipeline:
        args = parse_args(step)
        print(step[0])
        images = args.func(args, images)

