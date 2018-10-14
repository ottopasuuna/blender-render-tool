#! /bin/env python
import argparse

from common_ops import (extract_foreground, diff, blend_all,
                        add_noise)
from core import (get_paths, show_img, save_images,
                  load_image, load_images, save_image,
                  save_or_show, output_to_basenames)
import pipeline
from pipeline import (run_subjects_pipe, run_scale, run_denoise,
                      run_interpolate)


def call_diff(args):
    im1, im2 = load_images([args.image1, args.image2])
    res = diff(im1, im2)
    save_or_show(res, args.output)


def pipeline_wrapper(args):
    paths = get_paths(args.images)
    images = load_images(paths)
    results = args.tool(args, images)
    output_to_basenames(paths, results, args.output)


def call_extract_foreground(args):
    full_image = load_image(args.full_image)
    background = load_image(args.background)
    foreground = extract_foreground(full_image, background, args.threshold)
    save_or_show(foreground, args.output)


def call_blend(args):
    images = [load_image(img) for img in args.images]
    res = blend_all(images)
    save_or_show(res, args.output)


def call_interpolate(args):
    images = load_images(args.images)
    frames = run_interpolate(args, images)
    save_images(frames, args.output)


def call_test(args):
    image = load_image(args.image)
    res = add_noise(image)
    if args.output:
        save_image(res, args.output)
    else:
        show_img(res)


def parse_arguments(arguments=None):
    parser = argparse.ArgumentParser(
        description='Multitool for post processing blender renders.')
    subparsers = parser.add_subparsers()

    # Image diff
    diff_parser = subparsers.add_parser(
        'diff', help='calculate difference between two images.')
    diff_parser.add_argument('image1', type=str,
                             help='The image with all the items in it.')
    diff_parser.add_argument('image2', type=str,
                             help='The image with just the background.')
    diff_parser.add_argument('-o', '--output')
    diff_parser.set_defaults(func=call_diff)

    # Add subjects
    add_subjects_parser = subparsers.add_parser(
        'add-subjects', help='Add subject frames to static background')
    add_subjects_parser.add_argument('background', type=str,
                                     help='Image of the background')
    add_subjects_parser.add_argument('images', type=str, nargs='+',
                                     help='Image(s) of just the subject, transparent everywhere else')
    add_subjects_parser.add_argument('-o', '--output')
    add_subjects_parser.set_defaults(func=pipeline_wrapper,
                                     tool=run_subjects_pipe)

    # Extract foreground
    extract_foreground_parser = subparsers.add_parser('extract-foreground',
                                                      help='Extract the foreground items from a background')
    extract_foreground_parser.add_argument('full_image', type=str,
                                           help='The image with all the items in it.')
    extract_foreground_parser.add_argument('background', type=str,
                                           help='The image with just the background.')
    extract_foreground_parser.add_argument('-t', '--threshold', default=1, type=int,
                                           help='Threshold value to use during foreground extraction.')
    extract_foreground_parser.add_argument('-o', '--output')
    extract_foreground_parser.set_defaults(func=call_extract_foreground)

    # Image Blend
    blend_parser = subparsers.add_parser('blend', help='blend frames together')
    blend_parser.add_argument('images', nargs='+')
    blend_parser.add_argument('-o', '--output')
    blend_parser.set_defaults(func=call_blend)

    # Frame interpolation
    interp_parser = subparsers.add_parser(
        'interpolate', help='Interpolate frames')
    interp_parser.add_argument('images', type=str)
    interp_parser.add_argument('-o', '--output', default='interp_frames')
    interp_parser.add_argument('-f', '--format', default='{0:04d}.png')
    interp_parser.add_argument('-m', '--mode', default='flow')
    interp_parser.set_defaults(func=call_interpolate)

    # Image scaling
    scale_parser = subparsers.add_parser('scale', help='Resize frames')
    scale_parser.add_argument('images', nargs='+', type=str)
    scale_parser.add_argument('-m', '--mode', default='lanczos',
                              help='Interpolation mode to use when resizing.')
    scale_parser.add_argument('-p', '--percent', type=float,
                              help=('Percentage change as a float'))
    scale_parser.add_argument('--width', type=int)
    scale_parser.add_argument('--height', type=int)
    scale_parser.add_argument('-o', '--output')
    scale_parser.set_defaults(func=pipeline_wrapper, tool=run_scale)

    # Image denoising
    denoise_parser = subparsers.add_parser('denoise', help='Denoise images')
    denoise_parser.add_argument('images', nargs='+', type=str)
    denoise_parser.add_argument('-s', '--strength', default=5, type=int,
                                help='The strength of the filter')
    denoise_parser.add_argument('-m', '--mode', default='fastNL')
    denoise_parser.add_argument('-o', '--output')
    denoise_parser.set_defaults(func=pipeline_wrapper, tool=run_denoise)

    # Misc parser for testing
    test_parser = subparsers.add_parser(
        'test', help='Misc for testing, dont use')
    test_parser.add_argument('image', type=str)
    test_parser.add_argument('-o', '--output')
    test_parser.set_defaults(func=call_test)

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
