#! /bin/env python
import os
import argparse
from multiprocessing import Pool, cpu_count

import cv2

from common_ops import (add, extract_foreground, diff, blend_all,
                        interpolate_flow, scale, denoise, add_noise,
                        blend, transparentOverlay)

NUM_CORES = cpu_count()


def parallelize(func, params_list):
    with Pool(processes=NUM_CORES) as pool:
        results_async = [pool.apply_async(func, params)
                         for params in params_list]
        results = [res.get() for res in results_async]
    return results


def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_paths(path):
    if isinstance(path, str):
        if os.path.isdir(path):
            paths = os.listdir(path)
        elif os.path.isfile(path):
            paths = [paths]
        else:
            raise os.FileNotFoundError('Specified path(s) does not exist')
        return paths
    else:
        return path

def show_img(img):
    while True:
        cv2.imshow('img', img)
        k = cv2.waitKey(1)
        if k == ord('q'):
            break

def show_images(imgs):
    for img in imgs:
        show_img(img)

def load_image(path):
    '''Just an alias for imread'''
    return cv2.imread(path, cv2.IMREAD_UNCHANGED)

def load_images(paths):
    paths = get_paths(paths)
    imgs = [load_image(path) for path in paths]
    return imgs

def save_image(img, path):
    cv2.imwrite(path, img)


def save_or_show(img, path):
    if path:
        save_image(img, path)
    else:
        show_img(img)

def output_to_basenames(input_paths, images, output_path):
    assert len(input_paths) == len(images)
    file_basenames = [os.path.basename(path) for path in input_paths]
    if len(images) > 1:
        if output_path:
            make_dir(output_path)
            out_paths = [os.path.join(output_path, basename)
                         for basename in file_basenames]
        else:
            out_paths = [None] * len(images)
    else:
        out_paths = [output_path]
    for (img, path) in zip(images, out_paths):
        save_or_show(img, path)

def call_diff(args):
    im1, im2 = load_images([args.image1, args.image2])
    res = diff(im1, im2)
    save_or_show(res, args.output)

def call_add_subjects(args):
    subject_paths = get_paths(args.subjects)
    output_path = args.output
    bg_path = args.background
    bg = load_image(bg_path)
    subject_imgs = load_images(subject_paths)
    merged = [transparentOverlay(subject, bg) for subject in subject_imgs]
    output_to_basenames(subject_paths, merged, output_path)

def call_add(args):
    im1 = load_image(args.image1)
    im2 = load_image(args.image2)
    res = transparentOverlay(im1, im2)
    if args.output:
        save_image(res, args.output)
    else:
        show_img(res)


def call_extract_foreground(args):
    full_image = load_image(args.full_image)
    background = load_image(args.background)
    foreground = extract_foreground(full_image, background, args.threshold)
    save_or_show(foreground, args.output)


def call_blend(args):
    images = [load_image(img) for img in args.images]
    res = blend_all(images)
    save_or_show(res, args.output)

def _interp(path1, name_2, path3, func):
    frame1 = load_image(path1)
    frame3 = load_image(path3)
    frame2 = func(frame1, frame3)
    return (frame2, name_2)

def call_interpolate(args):
    name_format = args.format
    parent_dir = args.frame_dir
    start, end = args.start, args.end
    assert(start < end)
    curr = start
    make_dir(args.output)

    if args.mode == 'flow':
        interp_func = interpolate_flow
    elif args.mode == 'blend':
        interp_func = blend
    else:
        raise ValueError('Invalid interpolation mode')

    step = 2  # TODO: allow different step sizes, 4, 8...
    # Grouping files to interpolate
    path_groups = []
    while curr < end:
        frame1 = os.path.join(parent_dir, name_format.format(curr))
        frame3 = os.path.join(parent_dir, name_format.format(curr + step))
        to_interp = name_format.format(curr + step // 2)
        path_groups.append((frame1, to_interp, frame3, interp_func))
        curr += step

    # Interpolation
    interp_frames = parallelize(_interp, path_groups)
    for (frame, name) in interp_frames:
        save_image(frame, os.path.join(args.output, name))


def call_scale(args):
    output_path = args.output
    images = load_images(args.images)
    template = images[0]
    if args.percent:
        height, width = int(
            template.shape[0] * args.percent), int(template.shape[1] * args.percent)
    else:
        width, height = args.width, args.height
    params = [(img, width, height, args.mode) for img in images]
    scaled = parallelize(scale, params)
    output_to_basenames(args.images, scaled, output_path)


def call_denoise(args):
    images = [load_image(path) for path in args.images]
    params = [(img, args.strength, args.mode) for img in images]
    denoised = parallelize(denoise, params)
    output_to_basenames(args.images, denoised, args.output)

def call_test(args):
    image = load_image(args.image)
    res = add_noise(image)
    if args.output:
        save_image(res, args.output)
    else:
        show_img(res)


def parse_arguments():
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

    # Image Add
    add_parser = subparsers.add_parser('add', help='Add two images together')
    add_parser.add_argument('image1', type=str)
    add_parser.add_argument('image2', type=str)
    add_parser.add_argument('-o', '--output', required=False)
    add_parser.set_defaults(func=call_add)

    # Add subjects
    add_subjects_parser = subparsers.add_parser(
        'add-subjects', help='Add subject frames to static background')
    add_subjects_parser.add_argument('background', type=str,
                                     help='Image of the background')
    add_subjects_parser.add_argument('subjects', type=str, nargs='+',
                                     help='Image(s) of just the subject, transparent everywhere else')
    add_subjects_parser.add_argument('-o', '--output')
    add_subjects_parser.set_defaults(func=call_add_subjects)

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
    interp_parser.add_argument('frame_dir', type=str)
    interp_parser.add_argument('-s', '--start', required=True, type=int)
    interp_parser.add_argument('-e', '--end', required=True, type=int)
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
    scale_parser.set_defaults(func=call_scale)

    # Image denoising
    denoise_parser = subparsers.add_parser('denoise', help='Denoise images')
    denoise_parser.add_argument('images', nargs='+', type=str)
    denoise_parser.add_argument('-s', '--strength', default=5, type=int,
                                help='The strength of the filter')
    denoise_parser.add_argument('-m', '--mode', default='fastNL')
    denoise_parser.add_argument('-o', '--output')
    denoise_parser.set_defaults(func=call_denoise)

    # Misc parser for testing
    test_parser = subparsers.add_parser(
        'test', help='Misc for testing, dont use')
    test_parser.add_argument('image', type=str)
    test_parser.add_argument('-o', '--output')
    test_parser.set_defaults(func=call_test)

    args = parser.parse_args()

    try:
        args.func(args)
    except AttributeError as e:
        parser.print_help()


if __name__ == '__main__':
    parse_arguments()
