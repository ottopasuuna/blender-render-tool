#! /bin/env python
import argparse
import cv2
from common_ops import add, extract_foreground, diff, blend_all

def show_img(img):
    while True:
        cv2.imshow('img', img)
        k = cv2.waitKey(1)
        if k == ord('q'):
            break

def load_image(path):
    '''Just an alias for imread'''
    return cv2.imread(path)

def save_image(img, path):
    cv2.imwrite(path, img)

def call_diff(args):
    im1 = load_image(args.image1)
    im2 = load_image(args.image2)
    res = diff(im1, im2)
    if args.output:
        save_image(res, args.output)
    else:
        show_img(res)

def call_add_clutter(args):
    bg_with_clutter = load_image(args.bg_with_clutter)
    bg = load_image(args.background)
    bg_with_subject = load_image(args.bg_with_subject)
    # clutter = diff(bg_with_clutter, bg, 1)
    subject = extract_foreground(bg_with_subject, bg, args.threshold)
    merged = add(subject, bg_with_clutter)
    if args.output:
        save_image(merged, args.output)
    else:
        show_img(merged)

def call_add(args):
    im1 = load_image(args.image1)
    im2 = load_image(args.image2)
    res = add(im1, im2)
    if args.output:
        save_image(res, args.output)
    else:
        show_img(res)

def call_extract_foreground(args):
    full_image = load_image(args.full_image)
    background = load_image(args.background)
    foreground = extract_foreground(full_image, background, args.threshold)
    if args.output:
        save_image(foreground, args.output)
    else:
        show_img(foreground)

def call_blend(args):
    images = [load_image(img) for img in args.images]
    res = blend_all(images)
    if args.output:
        save_image(res, args.output)
    else:
        show_img(res)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Multitool for post processing blender renders.')
    subparsers = parser.add_subparsers()

    # Image diff
    diff_parser = subparsers.add_parser('diff', help='calculate difference between two images.')
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

    # Add clutter
    # TODO: Support multiple bg_with_subject
    add_clutter_parser = subparsers.add_parser('add-clutter', help='Add background object details to a frame')
    add_clutter_parser.add_argument('bg_with_clutter', type=str,
            help='Image with all background items or "clutter".')
    add_clutter_parser.add_argument('background', type=str,
            help='Image of just the background')
    add_clutter_parser.add_argument('bg_with_subject', type=str,
            help='Image of the background with the subject')
    add_clutter_parser.add_argument('-t', '--threshold', default=8, type=int,
                             help='Threshold value to use during subject extraction.')
    add_clutter_parser.add_argument('-o', '--output')
    add_clutter_parser.set_defaults(func=call_add_clutter)

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

    args = parser.parse_args()

    if args:
        args.func(args)


if __name__ == '__main__':
    parse_arguments()
