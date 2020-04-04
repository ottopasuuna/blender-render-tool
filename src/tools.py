import os
from .blender import (split_frames_per_host, remote_blender, blender,
                      copy_results_from_host, cleanup_host)
from .core import (load_images, parallelize, load_image, save_images,
                  pipeline_wrapper, save_or_show)
from .common_ops import (transparentOverlay, interpolate_flow, blend,
                        scale, denoise, add_noise, diff, blend_all,
                        extract_foreground)


class Tool:

    @classmethod
    def build_pipeline_parser(cls, subparsers):
        raise NotImplementedError

    @classmethod
    def build_standalone_parser(cls, subparsers):
        raise NotImplementedError

    @classmethod
    def run(cls, args, images):
        raise NotImplementedError


class AddOverlayTool(Tool):

    @classmethod
    def build_standalone_parser(cls, subparsers):
        overlay_parser = cls.build_pipeline_parser(subparsers)
        overlay_parser.add_argument('images', type=str, nargs='+',
                                    help='Image(s) of just the foreground, transparent everywhere else')
        overlay_parser.add_argument('-o', '--output')
        overlay_parser.set_defaults(func=pipeline_wrapper,
                                    tool=cls.run)
        return overlay_parser

    @classmethod
    def build_pipeline_parser(cls, subparsers):
        overlay_parser = subparsers.add_parser(
            'add-overlay', help='Add transparent overlay frames to static background')
        overlay_parser.add_argument('background', type=str,
                help='Image of the background')
        overlay_parser.set_defaults(func=AddOverlayTool.run)
        return overlay_parser

    @classmethod
    def run(cls, args, images):
        bg = load_image(args.background)
        params = [(subject, bg) for subject in images]
        merged = parallelize(transparentOverlay, params)
        return merged

class InterpolateTool(Tool):

    @classmethod
    def build_pipeline_parser(cls, subparsers):
        interp_parser = subparsers.add_parser('interpolate', help='Interpolate frames')
        interp_parser.add_argument('-m', '--mode', default='flow')
        interp_parser.set_defaults(func=cls.run)
        return interp_parser

    @classmethod
    def build_standalone_parser(cls, subparsers):
        interp_parser = cls.build_pipeline_parser(subparsers)
        interp_parser.add_argument('images', type=str)
        interp_parser.add_argument('-o', '--output', default='interp_frames')
        interp_parser.add_argument('-f', '--format', default='{0:04d}.png')
        interp_parser.set_defaults(func=cls._run)
        return interp_parser

    @classmethod
    def _run(cls, args):
        images = load_images(args.images)
        frames = cls.run(args, images)
        save_images(frames, args.output)

    @classmethod
    def run(cls, args, images):
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
        all_frames = [None]*(len(interp_frames) + len(images))
        all_frames[::2] = images
        all_frames[1::2] = interp_frames
        return all_frames


class ScaleTool(Tool):

    @classmethod
    def build_pipeline_parser(cls, subparsers):
        scale_parser = subparsers.add_parser('scale', help='Resize frames')
        scale_parser.add_argument('-m', '--mode', default='lanczos',
                help='Interpolation mode to use when resizing.')
        scale_parser.add_argument('-p', '--percent', type=float,
                help=('Percentage change as a float'))
        scale_parser.add_argument('--width', type=int)
        scale_parser.add_argument('--height', type=int)
        scale_parser.set_defaults(func=cls.run)
        return scale_parser

    @classmethod
    def build_standalone_parser(cls, subparsers):
        scale_parser = cls.build_pipeline_parser(subparsers)
        scale_parser.add_argument('images', nargs='+', type=str)
        scale_parser.add_argument('-o', '--output')
        scale_parser.set_defaults(func=pipeline_wrapper, tool=cls.run)
        return scale_parser

    @classmethod
    def run(cls, args, images):
        template = images[0]
        if args.percent:
            height, width = int(template.shape[0]*args.percent), int(template.shape[1]*args.percent)
        else:
            width, height = args.width, args.height
        params = [(img, width, height, args.mode) for img in images]
        scaled = parallelize(scale, params)
        return scaled


class DenoiseTool(Tool):

    @classmethod
    def build_pipeline_parser(cls, subparsers):
        denoise_parser = subparsers.add_parser('denoise', help='Denoise images')
        denoise_parser.add_argument('-s', '--strength', default=5, type=int,
                help='The strength of the filter')
        denoise_parser.add_argument('-m', '--mode', default='fastNL')
        denoise_parser.set_defaults(func=cls.run)
        return denoise_parser

    @classmethod
    def build_standalone_parser(cls, subparsers):
        denoise_parser = cls.build_pipeline_parser(subparsers)
        denoise_parser.add_argument('images', nargs='+', type=str)
        denoise_parser.add_argument('-o', '--output')
        denoise_parser.set_defaults(func=pipeline_wrapper, tool=cls.run)

    @classmethod
    def run(cls, args, images):
        params = [(img, args.strength, args.mode) for img in images]
        denoised = parallelize(denoise, params)
        return denoised


class AddNoiseTool(Tool):

    @classmethod
    def build_standalone_parser(cls, subparsers):
        addnoise_parser = subparsers.add_parser(
            'add-noise', help='Add noise to an image')
        addnoise_parser.add_argument('image', type=str)
        addnoise_parser.add_argument('-o', '--output')
        addnoise_parser.set_defaults(func=cls._run)
        return addnoise_parser

    @classmethod
    def _run(cls, args):
        image = load_image(args.image)
        res = add_noise(image)
        save_or_show(res, args.output)


class DiffTool(Tool):

    @classmethod
    def build_standalone_parser(cls, subparsers):
        diff_parser = subparsers.add_parser(
            'diff', help='calculate difference between two images.')
        diff_parser.add_argument('image1', type=str,
                                 help='The image with all the items in it.')
        diff_parser.add_argument('image2', type=str,
                                 help='The image with just the background.')
        diff_parser.add_argument('-o', '--output')
        diff_parser.set_defaults(func=cls._run)
        return diff_parser

    @classmethod
    def _run(cls, args):
        im1, im2 = load_images([args.image1, args.image2])
        res = diff(im1, im2)
        save_or_show(res, args.output)


class BlendTool(Tool):

    @classmethod
    def build_standalone_parser(cls, subparsers):
        blend_parser = subparsers.add_parser('blend', help='blend frames together')
        blend_parser.add_argument('images', nargs='+')
        blend_parser.add_argument('-o', '--output')
        blend_parser.set_defaults(func=cls._run)
        return blend_parser

    @classmethod
    def _run(cls, args):
        images = [load_image(img) for img in args.images]
        res = blend_all(images)
        save_or_show(res, args.output)


class ExtractForegroundTool(Tool):

    @classmethod
    def build_standalone_parser(cls, subparsers):
        ef_parser = subparsers.add_parser('extract-foreground',
                                          help='Extract the foreground items from a background')
        ef_parser.add_argument('full_image', type=str,
                               help='The image with all the items in it.')
        ef_parser.add_argument('background', type=str,
                               help='The image with just the background.')
        ef_parser.add_argument('-t', '--threshold', default=1, type=int,
                               help='Threshold value to use during foreground extraction.')
        ef_parser.add_argument('-o', '--output')
        ef_parser.set_defaults(func=cls._run)
        return ef_parser

    @classmethod
    def _run(cls, args):
        full_image = load_image(args.full_image)
        background = load_image(args.background)
        foreground = extract_foreground(full_image, background, args.threshold)
        save_or_show(foreground, args.output)


class BlenderRender(Tool):

    @classmethod
    def build_pipeline_parser(cls, subparsers):
        render_parser = subparsers.add_parser('render',
                                              help='Wrapper to Blender for rendering a project')
        render_parser.add_argument('blend_file', type=str,
                                   help='.blend file to render')
        render_parser.add_argument('-s', '--start_frame', type=int, default=1,
                                   help='Frame to start rendering from')
        render_parser.add_argument('-e', '--end_frame', type=int,
                                   help='Frame to end rendering at')
        render_parser.add_argument('-d', '--distribute', type=str, nargs='+', default=['localhost'],
                                   help='Distribute work to another machine')
        render_parser.add_argument('-j', '--jump', type=int, default=1,
                                   help='Number of frames to skip.')
        render_parser.set_defaults(func=cls.pipeline_run, output='render_output')
        return render_parser

    @classmethod
    def build_standalone_parser(cls, subparsers):
        parser = cls.build_pipeline_parser(subparsers)
        parser.add_argument('-o', '--output', type=str, default='render_output',
                            help='Output directory to save render results to.')
        parser.set_defaults(func=cls._run)
        return parser

    @classmethod
    def pipeline_run(cls, args, _):
        cls._run(args)
        images = load_images(args.output)
        return images

    @classmethod
    def _run(cls, args):
        frames = range(args.start_frame, args.end_frame+1, args.jump)
        frames_per_host = split_frames_per_host(frames, args.distribute)
        print('Frames per host: {}'.format(frames_per_host))

        processes = []
        for host in args.distribute:
            _frames = frames_per_host[host]
            if host == 'localhost':
                p = blender(blend_file=os.path.expanduser(args.blend_file),
                            output=args.output,
                            frames=_frames)
            else:
                p = remote_blender(host,
                                   blend_file=args.blend_file,
                                   output=args.output,
                                   frames=_frames)
            processes.append(p)
        for p in processes:
            p.wait()
        for host in args.distribute:
            if host != 'localhost':
                copy_results_from_host(host, args.output)
                cleanup_host(host, args.blend_file, args.output)
