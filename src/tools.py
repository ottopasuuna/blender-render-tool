import argparse
import os
import textwrap
from .blender import (split_frames_per_host, remote_blender, blender,
                      copy_results_from_host, cleanup_host)
from .core import (load_images, parallelize, load_image, save_images,
                  pipeline_wrapper, save_or_show, download_url, MODEL_CACHE_DIR)
from .common_ops import (transparentOverlay, interpolate_flow, blend,
                        scale, dnn_upscale, denoise, add_noise, diff, blend_all,
                        extract_foreground)


class Parameter(object):

    def __init__(self, cli_flags=None, type=str, default=None, help=None, nargs=None):
        assert cli_flags is not None, "Parameter: cli_flags must be specified as minimum"
        self.cli_flags = cli_flags
        self.type = type
        self.default = default
        self.help = help
        self.nargs = nargs

class Tool:
    name='GENERIC TOOL'
    description='GENERIC TOOL'
    description_long = ''

    params = {}

    def __str__(self):
        params = ["{}={}".format(name, value) for name, value in self.__dict__.items()]
        param_str = ' '.join(params)
        return f"{self.name} {param_str}"

    @classmethod
    def build_pipeline_parser(cls, subparsers):
        raise NotImplementedError

    @classmethod
    def build_standalone_parser(cls, subparsers):
        raise NotImplementedError

    @classmethod
    def run(cls, args, images):
        raise NotImplementedError

    @classmethod
    def from_dict(cls, dct):
        kwargs = {}
        for name, param in cls.params.items():
            if isinstance(dct, dict):
                dct_val = dct.get(name, param.default)
            else:
                dct_val = param.default
            if param.nargs in {'*', '+'}:
                param_type = lambda l: [param.type(v) for v in l]
            else:
                param_type = param.type
            kwargs[name] = param_type(dct_val) if dct_val is not None else None
        return cls(**kwargs)

    @classmethod
    def from_args(cls, args):
        kwargs = {name: getattr(args, name, param.default) for name, param in cls.params.items()}
        return cls(**kwargs)

    @classmethod
    def build_parser(cls, subparsers):
        tool_parser = subparsers.add_parser(cls.name, help=cls.description,
                                            formatter_class=argparse.RawDescriptionHelpFormatter,
                                            description=cls.description_long)
        for name, param in cls.params.items():
            tool_parser.add_argument(*param.cli_flags, type=param.type,
                                     default=param.default, help=param.help, nargs=param.nargs)
        return tool_parser


class AddOverlayTool(Tool):
    name = 'add-overlay'
    description = 'Add transparent overlay frames to static background'

    params = {
        'background': Parameter(cli_flags=['background'], type=str,
                                help='Image of the background')
    }

    def __init__(self, background):
        self.background = background

    @classmethod
    def build_standalone_parser(cls, subparsers):
        overlay_parser = cls.build_parser(subparsers)
        overlay_parser.add_argument('images', type=str, nargs='+',
                                    help='Image(s) of just the foreground, transparent everywhere else')
        overlay_parser.add_argument('-o', '--output')
        overlay_parser.set_defaults(func=pipeline_wrapper, tool=cls)
        return overlay_parser

    def __call__(self, images):
        bg = load_image(self.background)
        params = [(subject, bg) for subject in images]
        merged = parallelize(transparentOverlay, params)
        return merged

class InterpolateTool(Tool):
    name = 'interpolate'
    description = 'Interpolate frames'

    params = {
        'mode': Parameter(cli_flags=['-m', '--mode'], type=str, default='flow',
                          help='Method of computing iterpolated frames.')
    }

    def __init__(self, mode):
        self.mode = mode

    @classmethod
    def build_standalone_parser(cls, subparsers):
        interp_parser = cls.build_parser(subparsers)
        interp_parser.add_argument('images', type=str)
        interp_parser.add_argument('-o', '--output', default='interp_frames')
        interp_parser.set_defaults(func=pipeline_wrapper, tool=cls)
        return interp_parser

    def __call__(self, images):
        if self.mode == 'flow':
            interp_func = interpolate_flow
        elif self.mode == 'blend':
            interp_func = blend
        else:
            raise ValueError(f'Invalid interpolation mode \"{self.mode}\"')

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
    name = 'scale'
    description = 'Resize frames'
    description_long = textwrap.dedent('''
        Resize frames.

        Size can be specified by width and height in pixels, or percentages (as a float).
        For example, "--percent 2"  will double the resolution,
        "-p 0.5" will half the resolution, etc.

        Different modes are available with the --mode option:
        cubic
        lanczos
        edsr:    A DNN upsampler, quite slow. Supports 2, 3, 4 scaling factors
        espcn:   A DNN upsampler, much faster than edsr. Supports 2, 3, 4 factors
        fsrcnn:  A DNN upsampler, similar to espcn. Supports 2, 3, 4 factors
        lapsrn:  A DNN upsampler, in between edsr and espcn. Supports 2, 4, 8 factors

        DNN based upsamplers will have to download pre-trained models,
        stored at {}
        '''.format(MODEL_CACHE_DIR))

    params = {
        'mode': Parameter(cli_flags=['-m', '--mode'], type=str, default='lanczos',
                          help='Interpolation mode to use when resizing'),
        'percent': Parameter(cli_flags=['-p', '--percent'], type=float,
                             help='Percentage change as a float'),
        'width': Parameter(cli_flags=['--width'], type=int),
        'height': Parameter(cli_flags=['--height'], type=int),
    }

    def __init__(self, mode, percent, width, height):
        self.mode = mode
        self.percent = percent
        self.width = width
        self.height = height

    @classmethod
    def build_standalone_parser(cls, subparsers):
        scale_parser = cls.build_parser(subparsers)
        scale_parser.add_argument('images', nargs='+', type=str)
        scale_parser.add_argument('-o', '--output')
        scale_parser.set_defaults(func=pipeline_wrapper, tool=cls)
        return scale_parser

    def __call__(self, images):
        template = images[0]
        if self.percent:
            height, width = int(template.shape[0]*self.percent), int(template.shape[1]*self.percent)
        else:
            width, height = self.width, self.height
        if self.mode in {'edsr', 'espcn', 'fsrcnn', 'lapsrn'}:
            supported_factors = {'edsr': {2, 3, 4},
                                 'espcn': {2, 3, 4},
                                 'fsrcnn': {2, 3, 4},
                                 'lapsrn': {2, 4, 8}}
            factor = int(self.percent)
            if self.percent not in supported_factors[self.mode]:
                raise ValueError("Supported factors for {} are {}".format(self.mode, supported_factors[self.mode]))
            self.get_dnn_model(self.mode, factor)
            params = [(img, self.mode, self.percent) for img in images]
            scaled = parallelize(dnn_upscale, params)
        else:
            params = [(img, width, height, self.mode) for img in images]
            scaled = parallelize(scale, params)
        return scaled

    @staticmethod
    def get_dnn_model(model, factor):
        base_url = {'edsr': 'https://github.com/Saafke/EDSR_Tensorflow/raw/master/models/EDSR_x{factor}.pb',
                    'espcn': 'https://github.com/fannymonori/TF-ESPCN/raw/master/export/ESPCN_x{factor}.pb',
                    'fsrcnn': 'https://github.com/Saafke/FSRCNN_Tensorflow/raw/master/models/FSRCNN_x{factor}.pb',
                    'lapsrn': 'https://github.com/fannymonori/TF-LapSRN/raw/master/export/LapSRN_x{factor}.pb',
                    }[model]
        url = base_url.format(factor=factor)
        filename = url.split('/')[-1]
        save_path = os.path.join(MODEL_CACHE_DIR, filename)
        if not os.path.exists(save_path):
            os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
            download_url(url, save_path)

class DenoiseTool(Tool):
    name = 'denoise'
    description = 'Denoise images'

    params = {
        'mode': Parameter(cli_flags=['-m', '--mode'], type=str, default='fastNL',
                          help='Method to use for denoising'),
        'strength': Parameter(cli_flags=['-s', '--strength'], type=int, default=5,
                              help='The strength of the filter')
    }

    def __init__(self, mode, strength):
        self.mode = mode
        self.strength = strength

    @classmethod
    def build_standalone_parser(cls, subparsers):
        denoise_parser = cls.build_parser(subparsers)
        denoise_parser.add_argument('images', nargs='+', type=str)
        denoise_parser.add_argument('-o', '--output')
        denoise_parser.set_defaults(func=pipeline_wrapper, tool=cls)

    def __call__(self, images):
        params = [(img, self.strength, self.mode) for img in images]
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
    name = 'render'
    description = 'Wrapper to Blender for rendering a project'

    params = {
        'blend_file': Parameter(cli_flags=['blend_file'], type=str,
                                help='.blend file to render'),
        'frames': Parameter(cli_flags=['-f', '--frames'], type=str, default='1',
                            help='Frames to render. Can be a comma separated list, or python range syntax. '
                                  'Ex: 1,4,5,7:20:2'),
        'scene': Parameter(cli_flags=['-S', '--scene'], type=str,
                           help='Scene to render'),
        'layer': Parameter(cli_flags=['-l', '--layer'], type=str, default="",
                           help='View Layer to render'),
        'distribute': Parameter(cli_flags=['-d', '--distribute'], type=str,
                                default=['localhost'], nargs='+',
                                help='Distribute work to another machine'),
        'output': Parameter(cli_flags=['-o', '--output'], type=str, default='render_output',
                            help='Output path to write rendered frame(s) to')
    }

    def __init__(self, blend_file, frames, scene, layer, distribute, output):
        self.blend_file = blend_file
        self.frames = frames
        self.scene = scene
        self.layer = layer
        self.distribute = distribute
        self.output = output

    @classmethod
    def build_standalone_parser(cls, subparsers):
        parser = cls.build_parser(subparsers)
        parser.set_defaults(func=cls._run)
        return parser

    @classmethod
    def _run(cls, args):
        tool = cls.from_args(args)
        tool(None)

    @staticmethod
    def parse_frames(frame_string):
        frames = []
        for sequence in frame_string.split(','):
            if ':' in sequence:
                r = sequence.split(':')
                start = int(r[0])
                end = int(r[1])
                step = int(r[2]) if len(r) >= 3 else 1
                frames.extend(list(range(start, end, step)))
            else:
                frames.append(int(sequence))
        return frames

    def __call__(self, images):
        frames = self.parse_frames(self.frames)
        frames_per_host = split_frames_per_host(frames, self.distribute)
        print('Frames per host: {}'.format(frames_per_host))

        processes = []
        for host in self.distribute:
            _frames = frames_per_host[host]
            if _frames:
                if host == 'localhost':
                    p = blender(blend_file=os.path.expanduser(self.blend_file),
                                output=self.output,
                                scene=self.scene,
                                layer=self.layer,
                                frames=_frames)
                else:
                    p = remote_blender(host,
                                       blend_file=self.blend_file,
                                       output=self.output,
                                       frames=_frames)
                processes.append(p)
        try:
            for p in processes:
                p.wait()
        except KeyboardInterrupt:
            for p in processes:
                p.terminate()
        except:
            for p in processes:
                p.terminate()
            raise
        for host in self.distribute:
            if host != 'localhost':
                copy_results_from_host(host, self.output)
                cleanup_host(host, self.blend_file, self.output)
