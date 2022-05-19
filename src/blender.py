from subprocess import Popen
import json
import os
import shlex

from .core import (slice_list, shell, get_file_mod_date)


def build_frames(frames):
    args = {}
    if frames is not None:
        args['frames'] = '-f ' + ','.join([str(f) for f in list(frames)])
    else:
        args['frames'] = ''
    return args

def build_output(path):
    file_extension = 'png'
    file_name_template = '####.{}'.format(file_extension)
    return {'output': '-o {}'.format(os.path.join(path, file_name_template))}


def build_blender(blend_file, output, frames=None):
    args = {'blend_file': blend_file}
    args.update(build_frames(frames))
    args.update(build_output(output))
    cmd = 'blender -b {blend_file} {output} {frames}'.format(**args)
    return cmd

def build_blender_script(blend_file, output, scene=None, layer="", frames=None):
    args = {
        'output': output,
        'frames': frames,
        'scene' : scene,
        'layer': layer,
    }
    json_args = json.dumps(args)
    blender_proc_script_path = os.path.join(os.path.dirname(__file__), "rentool_blender_proc.py")
    cmd = f'blender -b {blend_file} --python {blender_proc_script_path} -- \'{json_args}\''
    return cmd

def blender(*args, **kwargs):
    cmd = build_blender_script(*args, **kwargs)
    return Popen(shlex.split(cmd))

def copy_to_host(blend_file, host):
    shell('scp -p {} {}:'.format(blend_file, host))

def copy_results_from_host(host, output):
    shell('scp -r "{}:{}/*" ./{}'.format(host, output, output))

def cleanup_host(host, blend_file, output):
    # shell('ssh {} rm -r {} {}'.format(host, blend_file, output))
    shell('ssh {} rm -r {}'.format(host, output))


def remote_blender(host, *args, **kwargs):
    print('Host: {}'.format(host))
    print(kwargs)
    blend_file = kwargs['blend_file']
    if get_file_mod_date(blend_file, host=host) != get_file_mod_date(blend_file):
        copy_to_host(blend_file, host)
    blend_cmd = build_blender(*args, **kwargs)
    cmd = 'ssh {} "{}"'.format(host, blend_cmd)
    return Popen(shlex.split(cmd))


def split_frames_per_host(frames, hosts):
    frame_slices = slice_list(frames, len(hosts))
    frames_per_host = {host: frames for host, frames in zip(hosts, frame_slices)}
    return frames_per_host
