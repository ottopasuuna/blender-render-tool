from .context import src

from src.blender import *


def test_build_blender_cmd():
    cmd = build_blender('test.blend', 'outdir', frames=range(1, 3))
    assert cmd == 'blender -b test.blend -o outdir/####.png -f 1,2'

    cmd = build_blender('test.blend', 'outdir', frames=range(1, 6, 2))
    assert cmd == 'blender -b test.blend -o outdir/####.png -f 1,3,5'

def test_split_frames_per_host():
    fph = split_frames_per_host(range(1, 5), ['localhost', 'server'])
    assert fph == {'localhost': [1, 2], 'server': [3, 4]}
