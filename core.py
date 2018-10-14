import os
from multiprocessing import Pool, cpu_count

import cv2

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
            paths = [os.path.join(path, p) for p in paths]
        elif os.path.isfile(path):
            paths = [path]
        else:
            raise RuntimeError('Specified path(s) does not exist')
    elif len(path) == 1:
        path = path[0]
        if os.path.isdir(path):
            paths = os.listdir(path)
            paths = [os.path.join(path, p) for p in paths]
        elif os.path.isfile(path):
            paths = [path]
        else:
            raise RuntimeError('Specified path(s) does not exist')
    else:
        paths = path
    return paths


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
    if not os.path.isfile(path):
        raise RuntimeError('Specified path(s) does not exist')
    return cv2.imread(path, cv2.IMREAD_UNCHANGED)


def load_images(paths):
    paths = get_paths(paths)
    params = [(path,) for path in paths]
    imgs = parallelize(load_image, params)
    return imgs


def save_image(img, path):
    cv2.imwrite(path, img)

def save_images(images, path):
    name_format = '{0:04d}.png'
    make_dir(path)
    for i, image in enumerate(images):
        p = os.path.join(path, name_format.format(i+1))
        save_image(image, p)


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
            out_paths = [os.path.join(output_path, basename) for basename in file_basenames]
        else:
            out_paths = [None]*len(images)
    else:
        out_paths = [output_path]
    for (img, path) in zip(images, out_paths):
        save_or_show(img, path)
