import cv2
import numpy as np
from add import add

def diff(im1, im2, th):
    df = cv2.absdiff(im1, im2)
    mask = cv2.cvtColor(df, cv2.COLOR_BGR2GRAY)
    imask = mask > th
    canvas = np.zeros_like(im1, np.uint8)
    canvas[imask] = im1[imask]
    return canvas

def diff_score(im1, im2):
    '''
    Calculates a score to quantify the difference of two images.
    A lower score meens the images are similar.
    '''
    df = cv2.absdiff(im1, im2)
    df_gray = cv2.cvtColor(df, cv2.COLOR_BGR2GRAY)
    diff_pixels_norm = cv2.normalize(df_gray, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    score = sum(sum(diff_pixels_norm))
    return score

def diff_optimize(im1, im2):
    thresh = 0
    best_score = (thresh, np.inf)
    while thresh < 25:
        foreground = diff(im1, im2, thresh)
        re_added = add(foreground, im2)
        score = diff_score(im1, re_added)
        print(score)
        if score < best_score[1]:
            best_score = (thresh, score)
        thresh += 1
    return best_score
