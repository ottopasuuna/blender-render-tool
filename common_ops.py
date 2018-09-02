import cv2
import numpy as np

def add(im1, im2):
    im1gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(im1gray, 0, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    im2_bg = cv2.bitwise_and(im2, im2, mask=mask_inv)
    im1_fg = cv2.bitwise_and(im1, im1, mask=mask)
    res = cv2.add(im2_bg, im1_fg)
    return res

def diff(im1, im2):
    df = cv2.absdiff(im1, im2)
    df_gray = cv2.cvtColor(df, cv2.COLOR_BGR2GRAY)
    return df_gray

def extract_foreground(full_image, background, threshold):
    mask = diff(full_image, background)
    imask = mask > threshold
    canvas = np.zeros_like(full_image, np.uint8)
    canvas[imask] = full_image[imask]
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
        foreground = extract_foreground(im1, im2, thresh)
        re_added = add(foreground, im2)
        score = diff_score(im1, re_added)
        print(score)
        if score < best_score[1]:
            best_score = (thresh, score)
        thresh += 1
    return best_score

def blend(im1, im2):
    res = cv2.addWeighted(im1, 0.5, im2, 0.5, 0)
    return res

def blend_all(imgs):
    final = blend(imgs[0], imgs[1])
    if len(imgs) > 2:
        for img in imgs[2:]:
            final = blend(final, img)
    return final

