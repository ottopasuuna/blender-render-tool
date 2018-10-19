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

def transparentOverlay(overlay , background):
    background = background[:,:,:3]
    out = np.zeros(background.shape, background.dtype)
    a = overlay[:,:,3]
    overlay = overlay[:,:,:3]
    alpha = a/255
    alpha_inv = 1-alpha
    for ch in range(3):
        out[:,:,ch] = alpha*overlay[:,:,ch] + alpha_inv*background[:,:,ch]
    return out

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

def flow(frame1, frame2):
    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    _flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 2, 10, 5, 7, 1.2, 0)
    return _flow

def interpolate_flow(frame1, frame3):
    # frame2 = cv2.addWeighted(frame1, 0.5, frame3, 0.5, 0)
    # Thanks to github user koteth for the interpolation algorithm
    _flow = flow(frame1, frame3)
    h, w = _flow.shape[:2]
    _flow = -_flow #?
    _flow[:, :, 0] /= 2.0
    _flow[:, :, 1] /= 2.0
    _flow[:, :, 0] += np.arange(w)
    _flow[:, :, 1] += np.arange(h)[:, np.newaxis]
    frame2 = cv2.remap(frame1, _flow, None, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return frame2

def scale(img, width, height, mode):
    if mode == 'cubic':
        interp = cv2.INTER_CUBIC
    elif mode == 'lanczos':
        interp = cv2.INTER_LANCZOS4
    scaled_img = cv2.resize(img, (width, height), interpolation=interp)
    return scaled_img

def denoise(img, strength, mode='fastNL'):
    if mode == 'fastNL':
        strength_color = strength
        templateWindowSize = 7
        searchWindowSize = 35
        res = cv2.fastNlMeansDenoisingColored(img, None, strength, strength_color,
                                              templateWindowSize, searchWindowSize)
    elif mode == 'median':
        res = cv2.medianBlur(img, 5)
    return res

def add_noise(img):
    mean = 0
    sigma = 20
    m = (mean,mean,mean)
    s = (sigma,sigma,sigma)
    im = np.zeros(img.shape, np.uint8)
    im = cv2.randn(im, m, s)
    noisy = img + im
    return noisy
