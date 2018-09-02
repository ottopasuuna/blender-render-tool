import cv2
import numpy as np

def flow(frame1, frame2):
    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    _flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 2, 10, 5, 7, 1.2, 0)
    return _flow

def interpolate(frame1, frame3):
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

def interpolate_all(start, end):
    assert(start < end)
    curr = start
    while curr < end:
        frame1 = cv2.imread('{0:04d}.png'.format(curr))
        frame3 = cv2.imread('{0:04d}.png'.format(curr+2))
        frame2 = interpolate(frame1, frame3)
        cv2.imwrite('{0:04d}.png'.format(curr+1), frame2)
        curr += 2

def show_img(img):
    while True:
        cv2.imshow('img', img)
        k = cv2.waitKey(1)
        if k == ord('q'):
            break


interpolate_all(1, 47)
