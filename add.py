import cv2

def add(im1, im2):
    im1gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(im1gray, 0, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    im2_bg = cv2.bitwise_and(im2, im2, mask=mask_inv)
    im1_fg = cv2.bitwise_and(im1, im1, mask=mask)
    res = cv2.add(im2_bg, im1_fg)
    return res
