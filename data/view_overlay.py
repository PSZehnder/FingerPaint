import cv2

im = cv2.imread('frame_0000.jpg')

def overlay_rect(im, cx, cy, xw, yw):
    print(im.shape)
    top_left = (int((cx - xw / 2) * im.shape[1]), int((cy - yw / 2)*im.shape[0]))
    bottom_right = (int((cx + xw / 2)*im.shape[1]), int((cy + yw / 2)*im.shape[0]))
    print(top_left)
    print(bottom_right)
    return cv2.rectangle(im, top_left, bottom_right, color=(256, 0, 0), thickness=1)

cv2.imshow('new_window', overlay_rect(im, 0.525132893041237, 0.6454825315005728, 0.11030927835051534, 0.15945017182130575 ))
cv2.waitKey()