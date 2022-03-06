import cv2
import numpy as np
from IPython.display import display
from PIL import Image


def imshow(a):
    a = a.clip(0, 255).astype('uint8')
    if a.ndim == 3:
        if a.shape[2] == 4:
            a = cv2.cvtColor(a, cv2.COLOR_BGRA2RGBA)
        else:
            a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
    display(Image.fromarray(a))


def rotate_polygon(pts, rotation_matrix):
    """Transforms a list of points, `pts`,
    using the affine transform `A`."""
    src = np.zeros((len(pts), 1, 2))
    src[:, 0] = pts
    dst = np.squeeze(cv2.perspectiveTransform(src, rotation_matrix))
    return numpy_to_list(dst)


def numpy_to_list(array):
    lst2 = []
    for i in array:
        lst = []
        for k in i:
            lst.append(float(k))
        lst2.append(lst)
    return lst2