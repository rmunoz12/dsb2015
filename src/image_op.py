"""
Image processing operations and a function to generate `Frame` objects.
"""

from collections import namedtuple
import random

import numpy as np
from skimage import transform


class Frame(namedtuple('Frame', 'data func aug_name')):
    """
    Represents a frame (30 dicom images), along with the data augmentation
    function that will be applied to the frame.

    Parameters
    ----------
    data : list[str]
        List of paths to the images in the frame

    func :  np.array[float] -> np.array[float]
        Data augmentation function. The input and output arrays are 2D.

    aug_name : str
        augmentation function name
    """
    def __lt__(self, other):
        return self.data < other.data


def crop_resize(img, size):
    """
    Crop image `img` into a square with side length `size`. Cropping is
    performed from the center of the image. Furthermore, the image pixel values
    are converted from floats to integers from 0 to 255.

    Parameters
    ----------
    img : np.array
        2D array

    size : int

    Returns
    -------
    resized_img : np.array[np.dtype('uint8')]
        2D array
    """
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    resized_img = transform.resize(crop_img, (size, size))
    resized_img *= 255
    return resized_img.astype("uint8")


def rotate(img, degree):
    return transform.rotate(img, degree)


def flip(img):
    return np.fliplr(img)


def get_aug_funcs(size, normal_only=False):
    funcs = [(lambda x: crop_resize(x, size), 'n')]
    if not normal_only:
        deg = random.random() * 360
        funcs.extend([(lambda x: crop_resize(rotate(x, deg), size), 'r'),
                      (lambda x: crop_resize(flip(x), size), 'f'),
                      (lambda x: crop_resize(flip(rotate(x, deg)), size), 'rf')])
    return funcs


def gen_augmented_frames(paths, size, normal_only=False):
    """
    Generates a `Frame` that includes the paths to its images as well as the
    data augmentation function that should applied to the frame.

    Parameters
    ----------
    paths : list[str]
        Paths tot he dicom images of the frame

    size : int
        Size of the square to which the images will be cropped.

    normal_only : bool
        If true, will only return the 'normal' augmentation function, which
        only crops the input image to the size needed for the training network.

    Yields
    ------
    Frame
    """
    for lst in paths:
        funcs = get_aug_funcs(size, normal_only)
        for f, name in funcs:
            yield Frame(data=lst, func=f, aug_name=name)
