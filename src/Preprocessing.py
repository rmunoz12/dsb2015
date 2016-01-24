"""Preprocessing script.

This script walks over the directories and dump the frames into a csv file
"""
import os
from collections import namedtuple
import csv
import random

import scipy
import numpy as np
import dicom
from skimage import transform


# TODO handle sax files with n != 30 dicom images
# TODO explore 2ch and 4ch folders
# TODO order images by slice depth


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


def gen_frame_paths(root_path):
    """Get path to all the frame in view SAX and contain complete frames"""
    for root, _, files in os.walk(root_path):
        if len(files) == 0 or not files[0].endswith(".dcm") or root.find("sax") == -1:
            continue
        prefix = files[0].rsplit('-', 1)[0]
        fileset = set(files)
        expected = ["%s-%04d.dcm" % (prefix, i + 1) for i in range(30)]
        if all(x in fileset for x in expected):
            yield [root + "/" + x for x in expected]


def get_label_map(file_path):
    """
    Build a dictionary for looking up training labels.

    Parameters
    ----------
    file_path : str

    Returns
    -------
    label_map : dict[str, str]
        The key is the class number (as an int).
    """
    label_map = {}
    with open(file_path) as fi:
        fi.readline()  # ignore header
        for line in fi:
            arr = line.split(',')
            label_map[int(arr[0])] = line
    return label_map


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


def write_data_and_label_csvs(data_fname, label_fname, frames, label_map):
    label_fo = open(label_fname, 'w')
    data_fo = open(data_fname, 'w')
    dwriter = csv.writer(data_fo)
    counter, result = 0, []
    for frame in frames:
        data = []
        index = int(frame.data[0].split('/')[3])
        if label_map:
            label_fo.write(label_map[index])
        else:
            label_fo.write("%d,0,0\n" % index)
        for path in frame.data:
            f = dicom.read_file(path)
            f = f.pixel_array.astype(float) / np.max(f.pixel_array)
            img = frame.func(f)
            dst_path = path.rsplit(".", 1)[0] + "." + frame.aug_name + ".jpg"
            scipy.misc.imsave(dst_path, img)
            result.append(dst_path)
            data.append(img)
        data = np.array(data, dtype=np.uint8)
        data = data.reshape(data.size)
        dwriter.writerow(data)
        counter += 1
        if counter % 100 == 0:
            print("%d slices processed" % counter)
    print("All finished, %d slices in total" % counter)
    label_fo.close()
    data_fo.close()
    return result


def local_split(train_index, test_frac):
    random.seed(0)
    train_index = set(train_index)
    all_index = sorted(train_index)
    num_test = int(len(all_index) * test_frac)
    random.shuffle(all_index)
    train_set = set(all_index[num_test:])
    test_set = set(all_index[:num_test])
    return train_set, test_set


def split_csv(src_csv, split_to_train, train_csv, test_csv):
    ftrain = open(train_csv, "w")
    ftest = open(test_csv, "w")
    cnt = 0
    for l in open(src_csv):
        if split_to_train[cnt]:
            ftrain.write(l)
        else:
            ftest.write(l)
        cnt = cnt + 1
    ftrain.close()
    ftest.close()


def get_aug_funcs(normal_only=False):
    funcs = [(lambda x: crop_resize(x, 128), 'n')]
    if not normal_only:
        deg = random.random() * 360
        funcs.extend([(lambda x: crop_resize(rotate(x, deg), 128), 'r'),
                      (lambda x: crop_resize(flip(x), 128), 'f'),
                      (lambda x: crop_resize(flip(rotate(x, deg)), 128), 'rf')])
    return funcs


def gen_augmented_frames(paths, normal_only=False):
    """
    Generates a `Frame` that includes the paths to its images as well as the
    data augmentation function that should applied to the frame.

    Parameters
    ----------
    paths : list[str]
        Paths tot he dicom images of the frame

    normal_only : bool
        If true, will only return the 'normal' augmentation function, which
        only crops the input image to the size needed for the training network.

    Yields
    ------
    Frame
    """
    for lst in paths:
        funcs = get_aug_funcs(normal_only)
        for f, name in funcs:
            yield Frame(data=lst, func=f, aug_name=name)


random.seed(100)
train_paths = gen_frame_paths("../data/train")
vld_paths = gen_frame_paths("../data/validate")

os.makedirs("../output/", exist_ok=True)

train_frames = gen_augmented_frames(train_paths)
train_frames = sorted(train_frames)  # for reproducibility
random.shuffle(train_frames)
write_data_and_label_csvs('../output/train-64x64-data.csv', '../output/train-label.csv', train_frames, get_label_map('../data/train.csv'))


vld_frames = gen_augmented_frames(vld_paths, normal_only=True)
write_data_and_label_csvs("../output/validate-64x64-data.csv", "../output/validate-label.csv", vld_frames, None)


# Generate local train/test split, which you could use to tune your model locally.
train_index = np.loadtxt("../output/train-label.csv", delimiter=",")[:,0].astype("int")
train_set, test_set = local_split(train_index, 0.1)
split_to_train = [x in train_set for x in train_index]
split_csv("../output/train-label.csv", split_to_train, "../output/local_train-label.csv", "../output/local_test-label.csv")
split_csv("../output/train-64x64-data.csv", split_to_train, "../output/local_train-64x64-data.csv", "../output/local_test-64x64-data.csv")
