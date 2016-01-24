"""
Preprocessing of training and validation data sets.
"""

import os
from collections import namedtuple
import csv
import random

import scipy
import numpy as np
import dicom
from skimage import transform

from .image_op import gen_augmented_frames


# TODO handle sax files with n != 30 dicom images
# TODO explore 2ch and 4ch folders
# TODO order images by slice depth


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


random.seed(100)
train_paths = gen_frame_paths("../data/train")
vld_paths = gen_frame_paths("../data/validate")

os.makedirs("../output/", exist_ok=True)

train_frames = gen_augmented_frames(train_paths, 128)
train_frames = sorted(train_frames)  # for reproducibility
random.shuffle(train_frames)
write_data_and_label_csvs('../output/train-64x64-data.csv', '../output/train-label.csv', train_frames, get_label_map('../data/train.csv'))


vld_frames = gen_augmented_frames(vld_paths, 128, normal_only=True)
write_data_and_label_csvs("../output/validate-64x64-data.csv", "../output/validate-label.csv", vld_frames, None)


# Generate local train/test split, which you could use to tune your model locally.
train_index = np.loadtxt("../output/train-label.csv", delimiter=",")[:,0].astype("int")
train_set, test_set = local_split(train_index, 0.1)
split_to_train = [x in train_set for x in train_index]
split_csv("../output/train-label.csv", split_to_train, "../output/local_train-label.csv", "../output/local_test-label.csv")
split_csv("../output/train-64x64-data.csv", split_to_train, "../output/local_train-64x64-data.csv", "../output/local_test-64x64-data.csv")
