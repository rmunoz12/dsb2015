"""
Preprocessing of training and validation data sets.
"""

import os
import csv
import random

import scipy
import numpy as np
import dicom

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
    """
    Store frame pixel data and labels as csv files. Each frame and label are
    written to their respective csv files in the same order. In addition, each
    processed image from a frame is saved for inspection.

    Parameters
    ----------
    data_fname : str
        Output path for the frame's image data

    label_fname : str
        Output path for the frame's labels

    frames : list[Frame]

    label_map : dict[int, str]
        Labels for `frames`. If `None`, then the output will include the frame
        index and zeros as the class labels.

    Returns
    -------
    result : list[str]
        Paths to processed images.
    """
    label_fo = open(label_fname, 'w')
    data_fo = open(data_fname, 'w')
    dwriter = csv.writer(data_fo)
    counter, result = 0, []
    for frame in frames:
        data = []
        index = int(frame.data[0].split('/')[-4])
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


def split_csv(src_csv, split_to_train, train_csv, test_csv):
    """
    Splits `src_csv` into `train_csv` and `test_csv`.

    Parameters
    ----------
    src_csv : str
        Path to source csv file.

    split_to_train : list[bool]
        List with the same length as the number of rows in `src_csv`, indicating
        whether the row should be included in `train_csv`

    train_csv : str
        Path to output training csv file.

    test_csv : str
        Path to output test csv file.
    """
    ftrain = open(train_csv, "w")
    ftest = open(test_csv, "w")
    cnt = 0
    for l in open(src_csv):
        if split_to_train[cnt]:
            ftrain.write(l)
        else:
            ftest.write(l)
        cnt += 1
    ftrain.close()
    ftest.close()


def make_local_split(test_frac, output_path):
    """
    Generate local train/test split, which can be used evaluate models locally,
    blind to the result of submission to Kaggle's validation set.

    Parameters
    ----------
    test_frac : float
        The fraction ([0, 1]) of data that should be used for the local test
        set.

    output_path : str
        Path to output folder, ending with a slash `/`.
    """
    train_index = \
        np.loadtxt(output_path + "train-label.csv", delimiter=",")[:, 0].\
            astype("int")
    train_index = set(train_index)
    num_test = int(len(train_index) * test_frac)
    random.shuffle(list(train_index))
    train_index = set(train_index[num_test:])
    split_to_train = [x in train_index for x in train_index]
    split_csv(output_path + "train-label.csv",
              split_to_train,
              output_path + "local_train-label.csv",
              output_path + "local_test-label.csv")
    split_csv(output_path + "train-64x64-data.csv",
              split_to_train,
              output_path + "local_train-64x64-data.csv",
              output_path + "local_test-64x64-data.csv")


def preprocess(cfg):
    """
    Main entry for preprocessing operations.

    Parameters
    ----------
    cfg : Config
    """
    random.seed(100)
    train_paths = gen_frame_paths(cfg.train_path + 'train')
    vld_paths = gen_frame_paths(cfg.train_path + 'validate')
    os.makedirs(cfg.output_path, exist_ok=True)
    train_frames = gen_augmented_frames(train_paths, 128)
    train_frames = sorted(train_frames)  # for reproducibility
    random.shuffle(train_frames)
    write_data_and_label_csvs(cfg.output_path + 'train-64x64-data.csv',
                              cfg.output_path + 'train-label.csv',
                              train_frames,
                              get_label_map(cfg.train_path + 'train.csv'))
    vld_frames = gen_augmented_frames(vld_paths, 128, normal_only=True)
    write_data_and_label_csvs(cfg.output_path + "validate-64x64-data.csv",
                              cfg.output_path + "validate-label.csv",
                              vld_frames,
                              None)
    make_local_split(0.1, cfg.output_path)
