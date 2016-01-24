"""Training & Validation Data Exploration"""

import os

TRN_PATH = '../data/train'
VLD_PATH = '../data/train'
OUT_PATH = '../output'


# TODO handle sax files with n != 30 dicom images
# TODO explore 2ch and 4ch folders
# TODO order images by slice depth


def mkdir(path):
    os.makedirs(path, exist_ok=True)
    # if not os.path.isdir(path):


def summarize_files(path):
    res = []
    for root, _, files in os.walk(path):
        if len(files) == 0 or root.find("sax") == -1:
            continue
        files = [x for x in files if os.path.splitext(x)[1] == '.dcm']
        if len(files) == 0:
            continue
        prefix = files[0].rsplit('-', 1)[0]
        fileset = set(files)
        expected = ["%s-%04d.dcm" % (prefix, i + 1) for i in range(30)]
        if not all(x in fileset for x in expected):
            n = len(fileset)
            # print("n: {}\t {}".format(n, root))


if __name__ == '__main__':
    summarize_files(TRN_PATH)
    summarize_files(VLD_PATH)
