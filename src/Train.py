"""Training script
"""

import os
import csv
import sys
import numpy as np
import mxnet as mx
import logging

from argparse import ArgumentParser
from collections import namedtuple


logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def get_args():
    p = ArgumentParser(description="train model and predict on validation data")
    p.add_argument("--local", help="train and predict on local sets",
                   action='store_true')
    return p.parse_args()


def get_paths(args):
    # TODO load config from seperate module
    Paths = namedtuple('Paths',
                       ['TRAIN_DATA_IN', 'TRAIN_LABEL_IN',
                        'TRAIN_SYSTOLE_OUT', 'TRAIN_DIASTOLE_OUT',
                        'VALID_DATA_IN', 'VALID_LABEL_OUT',
                        'SAMPLE_SUBMIT_IN', 'SUBMIT_OUT',
                        'MODEL_OUT', 'TEST_DATA_PATH'])
    TRAIN_DATA_IN = '../output/train-64x64-data.csv'
    TRAIN_LABEL_IN = '../output/train-label.csv'
    TRAIN_SYSTOLE_OUT = '../output/train-stytole.csv'
    TRAIN_DIASTOLE_OUT = '../output/train-diastole.csv'
    VALID_DATA_IN = '../output/validate-64x64-data.csv'
    VALID_LABEL_OUT = '../output/validate-label.csv'
    SAMPLE_SUBMIT_IN = '../data/sample_submission_validate.csv'
    SUBMIT_OUT = '../output/submission.csv'
    MODEL_OUT = None       # TODO save model output
    TEST_DATA_PATH = None  # TODO load test_data path (round 2)
    if args.local:
        TRAIN_DATA_IN = '../output/local_train-64x64-data.csv'
        TRAIN_LABEL_IN = '../output/local_train-label.csv'
        VALID_DATA_IN = '../output/local_test-64x64-data.csv'
        VALID_LABEL_OUT = '../output/local_test-label.csv'
    p = Paths(TRAIN_DATA_IN, TRAIN_LABEL_IN, TRAIN_SYSTOLE_OUT,
              TRAIN_DIASTOLE_OUT, VALID_DATA_IN, VALID_LABEL_OUT,
              SAMPLE_SUBMIT_IN, SUBMIT_OUT, MODEL_OUT, TEST_DATA_PATH)
    return p


def get_alexnet():
    """ A lenet style net, takes difference of each frame as input.
    """
    source = mx.sym.Variable("data")
    source = (source - 128) * (1.0/128)
    frames = mx.sym.SliceChannel(source, num_outputs=30)
    diffs = [frames[i+1] - frames[i] for i in range(29)]
    source = mx.sym.Concat(*diffs)
    net = mx.sym.Convolution(source, kernel=(4, 4), num_filter=32)
    net = mx.sym.BatchNorm(net, fix_gamma=True)
    net = mx.sym.Activation(net, act_type="relu")
    net = mx.sym.Pooling(net, pool_type="max", kernel=(2,2), stride=(2,2))
    net = mx.sym.Convolution(net, kernel=(3, 3), num_filter=64)
    net = mx.sym.BatchNorm(net, fix_gamma=True)
    net = mx.sym.Activation(net, act_type="relu")
    net = mx.sym.Pooling(net, pool_type="max", kernel=(2,2), stride=(2,2))
    net = mx.sym.Convolution(net, kernel=(3, 3), num_filter=64)
    net = mx.sym.BatchNorm(net, fix_gamma=True)
    net = mx.sym.Activation(net, act_type="relu")
    # net = mx.sym.Pooling(net, pool_type="max", kernel=(2,2), stride=(2,2))
    net = mx.sym.Convolution(net, kernel=(3, 3), num_filter=128)
    net = mx.sym.BatchNorm(net, fix_gamma=True)
    net = mx.sym.Activation(net, act_type="relu")
    net = mx.sym.Pooling(net, pool_type="max", kernel=(2,2), stride=(2,2))
    # first fullc
    flatten = mx.symbol.Flatten(net)
    flatten = mx.symbol.Dropout(flatten)
    fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=2048)
    fc1 = mx.symbol.Dropout(fc1)
    fc1 = mx.sym.Activation(fc1, act_type="relu")
    fc2 = mx.symbol.FullyConnected(data=fc1, num_hidden=2048)
    fc2 = mx.symbol.Dropout(fc2)
    fc2 = mx.sym.Activation(fc2, act_type="relu")
    fc3 = mx.symbol.FullyConnected(data=fc2, num_hidden=600)
    # Name the final layer as softmax so it auto matches the naming of data iterator
    # Otherwise we can also change the provide_data in the data iter
    return mx.symbol.LogisticRegressionOutput(data=fc3, name='softmax')


def get_vgg():
    source = mx.sym.Variable("data")
    source = (source - 128) * (1.0/128)
    frames = mx.sym.SliceChannel(source, num_outputs=30)
    diffs = [frames[i+1] - frames[i] for i in range(29)]
    source = mx.sym.Concat(*diffs)
    net = mx.sym.Convolution(source, kernel=(3, 3), num_filter=64)
    net = mx.sym.BatchNorm(net, fix_gamma=True)
    net = mx.sym.Activation(net, act_type="relu")
    net = mx.sym.Convolution(net, kernel=(3, 3), num_filter=64)
    net = mx.sym.BatchNorm(net, fix_gamma=True)
    net = mx.sym.Activation(net, act_type="relu")
    net = mx.sym.Pooling(net, pool_type="max", kernel=(2,2), stride=(2,2))
    net = mx.sym.Convolution(net, kernel=(3, 3), num_filter=128)
    net = mx.sym.BatchNorm(net, fix_gamma=True)
    net = mx.sym.Activation(net, act_type="relu")
    net = mx.sym.Convolution(net, kernel=(3, 3), num_filter=128)
    net = mx.sym.BatchNorm(net, fix_gamma=True)
    net = mx.sym.Activation(net, act_type="relu")
    net = mx.sym.Pooling(net, pool_type="max", kernel=(2,2), stride=(2,2))
    net = mx.sym.Convolution(net, kernel=(3, 3), num_filter=256)
    net = mx.sym.BatchNorm(net, fix_gamma=True)
    net = mx.sym.Activation(net, act_type="relu")
    net = mx.sym.Convolution(net, kernel=(3, 3), num_filter=256)
    net = mx.sym.BatchNorm(net, fix_gamma=True)
    net = mx.sym.Activation(net, act_type="relu")
    net = mx.sym.Convolution(net, kernel=(3, 3), num_filter=256)
    net = mx.sym.BatchNorm(net, fix_gamma=True)
    net = mx.sym.Activation(net, act_type="relu")
    net = mx.sym.Pooling(net, pool_type="max", kernel=(2,2), stride=(2,2))
    net = mx.sym.Convolution(net, kernel=(3, 3), num_filter=512)
    net = mx.sym.BatchNorm(net, fix_gamma=True)
    net = mx.sym.Activation(net, act_type="relu")
    net = mx.sym.Convolution(net, kernel=(3, 3), num_filter=512)
    net = mx.sym.BatchNorm(net, fix_gamma=True)
    net = mx.sym.Activation(net, act_type="relu")
    net = mx.sym.Convolution(net, kernel=(3, 3), num_filter=512)
    net = mx.sym.BatchNorm(net, fix_gamma=True)
    net = mx.sym.Activation(net, act_type="relu")
    net = mx.sym.Pooling(net, pool_type="max", kernel=(2,2), stride=(2,2))
    # net = mx.sym.Convolution(net, kernel=(3, 3), num_filter=512)
    # net = mx.sym.Activation(net, act_type="relu")
    # net = mx.sym.Convolution(net, kernel=(3, 3), num_filter=512)
    # net = mx.sym.Activation(net, act_type="relu")
    # net = mx.sym.Convolution(net, kernel=(3, 3), num_filter=512)
    # net = mx.sym.Activation(net, act_type="relu")
    # net = mx.sym.Pooling(net, pool_type="max", kernel=(2,2), stride=(2,2))
    # first fullc
    flatten = mx.symbol.Flatten(net)
    flatten = mx.symbol.Dropout(flatten)
    fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=4096)
    fc1 = mx.symbol.Dropout(fc1)
    fc1 = mx.sym.Activation(fc1, act_type="relu")
    fc2 = mx.symbol.FullyConnected(data=fc1, num_hidden=4096)
    fc2 = mx.symbol.Dropout(fc2)
    fc2 = mx.sym.Activation(fc2, act_type="relu")
    fc3 = mx.symbol.FullyConnected(data=fc2, num_hidden=600)
    # Name the final layer as softmax so it auto matches the naming of data iterator
    # Otherwise we can also change the provide_data in the data iter
    return mx.symbol.LogisticRegressionOutput(data=fc3, name='softmax')


def CRPS(label, pred):
    """ Custom evaluation metric on CRPS.
    """
    for i in range(pred.shape[0]):
        for j in range(pred.shape[1] - 1):
            if pred[i, j] > pred[i, j + 1]:
                pred[i, j + 1] = pred[i, j]
    return np.sum(np.square(label - pred)) / label.size


def encode_label(label_data):
    """Run encoding to encode the label into the CDF target.
    """
    stytole = label_data[:, 1]
    diastole = label_data[:, 2]
    stytole_encode = np.array([
            (x < np.arange(600)) for x in stytole
        ], dtype=np.uint8)
    diastole_encode = np.array([
            (x < np.arange(600)) for x in diastole
        ], dtype=np.uint8)
    return stytole_encode, diastole_encode


def encode_csv(label_csv, stytole_csv, diastole_csv):
    stytole_encode, diastole_encode = encode_label(np.loadtxt(label_csv, delimiter=","))
    np.savetxt(stytole_csv, stytole_encode, delimiter=",", fmt="%g")
    np.savetxt(diastole_csv, diastole_encode, delimiter=",", fmt="%g")


args = get_args()
paths = get_paths(args)


# Write encoded label into the target csv
# We use CSV so that not all data need to sit into memory
# You can also use inmemory numpy array if your machine is large enough
encode_csv(paths.TRAIN_LABEL_IN,
           paths.TRAIN_SYSTOLE_OUT, paths.TRAIN_DIASTOLE_OUT)



class KSH_Init(mx.initializer.Normal):
    def __init__(self, sigma=0.01):
        super(KSH_Init, self).__init__(sigma)

    def _init_bias(self, _, arr):
        arr[:] = 1.0


lr = mx.lr_scheduler.FactorScheduler(step=800, factor=0.9)

# # Training the stytole net

network = get_alexnet()
# network = get_vgg()
batch_size = 128
devs = [mx.gpu(0)]
data_train = mx.io.CSVIter(data_csv=paths.TRAIN_DATA_IN,
                           data_shape=(30, 128, 128),
                           label_csv=paths.TRAIN_SYSTOLE_OUT,
                           label_shape=(600,),
                           batch_size=batch_size,)

data_validate = mx.io.CSVIter(data_csv=paths.VALID_DATA_IN,
                              data_shape=(30, 128, 128),
                              batch_size=1)

stytole_model = mx.model.FeedForward(ctx=devs,
        symbol             = network,
        num_epoch          = 100,
        learning_rate      = 0.2,  # 0.2 for alexnet
        wd                 = 0.0005,
        momentum           = 0.9,
        initializer=KSH_Init(),
        lr_scheduler=lr)

stytole_model.fit(X=data_train, eval_metric=mx.metric.np(CRPS))


# # Predict stytole


stytole_prob = stytole_model.predict(data_validate)


# # Training the diastole net

network = get_alexnet()
batch_size = 128
devs = [mx.gpu(0)]
data_train = mx.io.CSVIter(data_csv=paths.TRAIN_DATA_IN,
                           data_shape=(30, 128, 128),
                           label_csv=paths.TRAIN_DIASTOLE_OUT,
                           label_shape=(600,),
                           batch_size=batch_size)

diastole_model = mx.model.FeedForward(ctx=devs,
        symbol             = network,
        num_epoch          = 100,
        learning_rate      = 0.2,  # 0.2 for alexnet
        wd                 = 0.0005,
        momentum           = 0.9,
        initializer=KSH_Init(),
        lr_scheduler=lr)

diastole_model.fit(X=data_train, eval_metric=mx.metric.np(CRPS))


# # Predict diastole

diastole_prob = diastole_model.predict(data_validate)


# # Generate Submission

def accumulate_result(validate_lst, prob):
    sum_result = {}
    cnt_result = {}
    size = prob.shape[0]
    fi = csv.reader(open(validate_lst))
    for i in range(size):
        line = fi.__next__()  # Python2: line = fi.next()
        idx = int(line[0])
        if idx not in cnt_result:
            cnt_result[idx] = 0.
            sum_result[idx] = np.zeros((1, prob.shape[1]))
        cnt_result[idx] += 1
        sum_result[idx] += prob[i, :]
    for i in cnt_result.keys():
        sum_result[i][:] /= cnt_result[i]
    return sum_result


# In[9]:

stytole_result = accumulate_result(paths.VALID_LABEL_OUT, stytole_prob)
diastole_result = accumulate_result(paths.VALID_LABEL_OUT, diastole_prob)


# In[10]:

# we have 2 person missing due to frame selection, use udibr's hist result instead
def doHist(data):
    h = np.zeros(600)
    for j in np.ceil(data).astype(int):
        h[j:] += 1
    h /= len(data)
    return h
train_csv = np.genfromtxt(paths.TRAIN_LABEL_IN, delimiter=',')
hSystole = doHist(train_csv[:, 1])
hDiastole = doHist(train_csv[:, 2])


# In[11]:

def submission_helper(pred):
    p = np.zeros(600)
    pred.resize(p.shape)
    p[0] = pred[0]
    for j in range(1, 600):
        a = p[j - 1]
        b = pred[j]
        if b < a:
            p[j] = a
        else:
            p[j] = b
    return p



def calc_local_CRPS(submit_path, label_path):
    # https://www.kaggle.com/c/second-annual-data-science-bowl/details/evaluation
    label_csv = np.genfromtxt(label_path, delimiter=",")
    label_csv = np.repeat(label_csv, 2, axis=0)
    fi = csv.reader(open(submit_path))
    fi.__next__()
    score = 0
    line_num = 0
    for line in fi:
        idx = line[0]
        _, target = idx.split('_')
        lbl = None
        if target == 'Diastole':
            lbl = label_csv[line_num, 2]
        else:
            lbl = label_csv[line_num, 1]
        p = np.array(line[1:], dtype=np.float64)
        h = np.array(lbl < np.arange(600), dtype=np.uint8)
        score += np.sum(np.square(p - h))
        line_num += 1
    score /= 600 * label_csv.shape[0]
    return score


# In[12]:

if not args.local:
    fi = csv.reader(open(paths.SAMPLE_SUBMIT_IN))
    f = open(paths.SUBMIT_OUT, "w")
    fo = csv.writer(f, lineterminator='\n')
    fo.writerow(fi.__next__())
    for line in fi:
        idx = line[0]
        key, target = idx.split('_')
        key = int(key)
        out = [idx]
        if key in stytole_result:
            if target == 'Diastole':
                out.extend(list(submission_helper(diastole_result[key])))
            else:
                out.extend(list(submission_helper(stytole_result[key])))
        else:
            print("Miss: %s" % idx)
            if target == 'Diastole':
                out.extend(hDiastole)
            else:
                out.extend(hSystole)
        fo.writerow(out)
    f.close()
else:
    fi = csv.reader(open(paths.SAMPLE_SUBMIT_IN))
    f = open(paths.SUBMIT_OUT, "w")
    fo = csv.writer(f, lineterminator='\n')
    fo.writerow(fi.__next__())

    # Grab local-test ids
    test_csv = np.genfromtxt(paths.VALID_LABEL_OUT, delimiter=',')
    ids = test_csv[:, 0].astype(int)
    # ids = np.repeat(ids, 2)
    for id in ids:
        s_out = [str(id) + "_Systole"]
        d_out = [str(id) + "_Diastole"]
        s_out.extend(list(submission_helper(stytole_result[id])))
        d_out.extend(list(submission_helper(diastole_result[id])))
        fo.writerow(d_out)
        fo.writerow(s_out)
    f.close()
    score = calc_local_CRPS(paths.SUBMIT_OUT, paths.VALID_LABEL_OUT)
    logger.info('Local CRPS: {}'.format(score))

