"""Train systole and diastole models and generate validation submission file"""

import csv
import logging

import numpy as np
import mxnet as mx

from .models import get_alexnet, get_vgg

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def CRPS(label, pred):
    """ Custom evaluation metric on CRPS.
    """
    for i in range(pred.shape[0]):
        for j in range(pred.shape[1] - 1):
            if pred[i, j] > pred[i, j + 1]:
                pred[i, j + 1] = pred[i, j]
    return np.sum(np.square(label - pred)) / label.size


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


class KSH_Init(mx.initializer.Normal):
    def __init__(self, sigma=0.01):
        super(KSH_Init, self).__init__(sigma)

    def _init_bias(self, _, arr):
        arr[:] = 1.0


def train_model(data_csv, label_csv, batch_size=128, network=get_alexnet()):
    devs = [mx.gpu(0)]
    lr = mx.lr_scheduler.FactorScheduler(step=800, factor=0.9)

    data_train = mx.io.CSVIter(data_csv=data_csv,
                               data_shape=(30, 128, 128),
                               label_csv=label_csv,
                               label_shape=(600,),
                               batch_size=batch_size)
    model = mx.model.FeedForward(ctx=devs,
                                 symbol=network,
                                 num_epoch=50,
                                 learning_rate=0.2,  # 0.2 for alexnet
                                 wd=0.0005,
                                 momentum=0.9,
                                 initializer=KSH_Init(),
                                 lr_scheduler=lr)
    model.fit(X=data_train, eval_metric=mx.metric.np(CRPS))
    return model


def accumulate_result(validate_lst, prob):
    sum_result = {}
    cnt_result = {}
    size = prob.shape[0]
    fi = csv.reader(open(validate_lst))
    for i in range(size):
        line = fi.__next__()
        idx = int(line[0])
        if idx not in cnt_result:
            cnt_result[idx] = 0.
            sum_result[idx] = np.zeros((1, prob.shape[1]))
        cnt_result[idx] += 1
        sum_result[idx] += prob[i, :]
    for i in cnt_result.keys():
        sum_result[i][:] /= cnt_result[i]
    return sum_result


def doHist(data):
    h = np.zeros(600)
    for j in np.ceil(data).astype(int):
        h[j:] += 1
    h /= len(data)
    return h


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


def write_validation_results(systole_result, diastole_result, cfg):
    # we have 2 person missing due to frame selection, use udibr's hist result instead
    train_csv = np.genfromtxt(cfg.train_label_csv, delimiter=',')
    hSystole = doHist(train_csv[:, 1])
    hDiastole = doHist(train_csv[:, 2])

    fi = csv.reader(open(cfg.sample_submit))
    f = open(cfg.submit_out, "w")
    fo = csv.writer(f, lineterminator='\n')
    fo.writerow(fi.__next__())
    if not cfg.local:
        for line in fi:
            idx = line[0]
            key, target = idx.split('_')
            key = int(key)
            out = [idx]
            if key in systole_result:
                if target == 'Diastole':
                    out.extend(list(submission_helper(diastole_result[key])))
                else:
                    out.extend(list(submission_helper(systole_result[key])))
            else:
                print("Miss: %s" % idx)
                if target == 'Diastole':
                    out.extend(hDiastole)
                else:
                    out.extend(hSystole)
            fo.writerow(out)
        f.close()
    else:
        # Grab local-test ids
        test_csv = np.genfromtxt(cfg.valid_label_out, delimiter=',')
        ids = test_csv[:, 0].astype(int)
        for id in ids:
            s_out = [str(id) + "_Systole"]
            d_out = [str(id) + "_Diastole"]
            s_out.extend(list(submission_helper(systole_result[id])))
            d_out.extend(list(submission_helper(diastole_result[id])))
            fo.writerow(d_out)
            fo.writerow(s_out)
        f.close()
        score = calc_local_CRPS(cfg.submit_out, cfg.valid_label_out)
        logger.info('Local CRPS: {}'.format(score))


def train(cfg):
    # Write encoded label into the target csv
    # We use CSV so that not all data need to sit into memory
    # You can also use inmemory numpy array if your machine is large enough
    encode_csv(cfg.train_label_csv,
               cfg.train_systole_out, cfg.train_diastole_out)

    systole_model = train_model(cfg.train_data_csv, cfg.train_systole_out)
    diastole_model = train_model(cfg.train_data_csv, cfg.train_diastole_out)

    data_validate = mx.io.CSVIter(data_csv=cfg.valid_data_csv,
                                  data_shape=(30, 128, 128),
                                  batch_size=1)
    systole_prob = systole_model.predict(data_validate)
    diastole_prob = diastole_model.predict(data_validate)
    systole_result = accumulate_result(cfg.valid_label_out, systole_prob)
    diastole_result = accumulate_result(cfg.valid_label_out, diastole_prob)

    write_validation_results(systole_result, diastole_result, cfg)
