from argparse import ArgumentParser
import json

from dsb2015.config import Config
from dsb2015.preprocess import preprocess
from dsb2015.train import train


def get_args():
    p = ArgumentParser(description="train model and predict on validation data")

    group = p.add_mutually_exclusive_group()
    group.add_argument("-p", help="preprocess only",
                       action='store_true')
    group.add_argument("-t", help="training only",
                       action="store_true")

    p.add_argument("--local", help="train and predict on local sets",
                   action='store_true')

    return p.parse_args()


if __name__ == '__main__':
    args = get_args()
    with open('settings.json') as f:
        params = json.load(f)
        params['local'] = args.local
        cfg = Config(**params)
    if not args.t:
        print("--- begin preprocessing ---")
        preprocess(cfg)
    if not args.p:
        print("--- begin training ---")
        train(cfg)
