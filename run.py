from argparse import ArgumentParser
import json

from dsb2015.Preprocessing import preprocess
from dsb2015.config import Config


def get_args():
    p = ArgumentParser(description="preprocess and train model")
    p.add_argument("-p", help="preprocess only",
                   action='store_true', default=False)
    return p.parse_args()


if __name__ == '__main__':
    args = get_args()
    with open('settings.json') as f:
        cfg = Config(**json.load(f))
    if args.p:
        preprocess(cfg)
    else:
        preprocess(cfg)
        # TODO call training

