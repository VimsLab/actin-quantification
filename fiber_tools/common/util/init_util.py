import argparse
from ..config import cfg, cfg_from_file

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', nargs='?', type=str, help="cfg")
    args = parser.parse_args()
    cfg_from_file(args.cfg)
    return cfg