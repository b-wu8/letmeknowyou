import argparse
from logging import raiseExceptions
import os

def get_args():
    p = argparse.ArgumentParser(description="Paring Arguments")

    p.add_argument("--mode", type=str, help="Train or Test mode")
    p.add_argument("--batch-size", type=int, help="Batch size")

def check_args(args):
    mandatory_args = {'mode'}
    if not mandatory_args.issubset(set(dir(args))):
        raise Exception("Missing essential arguments."
                        "Please pass them in as command line arguments")
    if args.mode is None:
        raise Exception("Please specify to train or test")