import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--unfreeze-layer", type=int, default=4)
    parser.add_argument("--save-path", type=str, default="./finetuned_weights")
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--epochs", type=int, default=10)
    # args=parser.parse_args(args)
    return parser.parse_args()
