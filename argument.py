import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--save-path", type=str, default="./finetuned_weights")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--vision_encoder", type=int, default=10)
    parser.add_argument("--text_encoder", type=int, default=0)
    # args=parser.parse_args(args)
    return parser.parse_args()
