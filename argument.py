import argparse
def parse_args():
    parser=argparse.ArgumentParser()
    parser.add_argument(
        "--batchsize", type=int, default=16
    )
    parser.add_argument(
        "--unfreeze_layer", type=int, default=4
    )
    parser.add_argument(
        "--save_path", type=str, default='./finetuned_weights'
    )
    # args=parser.parse_args(args)
    return parser.parse_args()