from preprocessing import *
from argparse import ArgumentParser


def create_parser():
    """
    Creates the arguments to interact with the script.
    :return:
    """
    parser = ArgumentParser()
    parser.add_argument("-c", "--config", default="config.yaml")
    args = parser.parse_args()
    return args


def main():
    args = create_parser()
    config = load_config(args.config)


if __name__ == '__main__':
    main()
