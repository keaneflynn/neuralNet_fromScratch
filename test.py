from src.forProp import ForwardFeed

from argparse import ArgumentParser
import pickle as p
import random as r


def main():
    parser = ArgumentParser(description='Testing the weights and bias values generated from train.py')
    parser.add_argument('weights_file', type=str, help='weights and bias file created from training')
    parser.add_argument('--index', type=float, default=r.randint(1,100), help='select random value from test dataset')   
    args = parser.parse_args()


    with open(args.weights_file, 'rb') as f:
        wb = p.load(f)


if __name__ == '__main__':
    main()