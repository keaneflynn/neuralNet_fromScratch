from argparse import ArgumentParser


def main():
    parser = ArgumentParser(description='Testing the weights and bias values generated from train.py')
    parser.add_argument('weights_file', type=str, help='weights and bias file created from training')   
    args = parser.parse_args()


    with open(args.weights_file, 'r') as f:
        weights = f.read()

if __name__ == '__main__':
    main()