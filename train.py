from src.backProp import Train
from src.forProp import 
from argparse import ArgumentParser


def args():
    parser = ArgumentParser(decsription='Neural network using python and numpy')
    parser.add_argument('training_data', type=str, help='training data set')
    parser.add_argument('testing_data', type=str, help='testing data set')
    parser.add_argument('alpha', type=float, default=0.001, help='learning rate for training')
    parser.add_argument('epochs', type=int, default=25, help='amount of training interations through dataset')

train_data, test_data, train_labs, test_labs = Train('training_data', 'testing_data', 'alpha')

def main():
    for step in range('epochs'):


if __name__ == '__main__':
    args()
    main()