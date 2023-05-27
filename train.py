from src.backProp import Train
from src.forProp import forwardFeed
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
        #1) forward prop step with initially random generated values
        #2) compute loss of forward prop: compare one hot array true values vs output from neural network forward feed
        #3) Layer 3 -> 2 backprop using derivative functions of forProp step (Softmax)
        #4) Layer 2 -> 1 backprop using derivative functions of forProp step (Leaky ReLU)
        #5) Layer 1 -> Input layer backprop using derivative functions of forProp step (ReLU)
        #6) Update weights and bias values for next iteration (new weight = previous weight - learning rate (alpha) * derivative weight from backprop step)

        if step % train_data.shape[1] == 0:
            print("Iteration", step, "loss: ", loss)

    print("Final loss: ", loss)


if __name__ == '__main__':
    args()
    main()