from src.backProp import Train
from src.forProp import forwardFeed
from argparse import ArgumentParser


def args():
    parser = ArgumentParser(decsription='Neural network using python and numpy')
    parser.add_argument('training_data', type=str, help='training data set')
    parser.add_argument('testing_data', type=str, help='testing data set')
    parser.add_argument('alpha', type=float, default=0.001, help='learning rate for training')
    parser.add_argument('epochs', type=int, default=25, help='amount of training interations through dataset')


train_data, test_data, train_labs, test_labs = Train('training_data', 'testing_data', 'alpha') #Takes 1 positional argument but 4 were given, fix tomorrow youre tired
ff = forwardFeed(train_data)
W1, B1, W2, B2, W3, B3 = ff.init_weights()
print(W1)

def main():
    for step in range('epochs'):
        #1) forward prop step with initially random generated values
        Z1, A1 = ff.HL1_fw(W1, B2)
        Z2, A2 = ff.HL2_fw(W2, B2, A1)
        Z3, A3 = ff.HL3_fw(W3, B3, A2)

        #2) compute loss of forward prop: compare one hot array true values vs output from neural network forward feed
        loss = Train.ceLoss(train_labs, A3)

        #3) Layer 3 -> 2 backprop using derivative functions of forProp step (Softmax)
        dZ3, dW3, dB3 = Train.HL3_bk(A2, A3, train_labs)

        #4) Layer 2 -> 1 backprop using derivative functions of forProp step (Leaky ReLU)
        dZ2, dW2, dB2 = Train.HL2_bk(A1, Z2, W3, dZ3)

        #5) Layer 1 -> Input layer backprop using derivative functions of forProp step (ReLU)
        dZ1, dW1, dB1 = Train.HL1_bk(Z1, W2, dZ2, train_data)

        #6) Update weights and bias values for next iteration (new weight = previous weight - learning rate (alpha) * derivative weight from backprop step)
        W1, B1, W2, B2, W3, B3 = Train.UpdateWB('alpha', dW1, dB2, dW2, dB2, dW3, dB3)

        if step % train_data.shape[1] == 0:
            print("Iteration", step, "loss: ", loss)

    print("Final loss: ", loss)


if __name__ == '__main__':
    args()
    main()