from src.backProp import Train
from src.forProp import ForwardFeed_Train
from argparse import ArgumentParser
from time import time
import numpy as np
import pickle



def main():
    parser = ArgumentParser(description='Neural network using python and numpy')
    parser.add_argument('training_data', type=str, help='training data set')
    parser.add_argument('--alpha', type=float, default=0.001, help='learning rate for training, should not exceed 0.005')
    parser.add_argument('--iterations', type=int, default=500, help='amount of training interations through dataset')
    args = parser.parse_args()


    T = Train(args.training_data, args.alpha) 
    train_data, train_labs = T.DataRet()
    ff = ForwardFeed_Train(train_data)
    W1, B1, W2, B2, W3, B3 = ff.init_weights()

    start_time = time()
    for step in range(args.iterations):
        #1) forward prop step with initially random generated values
        Z1, A1 = ff.HL1_fw(W1, B1)
        Z2, A2 = ff.HL2_fw(W2, B2, A1)
        _, A3 = ff.HL3_fw(W3, B3, A2)

        #2) compute loss of forward prop: compare one hot array true values vs output from neural network forward feed
        loss = T.ceLoss(train_labs, A3)

        #3) Layer 3 -> 2 backprop using derivative functions of forProp step (Softmax)
        dZ3, dW3, dB3 = T.HL3_bk(A2, A3, train_labs)

        #4) Layer 2 -> 1 backprop using derivative functions of forProp step (Leaky ReLU)
        dZ2, dW2, dB2 = T.HL2_bk(A1, Z2, W3, dZ3)

        #5) Layer 1 -> Input layer backprop using derivative functions of forProp step (ReLU)
        _, dW1, dB1 = T.HL1_bk(Z1, W2, dZ2, train_data)

        #6) Update weights and bias values for next iteration (new weight = previous weight - learning rate (alpha) * derivative weight from backprop step)
        W1, B1, W2, B2, W3, B3 = T.UpdateWB(W1, B1, W2, B2, W3, B3, dW1, dB1, dW2, dB2, dW3, dB3)

        if step % 10 == 0:
            print("Iteration", step, "loss: ", loss)


    end_time = time()
    training_time = round((end_time - start_time) / 60, 2)
    print("Final loss: ", loss)
    print("Elapsed training time: ", training_time, "minutes")


    weights = [W1, B1, W2, B2, W3, B3]
    with open('data/output.weights', 'wb') as f:
        pickle.dump(weights, f)
    print('Weights and bias values written!')
    

if __name__ == '__main__':
    main()