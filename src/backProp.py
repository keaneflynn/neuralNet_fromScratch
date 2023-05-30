import numpy as np
import pandas as pd
from keras.utils import to_categorical


def Onehot(train_labs):
    #N=10 vector of truth labels for each training iteration
    #Convert from sparse labels to categorical (ie [4] to [0,0,0,0,1,0,0,0,0,0])
    return np.array(train_labs[:, None]==np.arange(10), dtype=np.float32).T

def dReLU(Z):
    #Derivative of ReLU function, slope of 1 if Z > 0, else is 0
    return Z > 0

def dLeaky_ReLU(coef, Z):
    return coef if Z.any() < 0 else 1
    

class Train:
    def __init__(self, train, test, alpha):
        self.train = pd.read_csv(train).T #Transpose data
        self.test = pd.read_csv(test).T #Transpose data
        self.train = np.array(self.train)
        self.test = np.array(self.test)
        j,k = self.train.shape
        l,m = self.test.shape
        self.train_data = self.train[1:k] / 255 #normalise data to 0-1 values for pixel color
        self.test_data = self.test[1:m] / 255 #normalise data to 0-1 values for pixel color
        self.train_labs = self.train[0] 
        self.test_labs = self.test[0]
        self.alpha = alpha
        self.nh_0 = self.train_data.shape[0]
        self.nh_1 = 256 #Hidden layer 1 dimension, adjust as desired
        self.nh_2 = 64 #Hidden layer 2 dimension, adjust as desired

    def DataRet(self):
        return self.train_data, self.test_data, self.train_labs, self.test_labs

    def HL1_bk(self, Z1, W2, dZ2, train_data):
        #Hidden layer 1 backpropogation iteration from ReLU to Input layer
        dZ1 = np.matmul(W2.T, dZ2) * dReLU(Z1)
        dW1 = (1/self.nh_0) * np.matmul(dZ1, train_data.T)
        dB1 = (1/self.nh_0) * np.sum(dZ1)
        return dZ1, dW1, dB1

    def HL2_bk(self, A1, Z2, W3, dZ3):
        #Hidden layer 2 backpropogation iteration from leaky ReLU to HL1
        dZ2 = np.matmul(W3.T, dZ3) * dLeaky_ReLU(0.1, Z2)
        dW2 = (1/self.nh_0) * np.matmul(dZ2, A1.T)
        dB2 = (1/self.nh_0) * np.sum(dZ2)
        return dZ2, dW2, dB2

    def HL3_bk(self, A2, A3, train_labs):
        #Hidden layer 3 backpropogation iteration from softmax to HL2
        dZ3 = A3 - Onehot(train_labs) 
        dW3 = (1/self.nh_0) * np.matmul(dZ3, A2.T)
        dB3 = (1/self.nh_0) * np.sum(dZ3) #perhaps keep axis and dimension values if the matrix dont matrices
        return dZ3, dW3, dB3
    
    def UpdateWB(self, W1, B1, W2, B2, W3, B3, dW1, dB1, dW2, dB2, dW3, dB3):
        #Update weight and bias matrices for each hidden layer based on backpropogation results
        W1 = W1 - self.alpha * dW1
        B1 = B1 - self.alpha * dB1
        W2 = W2 - self.alpha * dW2
        B2 = B2 - self.alpha * dB2
        W3 = W3 - self.alpha * dW3
        B3 = B3 - self.alpha * dB3
        return W1, B1, W2, B2, W3, B3 

    def ceLoss(self, y, y_soft):
        #Cross-entropy loss function for categorical output layer
        loss_sum = np.sum(np.multiply(Onehot(y), np.log(y_soft)))  #y = validation array from oneHot, y_soft = softmax calculation at final output layer 
        n = self.train.shape[1]
        loss_avg = -(1/n) * loss_sum
        return loss_avg