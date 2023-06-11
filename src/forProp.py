import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def ReLU(Z):
    #activation function of choice for CNN based neural networks (Goodfellow et al. 2016)
    return np.maximum(0,Z)

def LeakyReLU(coef, Z):
    #idk its like the activation for the first layer but leakier?
    return np.maximum(coef * Z, Z)

def Sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

def TanH(Z):
    return (np.exp(Z) - np.exp(-Z)) / (np.exp(Z) + np.exp(-Z))

def Softmax(Z):
    return np.exp(Z) / sum(np.exp(Z))
    

class ForwardFeed_Train:
    def __init__(self, train):
        self.train_data = train
        self.input_dims = train.shape[0] #Input layer dimensions
        self.n_H1 = 256 #First hidden layer
        self.n_H2 = 64 #Second hidden layer
        self.n_H3 = 10 #Output nodes
        self.W1 = np.random.rand(self.n_H1, self.input_dims) - 0.5
        self.B1 = np.random.rand(self.n_H1, 1) - 0.5
        self.W2 = np.random.rand(self.n_H2, self.n_H1) - 0.5
        self.B2 = np.random.rand(self.n_H2, 1) - 0.5
        self.W3 = np.random.rand(self.n_H3, self.n_H2) - 0.5
        self.B3 = np.random.rand(self.n_H3, 1) - 0.5

    def init_weights(self):
        return self.W1, self.B1, self.W2, self.B2, self.W3, self.B3

    def HL1_fw(self, W1, B1):
        #Hidden layer 1 
        Z1 = np.matmul(W1, self.train_data) + B1
        A1 = ReLU(Z1) #Can be swapped out for other above activation functions
        return Z1, A1

    def HL2_fw(self, W2, B2, A1):
        #Hidden layer 2
        Z2 = np.matmul(W2, A1) + B2
        A2 = LeakyReLU(0.1, Z2) #Can be swapped out for other above activation functions
        return Z2, A2
    
    def HL3_fw(self, W3, B3, A2):
        #Output layer
        Z3 = np.matmul(W3, A2) + B3
        A3 = Softmax(Z3)
        return Z3, A3
    

class ForwardFeed_Test:
    def __init__(self, test_data, wb, index):
        self.test_data = pd.read_csv(test_data).T
        j,_ = self.test_data.shape
        self.test_datum = self.test_data[index][1:j]
        self.test_lab = self.test_data[index][0]
        self.wb = wb
        self.index = index

    def HL1_fw(self):
        Z1 = np.matmul(self.wb[0], self.test_datum) + self.wb[1]
        A1 = ReLU(Z1)
        return A1

    def HL2_fw(self, A1):
        Z2 = np.matmul(self.wb[2], A1) + self.wb[3]
        A2 = LeakyReLU(0.1, Z2)
        return A2

    def HL3_fw(self, A2):
        Z3 = np.matmul(self.wb[4], A2) + self.wb[5]
        A3 = Softmax(Z3)
        return A3
    
    def predict(self, A3):
        pred = np.argmax(A3, 1) 
        label = self.test_lab
        return label, pred
    
    def imShow(self):
        image = self.test_datum.values.reshape((28, 28)) * 255
        plt.gray()
        plt.imshow(image, interpolation='nearest')
        plt.show()