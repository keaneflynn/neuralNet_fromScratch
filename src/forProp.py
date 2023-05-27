import numpy as np


def ReLU(Z):
    #activation function of choice for CNN based neural networks (Goodfellow et al. 2016)
    return max(0,Z)

def LeakyReLU(Z):
    return max(0.1*Z, Z)

def Sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

def TanH(Z):
    return (np.exp(Z) - np.exp(-Z)) / (np.exp(Z) + np.exp(-Z))

def Softmax(Z):
    return np.exp(Z) / sum(np.exp(Z))
    

class forwardFeed:
    def __init__(self, train):
        self.input_dims = train.shape[1] #Input layer dimensions
        self.n_H1 = 256 #First hidden layer
        self.n_H2 = 64 #Second hidden layer
        self.n_H3 = 10 #Output nodes
        self.W1 = np.random.rand(self.n_H1, self.input_dims) - 0.5
        self.B1 = np.random.rand(self.n_H1, 1) - 0.5
        self.W2 = np.random.rand(self.n_H2, self.n_H1) - 0.5
        self.B2 = np.random.rand(self.n_H2, 1) - 0.5
        self.W3 = np.random.rand(self.n_H3, self.n_H2) - 0.5
        self.B3 = np.random.rand(self.n_H3, 1) - 0.5

    def HL1_fw(self, Z):
        #Hidden layer 1 
        Z1 = np.matmul(self.W1, Z) + self.B1
        A1 = ReLU(Z1) #Can be swapped out for other above activation functions
        return Z1, A1

    def HL2_fw(self, A1):
        #Hidden layer 2
        Z2 = np.matmul(self.W2, A1) + self.B2
        A2 = LeakyReLU(Z2) #Can be swapped out for other above activation functions
        return Z2, A2
    
    def HL3_fw(self, A2):
        #Output layer
        Z3 = np.matmul(self.W3, A2) + self.B3
        A3 = Softmax(Z3)
        return Z3, A3
