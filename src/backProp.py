import numpy as np
import pandas as pd

###Testing env###
train = pd.read_csv(r'~/code/python/neuralNet_fromScratch/data/train.csv').T
test = pd.read_csv(r'~/code/python/neuralNet_fromScratch/data/test.csv').T
print(train.shape[0])
###

class LoadData:
    #Import training and testing data
    def __init__(self, train, test):
        self.train = pd.read_csv(r'~/code/python/neuralNet_fromScratch/data/train.csv').T
        self.test = pd.read_csv(r'~/code/python/neuralNet_fromScratch/data/test.csv').T

    def data(self):
        return 
        

class Train:
    def __init__(self, train, test, alpha):
        self.train = train
        self.test = test
        self.alpha = alpha
        self.nh_0 = train.shape[0]
        self.nh_1 = 256
        self.nh_2 = 64
        self.W1 = 0
        self.W2 = 0
        self.B1 = 0
        self.B2 = 0
        

    def Onehot(self):
        #N=10 vector of truth labels for each training iteration
        #Convert from sparse labels to categorical (ie [4] to [0,0,0,0,1,0,0,0,0,0])
        return 0
    
    def dReLU(Z):
        return Z

    def HL1_bk(self):
        #Hidden layer 1
        return x

    def HL2_bk(self):
        #Hidden layer 2
        return x

    def dSoftmax(self, x):
        #Converting output node logits into probabilities
        return np.exp(x) / np.sum(np.exp(x))

    def ceLoss(self, y, y_soft):
        #Cross-entropy loss function for categorical output layer
        loss_sum = np.sum(np.multiply(y, np.log(y_soft))) #y = validation array from oneHot, y_soft = softmax calculation at final output layer 
        n = self.train.shape[1]
        loss_avg = -(1/n) * loss_sum
        return loss_avg
