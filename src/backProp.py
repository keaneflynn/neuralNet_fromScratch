import numpy as np
import pandas as pd
from keras.utils import to_categorical


###Testing env###
train = pd.read_csv(r'~/code/python/neuralNet_fromScratch/data/train.csv').T
test = pd.read_csv(r'~/code/python/neuralNet_fromScratch/data/test.csv').T
print(train.shape[0])
###


class Train:
    def __init__(self, train, test, alpha):
        self.train = pd.read_csv(train).T
        self.test = pd.read_csv(test).T
        self.train = np.array(self.train)
        self.test = np.array(self.test)
        j,k = self.train.shape
        l,m = self.test.shape
        np.random.shuffle(self.train)
        np.random.shuffle(self.test)
        self.train_data = self.train[1:k] / 255
        self.test_data = self.test[1:m] / 255
        self.train_labs = self.train[0]
        self.test_labs = self.test[0]
        self.alpha = alpha
        self.nh_0 = self.train_data.shape[0]
        self.nh_1 = 256 #Hidden layer 1 dimension, adjust as desired
        self.nh_2 = 64 #Hidden layer 2 dimension, adjust as desired
        self.W1 = 0
        self.W2 = 0
        self.B1 = 0
        self.B2 = 0

    def __new__(self):
        return self.train_data, self.test_data, self.train_labs, self.test_labs

    def Onehot(self):
        #N=10 vector of truth labels for each training iteration
        #Convert from sparse labels to categorical (ie [4] to [0,0,0,0,1,0,0,0,0,0])
        return to_categorical(self.train_labs)
    
    def dReLU(Z):
        #Derivative of ReLU function, slope of 1 if Z > 0, else is 0
        return Z > 0

    def HL1_bk(self):
        #Hidden layer 1
        return x

    def HL2_bk(self):
        #Hidden layer 2
        return x

    def HL3_bk(self):
        #Hidden layer 2
        return x

    def ceLoss(self, y, y_soft):
        #Cross-entropy loss function for categorical output layer
        loss_sum = np.sum(np.multiply(y, np.log(y_soft))) #y = validation array from oneHot, y_soft = softmax calculation at final output layer 
        n = self.train.shape[1]
        loss_avg = -(1/n) * loss_sum
        return loss_avg
