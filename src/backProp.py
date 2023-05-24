import numpy as np
import pandas as pd

train = pd.read_csv(r'~/code/python/neuralNet_fromScratch/data/train.csv').T
test = pd.read_csv(r'~/code/python/neuralNet_fromScratch/data/test.csv').T
print(train.shape[0])

class Train:
    def __init__(self, train, test, alpha):
        self.train = train
        self.test = test
        self.alpha = alpha
        self.nh_0 = train.shape[0]
        self.nh_1 = 256
        self.nh_2 = 64
        self.w1 = 0
        self.w2 = 0
        self.b1 = 0
        self.b2 = 0
        

    def oneHot(self):
    #N=10 vector of truth labels for each training iteration
    #Convert from sparse categorical labels to categorical (ie [4] to [0,0,0,0,1,0,0,0,0,0])
        return 0

    def HL1(self):
    #Hidden layer 1
        return 0

    def HL2(self, ):
    #Hidden layer 2
        return HL2_outputs

    def softmax(self, HL2_outputs):
    #Converting output node logits into probabilities
        return np.exp(HL2_outputs) / np.sum(np.exp(HL2_outputs))

    def ceLoss(self, y, y_soft):
    #Cross-entropy loss function for categorical output layer
        loss_sum = np.sum(np.multiply(y, np.log(y_soft))) #y is validation array from oneHot and y_soft is softmax calculation at final output layer 
        n = train.shape[1]
        loss_avg = -(1/m) * loss_sum
        return loss_avg
