import numpy as np

class activation:
    #Activation functions used

    def ReLU(x):
        #activation function of choice for CNN based neural networks (Goodfellow et al. 2016)
        return max(0,x)

    def LeakyReLU(x):
        return max(0.2*x, x)

    def Sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def TanH(x):
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))




