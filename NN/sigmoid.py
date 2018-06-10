import numpy as np

class Sigmoid:
    def __init__(self):
        pass
    def forward(self,input):
        self.output =  1.0 / (1 + np.exp( input * np.array(-1)))
        return self.output

    def backward(self,error):
        return error*self.output*(1-self.output)

