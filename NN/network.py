from .sigmoid import  Sigmoid
from .liner import  Linear
from .softmax import  SoftmaxLoss
from .max_pooling import  MaxPooling
from .conv import Conv
from .relu import Relu

import numpy as np

class Network:

    def __init__(self):
        self.c1 = Conv(1,10,5,5)
        self.ac1 =  Relu()
        self.pooling1 = MaxPooling(2)

        self.c2 = Conv(10,20,5,5)
        self.ac2 = Relu()
        self.pooling2 = MaxPooling(2)

        self.f1 = Linear(320, 50)
        self.f2 = Linear(50, 10)

        self.ac3 = Relu()
        self.softmax = SoftmaxLoss();

    def forward(self,input):
        #28*28
        c1_out =  self.c1.forward(input)
        # 24*24
        pool1_out = self.pooling1.forward(c1_out)
        # 24*24
        ac1_out = self.ac1.forward(pool1_out)

        # 12*12
        c2_out = self.c2.forward(ac1_out)
        # 8*8
        pool2_out = self.pooling2.forward(c2_out)
        # 8*8
        ac2_out = self.ac2.forward(pool2_out)

        ac2_out = ac2_out.reshape(-1, 320)

        # 4*4
        f1_out = self.f1.forward(ac2_out)
        # 1 * 10
        ac3_out =  self.ac3.forward(f1_out)

        f2_out =  self.f2.forward(ac3_out)
        out = self.softmax.forward(f2_out)
        return out

    def  backward(self,target,rate,momentum):
        error_soft =  self.softmax.backward(target)
        error_f2 = self.f2.backward(error_soft, rate,momentum)
        error_ac3 = self.ac3.backward(error_f2)
        error_f1 =  self.f1.backward(error_ac3,rate,momentum)

        error_f1 = error_f1.reshape((-1, 20, 4,4))

        error_ac2 = self.ac2.backward(error_f1)
        error_p2 = self.pooling2.backward(error_ac2)
        error_c2 = self.c2.backward(error_p2,rate,momentum)

        error_ac1 = self.ac1.backward(error_c2)
        error_p1 = self.pooling1.backward(error_ac1)
        error = self.c1.backward(error_p1,rate,momentum)

    def loss(self,target):
        return self.softmax.cal_loss(target);

