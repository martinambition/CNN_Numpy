import numpy as np
from .util import Util
class MaxPooling:
    def __init__(self,size):
        self.size = size


    def forward(self,input):
        #downsample. M
        self.input =  input
        self.batch_size = input.shape[0]
        self.channel =input.shape[1]
        self.target_size = int(input.shape[2]/self.size)
        ret = np.zeros((self.batch_size,self.channel,self.target_size,self.target_size))
        #self.arg_max = np.zeros((target_size, target_size))
        self.arg_max = dict()

        #64,10,24,24 - > 64, 10,12,12 ,2,2
        #pooling_window =  Util.patchify(input,(self.size,self.size),self.size,self.size)
        #ret= np.amax(pooling_window,axis=(4,5))

        #ret = ret[..., np.newaxis, np.newaxis]
        #ret = np.tile(ret, (1, 1, 1, 1, 2, 2))
        #arg = np.where(pooling_window == ret, 1, 0)


        for z in range(self.batch_size):
            for k in range(self.channel):

                for i in range(self.target_size):
                    for j in range(self.target_size):
                        xStart = i*self.size
                        yStart = j*self.size
                        temp =input[z,k,xStart: xStart+ self.size,yStart: yStart+ self.size]
                        val = np.max(temp)
                        ret[z,k,i,j]=val
                        arg =  np.where(temp == val)
                        self.arg_max[(z,k,i,j)] = (z,k,arg[0][0]+xStart,arg[1][0]+yStart)

                #self.arg_max[i,j]= (arg[0][0]+xStart,arg[1][0]+yStart)
        return ret

    def backward(self,error):
        error_out = np.zeros(self.input.shape);
        for z in range(self.batch_size):
            for k in range(self.channel):
                for i in range(self.target_size):
                    for j in range(self.target_size):
                        error_out[self.arg_max[(z,k,i,j)]] = error[z,k,i,j]
        return error_out