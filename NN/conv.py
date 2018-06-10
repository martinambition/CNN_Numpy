import numpy as np
import math
from .util import Util
class Conv:
    def __init__(self,input_channel, out_channel,kernel_width,kernel_height):
        self.input_channel = input_channel
        self.out_channel = out_channel
        self.kernel_w = kernel_width
        self.kernel_h = kernel_height

        n = self.input_channel* kernel_width
        stdv = 1. / math.sqrt(n)
        self.kernel_param = np.random.uniform(-stdv,stdv,(out_channel,input_channel,self.kernel_w ,self.kernel_w))
        self.velocity = np.zeros(self.kernel_param.shape)
        self.bais = np.zeros((out_channel))


    def forward(self,input):
        self.input = input
        #[64,1,28,28]
        self.batch_size = input.shape[0]
        self.output_w = self.input.shape[2]-self.kernel_w + 1
        self.output_h = self.input.shape[3]-self.kernel_h + 1
        self.output = np.zeros((self.batch_size,self.out_channel,self.output_w,self.output_h))

        for z in range(self.batch_size):
            for i in range(self.out_channel):
                stride_img = Util.patchify(self.input[z],(self.kernel_w,self.kernel_h))
                self.output[z,i]= np.einsum("zpkij,zij->pk",  stride_img, self.kernel_param[i]) \
                                  + self.bais[i]*self.input_channel
            # # 24*24*5*5
            # chanel_total = 0
            # for c in self.out_channel:
            #     for i in range(self.output_w):
            #         for j in range(self.output_h):
            #             self.output[z,i,j] = \
            #                 np.sum(self.input[z,i:i+self.kernel_w, j:j+self.kernel_h] * self.kernel_param[c]) + self.bais[c]
        return self.output

    def backward(self,error,rate,momentum=0.5):
        rotate_param =  np.rot90(self.kernel_param, 2,axes=(2,3))
        pad = self.kernel_w-1
        padding_error = z = np.lib.pad(error, ((0,0),(0,0),(pad,pad),(pad,pad)), 'constant', constant_values=0)
        #np.lib.pad(error, self.kernel_w-1, 'constant', constant_values=0)
        #self.out_error = np.zeros(self.input.shape)
        self.out_error = np.zeros((self.input.shape))
        param_derivative= np.zeros(self.kernel_param.shape)
        param_derivative_esi = np.zeros(self.kernel_param.shape)
        bais_derivative = np.einsum('abcd -> b',error)

        stride_padding = Util.patchify(padding_error, (self.kernel_w, self.kernel_w))
        #stride_padding: 64*10*24*24*5*5 , rotate_param: 1*5*5
        #np.einsum("abcdef,gef->a", stride_padding, rotate_param)



        for z in range(self.batch_size):
                for i in range(self.input.shape[2]):
                    for j in range(self.input.shape[3]):
                        for t in range(self.input_channel):
                            #rotate_param 20*10*5*5
                            self.out_error[z, t, i, j] = np.sum(padding_error[z, t, i:i + self.kernel_w, j:j + self.kernel_h] * rotate_param[:,t])



        #self.out_error = np.tile(self.out_error, (1, self.input_channel, 1, 1))

        #stride_imgs = Util.patchify(self.input, (self.output_w, self.output_h))


        for c in range(self.out_channel):
            for i in range(self.kernel_w):
                for j in range(self.kernel_w):
                    for ci in range(self.input_channel):
                        #error 64*20*24*24,  input 64*24*24
                        z =self.input[:,ci, i:i + self.output_w, j:j + self.output_h]
                        #z = np.expand_dims(z,axis=1)
                        #z = z.tile((1,self.out_channel,1,1))
                        t =np.sum(z * error[:,c])
                        param_derivative[c, ci, i, j] = t

                    #64*24*24  64*24*24

        #param_derivative_esi = np.einsum("agcdef,agef->cd", stride_imgs, error)

        # for i in range(self.kernel_w):
        #     for j in range(self.kernel_h):
        #         param_derivative[i,j] = np.sum(self.input[:,i:self.output_w+i,j:self.output_h+j] * error)
        self.velocity = momentum * self.velocity + rate * param_derivative;
        self.kernel_param =self.kernel_param-self.velocity
        self.bais = self.bais - bais_derivative*rate
        return self.out_error