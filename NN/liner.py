import numpy as np
import math
class Linear:
    def __init__(self,input_nodes,output_nodes):
        self.input_nodes = input_nodes
        self.output_nodes = output_nodes
        stdv = 1. / math.sqrt(input_nodes)
        self.param = np.random.uniform(-stdv,stdv,(self.input_nodes + 1,self.output_nodes))
        self.velocity = np.zeros(self.param.shape)


    def forward(self,input):
        # 64 28*28
        self.batch_size = input.shape[0]
        self.bais = np.zeros((self.batch_size,1))
        #input =  input.reshape(self.batch_size, self.input_nodes)
        self.input_value =  np.concatenate([self.bais,input],1)
        output_value =  np.dot(self.input_value,self.param)
        return output_value

    def backward(self,error,rate,momentum=0.5):
        #error =64* node out = 64*input
        param_derivative = self.input_value.transpose().dot(error)
        #input = 64 * (1+28*28)
        self.out_error =  error.dot(self.param.transpose())[:,1:]

        self.velocity = momentum * self.velocity + rate * param_derivative;
        self.param =  self.param - self.velocity
        origin_size = int(math.sqrt(self.input_nodes))
        #self.out_error = self.out_error.reshape(self.batch_size,origin_size,origin_size)
        return self.out_error