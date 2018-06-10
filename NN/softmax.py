import numpy as np
class SoftmaxLoss:
    def __init__(self):
        pass
    def forward(self,input):
        #64 * input
        self.input = input;
        self.batch_size = input.shape[0]
        temp = np.exp(self.input)
        self.output = temp / (np.sum(temp,1).reshape(self.batch_size ,1))
        if not np.isfinite(self.output).all():
            print('wrong')
        return self.output

    #Cross Entropy
    def cal_loss(self,target):
        #output 64*
        loss = 0

        try:
            for i in range(self.batch_size):
                temp =self.output[i].dot( np.transpose(target[i]))
                if temp  == 0:
                    print("wrong");
                if not np.isfinite(temp).all():
                    print('wrong')
                loss =  loss  - np.log(temp)
            loss = loss/self.batch_size
        except RuntimeWarning as exe:
            print(exe)
        return loss;



    def backward(self,target):
        return (self.output - target)/self.batch_size;