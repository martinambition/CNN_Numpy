class Relu:
    def __init__(self):
        pass
    def forward(self,input):
        self.input = input
        return input * (input > 0)
    def backward(self,error):
        return  error * (self.input > 0)