# Convolutional Neural Network written by pure numpy.
This CNN is written by pure numpy and support trainning by BP algorithem.
It follows pytorch's code style(forward, backward).
Due to numpy limiation, it doesn't use GPU to accelerate. But the training speed is acceptable for the small dataset,(e.g. mnist).

1. It implements the SGD (Stochastic Gradient Descent)
2. It support multiple channel.

# Install:
Downlaod the mnist from [website](http://yann.lecun.com/exdb/mnist/), and extract all the files into data/mnist

## Run:
python mnist_cnn.py
