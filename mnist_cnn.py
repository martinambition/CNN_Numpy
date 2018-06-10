import NN.network as cnn
import struct
import numpy as np
import os
import matplotlib.pyplot as pyplot
import matplotlib as mpl

def read(dataset = "training", path = "./data/mnist"):
    if dataset is "training":
        fname_img = os.path.join(path, 'train-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels-idx1-ubyte')
    elif dataset is "testing":
        fname_img = os.path.join(path, 't10k-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels-idx1-ubyte')
    else:
        raise ValueError("dataset must be 'testing' or 'training'")

    # Load everything in some numpy arrays
    with open(fname_lbl, 'rb') as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        lbl = np.fromfile(flbl, dtype=np.int8)

    with open(fname_img, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols)

    #get_img = lambda idx: [lbl[idx], img[idx]]

    # Create an iterator which returns each image in turn
    # for i in range(len(lbl)):
    #     yield get_img(i)
    return lbl,img

def show(image):

    fig = pyplot.figure()
    ax = fig.add_subplot(1,1,1)
    imgplot = ax.imshow(image, cmap=mpl.cm.Greys)
    imgplot.set_interpolation('nearest')
    ax.xaxis.set_ticks_position('top')
    ax.yaxis.set_ticks_position('left')
    pyplot.show()


def normalization(image):
    #mean 0.1307, 0.3081 std Deviation
    return (image/255-0.1307)/0.3081

def test_verify(net,labels,images):
    images = normalization(images)
    images = np.expand_dims(images, axis=1)
    test_size = len(labels)
    #size*10
    ret = net.forward(images)
    predict = np.argmax(ret,axis=1)
    correct = 0;
    for i in range(test_size):
        if predict[i] == labels[i]:
            correct+=1
    return correct,test_size
net = cnn.Network()
training_data_labels,training_data_images = read('training')
test_data_labels,test_data_images = read('testing')
batch_size =64
count = 0





train_size =  len(training_data_labels)
for epoch in range(3):
    for i in range(int(train_size/batch_size)):
        begin =  i * batch_size
        if (i + 1) * batch_size > train_size -1:
            end = train_size -1
        else:
            end = (i + 1) * batch_size
        images =  training_data_images[begin:end]
        labels =  training_data_labels[begin:end]

        label_ele = np.zeros((batch_size,10))

        for z in range(batch_size):
            label_ele[z,labels[z]] = 1

        #normalization image
        images =  normalization(images)
        images= np.expand_dims(images,axis=1)
        ret = net.forward(images)
        lost = net.loss(label_ele)

        net.backward(label_ele,rate=0.01,momentum=0.5)

        if count %100 == 0:
            test_correct_size,test_size =  test_verify(net,test_data_labels[0:500],test_data_images[0:500])
            print("Epoch:", epoch, " Count:", count, "Test Accurency:", test_correct_size/test_size *100,"%")

        # pyplot.scatter(count , lost)
        # pyplot.pause(0.1)
        count = count + 1
        print("Epoch:",epoch," Count:", count," Lost:",lost)

