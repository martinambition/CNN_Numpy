import pickle
import numpy as np
import cv2
cv2.namedWindow('Video',cv2.WINDOW_AUTOSIZE)
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

dict =  unpickle('./data/cifar10/data_batch_1')
imgs = np.reshape(dict[b'data'], (10000,3,32,32))
#
# img = dict[b'data'][22]
#
# r = img[0:1024].reshape(32,32)
# g = img[1024:2048].reshape(32,32)
# b = img[2048:3072].reshape(32,32)
# img = np.stack((b,g,r),axis=2)
#
# img = cv2.resize(img, (128,128))
img = imgs[0]
img = np.swapaxes(img,0,2)
img = np.swapaxes(img,0,1)
#img_merged = cv2.merge([img[2], img[1], img[0]])
#cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
print(img.shape)
cv2.imshow('Video', img)
key = cv2.waitKey(0)
cv2.destroyAllWindows()