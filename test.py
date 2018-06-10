import numpy as np

#64*20*24*24,  input 64*24*24

def patchify(img, patch_shape, stepsize_x=1, stepsize_y=1):
    strided = np.lib.stride_tricks.as_strided
    x, y = patch_shape
    p,q = img.shape[-2:]
    sp,sq = img.strides[-2:]

    out_shp = img.shape[:-2] + (p-x+1,q-y+1,x,y)
    out_stride = img.strides[:-2] + (sp,sq,sp,sq)

    imgs = strided(img, shape=out_shp, strides=out_stride)
    return imgs[...,::stepsize_x,::stepsize_y,:,:]

a = np.array([
        [[ 0,  1,  2,  3,  4],
        [ 5,  6,  7,  8,  9],
        [10, 11, 12, 13, 14],
        [15, 16, 17, 18, 19],
        [20, 21, 22, 23, 24]],

        [[ 0,  1,  2,  3,  4],
        [ 5,  6,  7,  8,  9],
        [10, 11, 12, 13, 14],
        [15, 16, 17, 18, 19],
        [20, 21, 22, 23, 24]],


])

# t = np.array([[[1,2],[3,4]],[[1,2],[3,4]]])
# #t = np.array([[1,2],[3,4]])
# print(t)
# z = np.rot90(t, 2,axes=(1,2))

# z = np.lib.pad(z, ((0, 0), (1, 1), (1, 1)), 'constant', constant_values=0)
# print(z)
#



print(a.shape)
z =a[np.newaxis,:]
print(z.shape)
t = patchify(z,(2,2))
print(t.shape)
param = np.random.uniform(-1,1,(2,2))
z= np.einsum("ij,tzpkij->tzpk",param,t)
print(z)

#
# t = np.zeros((64, 10,24,24))
#
# pooling_window = patchify(t, (2, 2), 2, 2)
#
# print(pooling_window.shape)
# ret = np.amax(pooling_window, axis=(4,5))
# ret =ret[...,np.newaxis,np.newaxis]
# ret = np.tile(ret,(1,1,1,1,2,2))
# z = np.where(pooling_window!=ret,pooling_window,0)
# print(z.shape)