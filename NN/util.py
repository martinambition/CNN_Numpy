import numpy as np

class Util:

    @staticmethod
    def patchify(img, patch_shape, stepsize_x=1, stepsize_y=1):
        strided = np.lib.stride_tricks.as_strided
        x, y = patch_shape
        p, q = img.shape[-2:]
        sp, sq = img.strides[-2:]

        out_shp = img.shape[:-2] + (p - x + 1, q - y + 1, x, y)
        out_stride = img.strides[:-2] + (sp, sq, sp, sq)

        imgs = strided(img, shape=out_shp, strides=out_stride)
        return imgs[..., ::stepsize_x, ::stepsize_y, :, :]
