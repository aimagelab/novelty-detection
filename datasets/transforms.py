import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage.morphology import binary_dilation


class ToFloatTensor1D(object):
    """ Convert vectors to FloatTensors """

    def __call__(self, sample):
        X, Y = sample

        return torch.from_numpy(X), torch.from_numpy(Y)


class ToFloatTensor2D(object):
    """ Convert images to FloatTensors """

    def __call__(self, sample):
        X, Y = sample

        X = np.array(X)
        Y = np.array(Y)

        # swap color axis because
        # numpy image: B x H x W x C
        X = X.transpose(2, 0, 1) / 255.
        Y = Y.transpose(2, 0, 1) / 255.

        X = np.float32(X)
        Y = np.float32(Y)
        return torch.from_numpy(X), torch.from_numpy(Y)


class ToFloatTensor3D(object):
    """ Convert videos to FloatTensors """
    def __init__(self, normalize=True):
        self._normalize = normalize

    def __call__(self, sample):
        X, Y = sample

        # swap color axis because
        # numpy image: T x H x W x C
        X = X.transpose(3, 0, 1, 2)
        Y = Y.transpose(3, 0, 1, 2)

        if self._normalize:
            X = X / 255.
            Y = Y / 255.

        X = np.float32(X)
        Y = np.float32(Y)
        return torch.from_numpy(X), torch.from_numpy(Y)

class ToFloatTensor3DMask(object):
    """ Convert videos to FloatTensors """
    def __init__(self, normalize=True, has_x_mask=True, has_y_mask=True):
        self._normalize = normalize
        self.has_x_mask = has_x_mask
        self.has_y_mask = has_y_mask

    def __call__(self, sample):
        X, Y = sample

        # swap color axis because
        # numpy image: T x H x W x C
        X = X.transpose(3, 0, 1, 2)
        Y = Y.transpose(3, 0, 1, 2)

        X = np.float32(X)
        Y = np.float32(Y)

        if self._normalize:
            if self.has_x_mask:
                X[:-1] = X[:-1] / 255.
            else:
                X = X / 255.
            if self.has_y_mask:
                Y[:-1] = Y[:-1] / 255.
            else:
                Y = Y / 255.

        return torch.from_numpy(X), torch.from_numpy(Y)

class RandomMirror(object):
    """ Randomly mirrors tensors horizontally """

    def __call__(self, sample):
        X, Y = sample

        if np.random.rand() > 0.5:
            if X.ndim == 3:  # images
                X = X[:, ::-1, :]
                Y = Y[:, ::-1, :]
            elif X.ndim == 4:  # videos
                X = X[:, :, ::-1, :]
                Y = Y[:, :, ::-1, :]

        return X, Y


class AddNoise(object):
    """ Adds random zero mean noise to torch tensors. """

    def __init__(self, sigma):
        self._sigma = sigma

    def __call__(self, sample):
        X, Y = sample

        X += self._sigma * torch.randn(*X.shape)
        X = torch.clamp(X, 0, 1)

        return X, Y


class ToFloat32(object):
    """ Casts. """

    def __call__(self, sample):
        X, Y = sample

        return np.float32(X), np.float32(Y)


class RemoveMean(object):
    """ Removes mean value of an image. """

    def __call__(self, sample):
        X, Y = sample

        if X.ndim == 3:  # images
            h, w, c = X.shape
            X -= np.mean(np.reshape(X, newshape=(-1, c)), axis=0)
            Y -= np.mean(np.reshape(Y, newshape=(-1, c)), axis=0)
        elif X.ndim == 4:  # videos
            t, h, w, c = X.shape
            X -= np.mean(np.reshape(X, newshape=(-1, c)), axis=0)
            Y -= np.mean(np.reshape(Y, newshape=(-1, c)), axis=0)

        return X, Y


class OCRemoveMean(object):
    """ Removes mean value of an image. """

    def __call__(self, sample):
        X, Y = sample

        if X.ndim == 3:  # images
            h, w, c = X.shape
            X -= np.mean(np.reshape(X, newshape=(-1, c)), axis=0)
        elif X.ndim == 4:  # videos
            t, h, w, c = X.shape
            X -= np.mean(np.reshape(X, newshape=(-1, c)), axis=0)

        return X, Y


class OCToFloatTensor1D(object):
    """
    Convert vectors to FloatTensors.
    Differently from ToFloatTensor1D, this transform
    is used on testing samples for one-class classification.
    Only applied on X

    """

    def __call__(self, sample):
        X, Y = sample

        return torch.from_numpy(X), Y


class OCToFloatTensor2D(object):
    """ 
    Convert images to FloatTensors.
    Differently from ToFloatTensor2D, this transform
    is used on testing samples for one-class classification.
    Only applied on X
     
    """

    def __call__(self, sample):
        X, Y = sample

        # swap color axis because
        # numpy image: B x H x W x C
        X = X.transpose(2, 0, 1) / 255.

        X = np.float32(X)

        return torch.from_numpy(X), Y


class OCToFloatTensor3D(object):
    """ Convert videos to FloatTensors """
    def __init__(self, normalize=True):
        self._normalize = normalize

    def __call__(self, sample):
        X, Y = sample

        # swap color axis because
        # numpy image: T x H x W x C
        X = X.transpose(3, 0, 1, 2)

        if self._normalize:
            X = X / 255.

        X = np.float32(X)

        return torch.from_numpy(X), Y

class SubtractBackground(object):
    """Removes background from training examples"""

    def __call__(self, sample: tuple):
        X, Y, background = sample

        X = np.float32(X) - background
        Y = np.float32(Y) - background

        return X, Y

class RemoveBackground:

    def __init__(self, threshold: float):
        self.threshold = threshold

    def __call__(self, sample: tuple) -> tuple:
        X, Y, background = sample

        mask = np.uint8(np.sum(np.abs(np.int32(X) - background), axis=-1) > self.threshold)
        mask = np.expand_dims(mask, axis=-1)

        mask = np.stack([binary_dilation(mask_frame, iterations=5) for mask_frame in mask])

        X *= mask
        Y *= mask

        return X, Y


class RemoveBackgroundAndConcatMaskToY:

    def __init__(self, threshold: float):
        self.threshold = threshold

    def __call__(self, sample: tuple) -> tuple:
        X, Y, background = sample

        mask = np.uint8(np.sum(np.abs(np.int32(X) - background), axis=-1) > self.threshold)
        mask = np.expand_dims(mask, axis=-1)

        mask = np.stack([binary_dilation(mask_frame, iterations=5) for mask_frame in mask])

        X *= mask
        Y *= mask

        Y = np.concatenate((Y, mask), axis=-1)

        return X, Y

class ToCrops(object):
    """ Crops input clips """

    def __init__(self, raw_shape, crop_shape):
        self.raw_shape = raw_shape
        self.crop_shape = crop_shape

    def __call__(self, sample: tuple):
        X, Y = sample

        c, t, h, w = self.raw_shape
        cc, tc, hc, wc = self.crop_shape

        crops_X = []
        crops_Y = []

        for k in range(0, t, tc):
            for i in range(0, h, hc // 2):
                for j in range(0, w, wc // 2):
                    if (i + hc <= h) and (j + wc <= w):
                        crops_X.append(X[:, k:k + tc, i:i + hc, j:j + wc])
                        crops_Y.append(Y[:, k:k + tc, i:i + hc, j:j + wc])

        X = torch.stack(crops_X, dim=0)
        Y = torch.stack(crops_Y, dim=0)

        return X, Y

class ToRandomCrops(object):
    """ Crops input clips randomly """

    def __init__(self, raw_shape, crop_shape):
        self.raw_shape = raw_shape
        self.crop_shape = crop_shape

    def __call__(self, sample: tuple):
        X, Y = sample

        c, t, h, w = self.raw_shape
        cc, tc, hc, wc = self.crop_shape

        crops_X = []
        crops_Y = []

        for k in range(0, t, tc):
            for i in range(0, h, hc // 2):
                for j in range(0, w, wc // 2):
                    rd_t = np.random.randint(0, t - tc)
                    rd_h = np.random.randint(0, h - hc)
                    rd_w = np.random.randint(0, w - wc)

                    crops_X.append(X[:, rd_t:rd_t + tc, rd_h:rd_h + hc, rd_w:rd_w + wc])
                    crops_Y.append(Y[:, rd_t:rd_t + tc, rd_h:rd_h + hc, rd_w:rd_w + wc])

        X = torch.stack(crops_X, dim=0)
        Y = torch.stack(crops_Y, dim=0)

        return X, Y

class DropoutNoise(object):
    """ Noises the autoencoder input with dropout """
    def __init__(self, p):
        self._p = p

    def __call__(self, sample):
        X, X = sample

        X_noise = F.dropout(X, p=self._p, training=True)

        return X_noise, X