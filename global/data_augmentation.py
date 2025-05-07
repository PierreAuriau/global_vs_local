# !/usr/bin/env python
# -*- coding: utf-8 -*-


# Import
import os
import logging
import numpy as np
from scipy.ndimage import rotate, gaussian_filter, shift
import torch
import torch.nn as nn
import numbers

from config import Config

logger = logging.getLogger()
config = Config()

class ToTensor(nn.Module):
    def forward(self, arr):
        arr = np.expand_dims(arr, axis=0)
        return torch.from_numpy(arr)


class ToArray(nn.Module):
    def forward(self, tensor):
        tensor = tensor.squeeze()
        return np.asarray(tensor)

def interval(obj, lower=None):
    """ Listify an object.

    Parameters
    ----------
    obj: 2-uplet or number
        the object used to build the interval.
    lower: number, default None
        the lower bound of the interval. If not specified, a symetric
        interval is generated.

    Returns
    -------
    interval: 2-uplet
        an interval.
    """
    if isinstance(obj, numbers.Number):
        if obj < 0:
            raise ValueError("Specified interval value must be positive.")
        if lower is None:
            lower = -obj
        return (lower, obj)
    if len(obj) != 2:
        raise ValueError("Interval must be specified with 2 values.")
    min_val, max_val = obj
    if min_val > max_val:
        raise ValueError("Wrong interval boudaries.")

    return tuple(obj)   


class Normalize(object):
    def __init__(self, mean=0.0, std=1.0, eps=1e-8):
        self.mean=mean
        self.std=std
        self.eps=eps

    def __call__(self, arr):
        return self.std * (arr - np.mean(arr))/(np.std(arr) + self.eps) + self.mean
    
    def __str__(self):
        return "Normalize"


class Rotation(object):

    def __init__(self, angles, axes=(0, 2), reshape=False, **kwargs):
        if isinstance(angles, (int, float)):
            self.angles = [-angles, angles]
        elif isinstance(angles, (list, tuple)):
            assert (len(angles) == 2 and angles[0] < angles[1]), print(f"Wrong angles format {angles}")
            self.angles = angles
        else:
            raise ValueError("Unkown angles type: {}".format(type(angles)))
        if isinstance(axes, tuple):
            self.axes = [axes]
        elif isinstance(axes, list):
            self.axes = axes
        else:
            logger.warning('Rotations: rotation plane will be determined randomly')
            self.axes = [tuple(np.random.choice(3, 2, replace=False))]
        self.reshape = reshape
        self.rotate_kwargs = kwargs

    def __call__(self, arr):
        return self._apply_random_rotation(arr)

    def _apply_random_rotation(self, arr):
        angles = [np.float16(np.random.uniform(self.angles[0], self.angles[1]))
                  for _ in range(len(self.axes))]
        for ax, angle in zip(self.axes, angles):
            arr = rotate(arr, angle, axes=ax, reshape=self.reshape, **self.rotate_kwargs)
        return arr

    def __str__(self):
        return f"Rotation(angles={self.angles}, axes={self.axes})"


class Cutout(object):
    def __init__(self, patch_size, random_size=False, localization="random", **kwargs):
        self.patch_size = patch_size
        self.random_size = random_size
        if localization in ["random", "on_data"] or isinstance(localization, (tuple, list)):
            self.localization = localization
        else:
            logger.warning("Cutout : localization is set to random")
            self.localization = "random"
        self.min_size = kwargs.get("min_size", 0)
        self.value = kwargs.get("value", 0)
        self.image_shape = kwargs.get("image_shape", None)
        if self.image_shape is not None:
            self.patch_size = self._get_patch_size(self.patch_size,
                                                   self.image_shape)
            self.min_size = self._get_patch_size(self.min_size,
                                                 self.image_shape)
        self.shuffle = kwargs.get("shuffle", False)
        if self.shuffle:
            logger.warning("Cutout: shuffle pixels, ignoring value")

    def __call__(self, arr):
        if self.image_shape is None:
            arr_shape = arr.shape
            self.patch_size = self._get_patch_size(self.patch_size,
                                                   arr_shape)
            self.min_size = self._get_patch_size(self.min_size,
                                                 arr_shape)
        return self._apply_cutout(arr)

    def _apply_cutout(self, arr):
        image_shape = arr.shape
        if self.localization == "on_data":
            nonzero_voxels = np.nonzero(arr)
            index = np.random.randint(0, len(nonzero_voxels[0]))
            localization = np.array([nonzero_voxels[i][index] for i in range(len(nonzero_voxels))])
        elif isinstance(self.localization, (tuple, list)):
            assert len(self.localization) == len(image_shape), f"Cutout : wrong localization shape"
            localization = self.localization
        else:
            localization = None
        indexes = []
        for ndim, shape in enumerate(image_shape):
            if self.random_size:
                size = np.random.randint(self.min_size[ndim], self.patch_size[ndim])
            else:
                size = self.patch_size[ndim]
            if localization is not None:
                delta_before = max(localization[ndim] - size // 2, 0)
            else:
                delta_before = np.random.randint(0, shape - size + 1)
            indexes.append(slice(delta_before, delta_before + size))
        if self.shuffle:
            sh = [s.stop - s.start for s in indexes]
            arr[tuple(indexes)] = np.random.shuffle(arr[tuple(indexes)].flat).reshape(sh)
        else:
            arr[tuple(indexes)] = self.value
        return arr

    @staticmethod
    def _get_patch_size(patch_size, image_shape):
        if isinstance(patch_size, int):
            size = [patch_size for _ in range(len(image_shape))]
        elif isinstance(patch_size, float):
            size = [int(patch_size*s) for s in image_shape]
        else:
            size = patch_size
        assert len(size) == len(image_shape), f"Incorrect patch dimension. {len(size)}/{len(image_shape)}"
        for ndim in range(len(image_shape)):
            if size[ndim] > image_shape[ndim] or size[ndim] < 0:
                size[ndim] = image_shape[ndim]
        return size

    def __str__(self):
        return f"Cutout(patch_size={self.patch_size}, random_size={self.random_size}, " \
               f"localization={self.localization})"

    
class Flip(object):
    """ Apply a random mirror flip."""
    def __init__(self, axis=None):
        '''
        :param axis: int, default None
            apply flip on the specified axis. If not specified, randomize the
            flip axis.
        '''
        self.axis = axis

    def __call__(self, arr):
        if self.axis is None:
            axis = np.random.randint(low=0, high=arr.ndim, size=1)[0]
        return np.flip(arr, axis=(self.axis or axis))
    
    def __str__(self):
        return f"Flip(axis={self.axis})"


class Blur(object):
    def __init__(self, snr=None, sigma=None):
        """ Add random blur using a Gaussian filter.
            Parameters
            ----------
            snr: float, default None
                the desired signal-to noise ratio used to infer the standard deviation
                for the noise distribution.
            sigma: float or 2-uplet
                the standard deviation for Gaussian kernel.
        """
        if snr is None and sigma is None:
            raise ValueError("You must define either the desired signal-to noise "
                             "ratio or the standard deviation for the noise "
                             "distribution.")
        self.snr = snr
        self.sigma = sigma

    def __call__(self, arr):
        sigma = self.sigma
        if self.snr is not None:
            s0 = np.std(arr)
            sigma = s0 / self.snr
        sigma = interval(sigma, lower=0)
        sigma_random = np.random.uniform(low=sigma[0], high=sigma[1], size=1)[0]
        return gaussian_filter(arr, sigma_random)
    
    def __str__(self):
        return f"Blur(snr={self.snr}, sigma={self.sigma})"
    

class Shift(object):
    """ Translate the image of a number of voxels.
    """
    def __init__(self, nb_voxels, random):
        self.random = random
        self.nb_voxels = nb_voxels

    def __call__(self, arr):
        ndim = arr.ndim
        if self.random:
            translation = np.random.randint(-self.nb_voxels, self.nb_voxels+1, size=ndim)
        else:
            if isinstance(self.nb_voxels, int):
                translation = [self.nb_voxels for _ in range(ndim)]
            else:
                translation = self.nb_voxels
        transformed = shift(arr, translation, order=0, mode='constant', cval=0.0, prefilter=False)
        return transformed


class Occlusion(object):

    def __init__(self, area, background=0):
        self.mask = np.load(os.path.join(config.path_to_masks, f"{area}.npy"))
        self.background = background

    def __call__(self, arr):
        arr[self.mask] = self.background
        return arr


if __name__ == "__main__":
    arr = np.load(os.path.join(config.path2data, "skeleton_sub-1000606.npy"))
    print(arr.shape)
    cutout = Cutout(patch_size=0.4, random_size=True, localization="on_data", min_size=0.1)
    
    view_1 = cutout(arr)
    view_2 = cutout(arr)
    view_3 = cutout(arr)
    print(np.all(view_1  == view_2))
    print(np.all(view_2 == view_3))