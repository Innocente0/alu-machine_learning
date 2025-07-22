#!/usr/bin/env python3
"""
Module for performing same convolution on a batch of grayscale images.
"""

import numpy as np


def convolve_grayscale_same(images, kernel):
    """
    Performs a same convolution on multiple grayscale images.

    Parameters
    ----------
    images : numpy.ndarray of shape (m, h, w)
        m: number of images
        h: height in pixels
        w: width in pixels
    kernel : numpy.ndarray of shape (kh, kw)
        kh: height of the kernel
        kw: width of the kernel

    Returns
    -------
    convolved : numpy.ndarray of shape (m, h, w)
        The batch of convolved images with same padding.
    """
    m, h, w = images.shape
    kh, kw = kernel.shape

    # Compute padding sizes
    pad_h = kh // 2
    pad_w = kw // 2

    # Pad all images
    images_padded = np.pad(
        images,
        pad_width=((0, 0), (pad_h, pad_h), (pad_w, pad_w)),
        mode='constant',
        constant_values=0
    )

    # Initialize output
    convolved = np.zeros((m, h, w))

    # Perform convolution
    for i in range(h):
        for j in range(w):
            patch = images_padded[:, i:i+kh, j:j+kw]
            convolved[:, i, j] = np.sum(patch * kernel, axis=(1, 2))

    return convolved
