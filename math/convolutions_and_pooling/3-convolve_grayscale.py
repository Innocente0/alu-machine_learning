#!/usr/bin/env python3
"""
Module for performing convolution on a batch of grayscale images
with support for custom padding modes and strides.
"""

import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """
    Performs a convolution on multiple grayscale images.

    Parameters
    ----------
    images : numpy.ndarray of shape (m, h, w)
        m: number of images
        h: height in pixels
        w: width in pixels
    kernel : numpy.ndarray of shape (kh, kw)
        kh: height of the kernel
        kw: width of the kernel
    padding : tuple of (ph, pw), 'same', or 'valid'
        If 'same', applies padding so output has same height/width as input
        If 'valid', no padding
        If a tuple, ph is padding for height and pw for width
    stride : tuple of (sh, sw)
        sh: stride for the height
        sw: stride for the width

    Returns
    -------
    convolved : numpy.ndarray of shape
        (m, out_h, out_w) containing the convolved images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    sh, sw = stride

    # Determine padding
    if isinstance(padding, tuple):
        ph, pw = padding
    elif padding == 'same':
        ph = kh // 2
        pw = kw // 2
    elif padding == 'valid':
        ph = 0
        pw = 0
    else:
        raise ValueError("padding must be 'same', 'valid', or a tuple")

    # Pad images
    images_padded = np.pad(
        images,
        pad_width=((0, 0), (ph, ph), (pw, pw)),
        mode='constant',
        constant_values=0
    )

    # Compute output dimensions
    out_h = (h + 2*ph - kh) // sh + 1
    out_w = (w + 2*pw - kw) // sw + 1

    # Initialize output
    convolved = np.zeros((m, out_h, out_w))

    # Convolution with two loops over output spatial dimensions
    for i in range(out_h):
        for j in range(out_w):
            vs = i * sh
            ve = vs + kh
            hs = j * sw
            he = hs + kw
            patch = images_padded[:, vs:ve, hs:he]
            convolved[:, i, j] = np.sum(patch * kernel, axis=(1, 2))

    return convolved
