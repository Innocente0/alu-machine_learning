#!/usr/bin/env python3
"""
Module for performing convolution on images with multiple kernels.
"""

import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """
    Performs a convolution on multiple images using multiple kernels.

    Parameters
    ----------
    images : numpy.ndarray of shape (m, h, w, c)
        m: number of images
        h: height in pixels
        w: width in pixels
        c: number of channels
    kernels : numpy.ndarray of shape (kh, kw, c, nc)
        kh: height of each kernel
        kw: width of each kernel
        c: number of channels (must match images)
        nc: number of kernels (output channels)
    padding : tuple of (ph, pw), 'same', or 'valid'
        If 'same', applies padding so output has same height/width as input.
        If 'valid', no padding.
        If tuple, ph is padding for height and pw is padding for width.
    stride : tuple of (sh, sw)
        sh: stride for the height
        sw: stride for the width

    Returns
    -------
    numpy.ndarray of shape (m, out_h, out_w, nc)
        The batch of convolved images.
    """
    m, h, w, c = images.shape
    kh, kw, kc, nc = kernels.shape
    sh, sw = stride

    if kc != c:
        raise ValueError("Kernel and image channel counts must match")

    # Determine padding amounts
    if isinstance(padding, tuple):
        ph, pw = padding
    elif padding == 'same':
        ph = kh // 2
        pw = kw // 2
    elif padding == 'valid':
        ph = pw = 0
    else:
        raise ValueError("padding must be 'same', 'valid', or a tuple")

    # Pad images
    images_padded = np.pad(
        images,
        ((0, 0), (ph, ph), (pw, pw), (0, 0)),
        mode='constant',
        constant_values=0
    )

    # Compute output dimensions
    out_h = (h + 2*ph - kh) // sh + 1
    out_w = (w + 2*pw - kw) // sw + 1

    # Allocate output
    convolved = np.zeros((m, out_h, out_w, nc))

    # Convolve using exactly three loops
    for i in range(out_h):
        for j in range(out_w):
            vs = i * sh
            hs = j * sw
            patch = images_padded[:, vs:vs+kh, hs:hs+kw, :]
            for k in range(nc):
                convolved[:, i, j, k] = np.sum(
                    patch * kernels[:, :, :, k],
                    axis=(1, 2, 3)
                )

    return convolved
