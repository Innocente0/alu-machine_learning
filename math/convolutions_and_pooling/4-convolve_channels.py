#!/usr/bin/env python3
"""
Module for performing convolution on images with channels.
"""

import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """
    Performs a convolution on multiple images with channels.

    Parameters
    ----------
    images : numpy.ndarray of shape (m, h, w, c)
        m: number of images
        h: height in pixels
        w: width in pixels
        c: number of channels
    kernel : numpy.ndarray of shape (kh, kw, c)
        kh: height of the kernel
        kw: width of the kernel
        c: number of channels (must match images)
    padding : tuple of (ph, pw), 'same', or 'valid'
        If 'same', output has same height/width as input.
        If 'valid', no padding.
        If tuple, ph is padding for height and pw for width.
    stride : tuple of (sh, sw)
        sh: stride for the height
        sw: stride for the width

    Returns
    -------
    numpy.ndarray of shape (m, out_h, out_w)
        The batch of convolved images.
    """
    m, h, w, c = images.shape
    kh, kw, kc = kernel.shape
    sh, sw = stride

    if kc != c:
        raise ValueError("Kernel and image channel counts must match")

    # Determine padding amounts
    if isinstance(padding, tuple):
        ph, pw = padding
    elif padding == 'same':
        ph = (kh - 1) // 2
        pw = (kw - 1) // 2
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
    out_h = (h + 2 * ph - kh) // sh + 1
    out_w = (w + 2 * pw - kw) // sw + 1

    # Allocate output
    convolved = np.zeros((m, out_h, out_w))

    # Convolution with exactly two loops
    for i in range(out_h):
        for j in range(out_w):
            vs = i * sh
            hs = j * sw
            patch = images_padded[:, vs:vs + kh, hs:hs + kw, :]
            # Sum over height, width, and channel dimensions
            convolved[:, i, j] = np.sum(patch * kernel, axis=(1, 2, 3))

    return convolved
