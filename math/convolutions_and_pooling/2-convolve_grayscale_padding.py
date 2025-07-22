#!/usr/bin/env python3
"""
Module for performing convolution on grayscale images
with custom padding.
"""

import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """
    Performs a convolution on multiple grayscale images
    with custom padding.

    Parameters
    ----------
    images : numpy.ndarray of shape (m, h, w)
        m: number of images
        h: height in pixels
        w: width in pixels
    kernel : numpy.ndarray of shape (kh, kw)
        kh: height of the kernel
        kw: width of the kernel
    padding : tuple of (ph, pw)
        ph: padding for the height
        pw: padding for the width

    Returns
    -------
    convolved : numpy.ndarray of shape
        (m, h + 2*ph - kh + 1, w + 2*pw - kw + 1)
        The batch of convolved images.
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    ph, pw = padding

    # Pad images with zeros on height and width
    images_padded = np.pad(
        images,
        pad_width=((0, 0), (ph, ph), (pw, pw)),
        mode='constant',
        constant_values=0
    )

    # Output dimensions after convolution
    out_h = h + 2 * ph - kh + 1
    out_w = w + 2 * pw - kw + 1

    # Prepare output array
    convolved = np.zeros((m, out_h, out_w))

    # Perform convolution using two loops
    for i in range(out_h):
        for j in range(out_w):
            patch = images_padded[:, i:i+kh, j:j+kw]
            convolved[:, i, j] = np.sum(patch * kernel, axis=(1, 2))

    return convolved
