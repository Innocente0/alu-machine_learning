#!/usr/bin/env python3
"""
Module for performing pooling on a batch of images.
"""

import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """
    Performs pooling on images.

    Parameters
    ----------
    images : numpy.ndarray of shape (m, h, w, c)
        m: number of images
        h: height in pixels
        w: width in pixels
        c: number of channels
    kernel_shape : tuple of (kh, kw)
        kh: height of the pooling kernel
        kw: width of the pooling kernel
    stride : tuple of (sh, sw)
        sh: stride for the height
        sw: stride for the width
    mode : str, optional
        'max' for max pooling, 'avg' for average pooling

    Returns
    -------
    numpy.ndarray of shape (m, out_h, out_w, c)
        The pooled images.
    """
    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride

    # Calculate output dimensions
    out_h = (h - kh) // sh + 1
    out_w = (w - kw) // sw + 1

    # Initialize output
    pooled = np.zeros((m, out_h, out_w, c))

    # Perform pooling with two loops
    for i in range(out_h):
        for j in range(out_w):
            patch = images[:, i*sh:i*sh+kh, j*sw:j*sw+kw, :]
            if mode == 'max':
                pooled[:, i, j, :] = np.max(patch, axis=(1, 2))
            elif mode == 'avg':
                pooled[:, i, j, :] = np.mean(patch, axis=(1, 2))
            else:
                raise ValueError("mode must be 'max' or 'avg'")

    return pooled
