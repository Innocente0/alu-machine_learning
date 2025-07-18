#!/usr/bin/env python3
import numpy as np

def convolve_grayscale_valid(images, kernel):
    """
    Performs a valid convolution on multiple grayscale images.

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
    convolved : numpy.ndarray of shape (m, h - kh + 1, w - kw + 1)
        The batch of convolved images.
    """
    m, h, w = images.shape
    kh, kw = kernel.shape

    # Calculate dimensions of the output
    out_h = h - kh + 1
    out_w = w - kw + 1

    # Initialize the output array
    convolved = np.zeros((m, out_h, out_w))

    # Slide the kernel over each valid position
    for i in range(out_h):            # loop 1: over output rows
        for j in range(out_w):        # loop 2: over output cols
            # Extract the (kh, kw) patch from all m images at once
            patch = images[:, i:i+kh, j:j+kw]
            # Elementwise multiply with the kernel, sum over kh and kw, keep batch dimension
            convolved[:, i, j] = np.sum(patch * kernel, axis=(1, 2))

    return convolved
