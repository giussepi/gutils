# -*- coding: utf-8 -*-
""" gutils/image_processing """

import numpy as np


def get_slices_coords(dim_axis, patch_size=16, patch_overlapping=0):
    """
    Generator that returns an iterable set of coordinates positions following the patch size
    and overlapping specified

    Args:
        dim_axis          (int): dimensionality of the axis
        patch_size        (int): patch size
        patch_overlapping (int): patch overlapping

    Returns:
        iterator
    """
    stride = patch_size-patch_overlapping
    last_coord = None

    for i in range(0, dim_axis, stride):
        if i + patch_size <= dim_axis:
            last_coord = i
            yield last_coord

    # if there is one missing patch coordinate, then return it
    if last_coord + patch_size < dim_axis:
        yield dim_axis-patch_size


def get_patches(img, patch_width=16, patch_height=0, patch_overlapping=0):
    """
    Calculates and returns an iterator over the patches coordinates following the patch width,
    height, and patch_overlapping provided. If no patch_height is provided then it set to
    path_width by default.

    Args:
        img        (np.ndarray): image loaded using numpy with shape (height, weight)
        patch_width       (int): width of the patch
        patch_height      (int): height of the patch
        patch_overlapping (int): overlap of patches

    Returns:
        iterator
    """
    # TODO: find out how it should work when using a 3-channel image (colored one)
    assert isinstance(img, np.ndarray)
    assert isinstance(patch_width, int)
    assert patch_width > 0
    assert isinstance(patch_height, int)
    assert patch_height >= 0
    assert isinstance(patch_overlapping, int)

    if patch_height == 0:
        patch_height = patch_width

    for x in get_slices_coords(img.shape[1], patch_width, patch_overlapping):
        for y in get_slices_coords(img.shape[0], patch_height, patch_overlapping):
            yield img[y:y+patch_height, x:x+patch_width]
