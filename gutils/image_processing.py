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
    # TODO: write its tests
    stride = patch_size-patch_overlapping
    last_coord = None

    for i in range(0, dim_axis, stride):
        if i + patch_size <= dim_axis:
            last_coord = i
            yield last_coord

    # if there is one missing patch coordinate, then return it
    if last_coord + patch_size < dim_axis:
        yield dim_axis-patch_size


def get_patches(img, patch_size=16, patch_overlapping=0):
    """
    Calculates and returns an iterator over the patches coordinates following the patch_size
    and patch_overlapping provided

    Args:
        img        (np.ndarray): image loaded using numpy
        patch_size        (int): size of the patchw
        patch_overlapping (int): overlap of patches

    Returns:
        iterator
    """
    # TODO: find out how it should work when using a 3-channel image (colored one)
    # TODO: write its tests
    assert isinstance(img, np.ndarray)
    assert isinstance(patch_size, int)
    assert isinstance(patch_overlapping, int)

    for x in get_slices_coords(img.shape[0], patch_size, patch_overlapping):
        for y in get_slices_coords(img.shape[1], patch_size, patch_overlapping):
            yield img[x:x+patch_size, y:y+patch_size]
            # yield([(x, x+patch_size), (y, y+patch_size)])
