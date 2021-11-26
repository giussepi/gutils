# -*- coding: utf-8 -*-
""" gutils/images/files """

from imutils.paths import image_types, list_files


def list_images(base_path, extra_types=None):
    """
    list images from a directory

    Args:
        base_path (str): directory
        extra_type (list): Extra image extension to list

    Returns:
        list of images <generator>
    """
    if extra_types is not None:
        assert isinstance(extra_types, list)
    else:
        extra_types = list()

    custom_image_types = tuple(list(image_types) + extra_types)

    return list_files(base_path, validExts=custom_image_types)
