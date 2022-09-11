# -*- coding: utf-8 -*-
""" gutils/constants """

import os


TEST_DATASET_PATH = os.path.join('gutils_test_datasets', 'TINY-CT-82')
TEST_IMAGES_PATH = os.path.join(TEST_DATASET_PATH, 'images')
TEST_MASKS_PATH = os.path.join(TEST_DATASET_PATH, 'labels')


class AugmentationType:
    """ Holds the augmentation transforms types  """
    PIXEL_LEVEL = 0  # Pixel-level transforms
    SPATIAL_LEVEL = 1  # Spatial-level transforms
