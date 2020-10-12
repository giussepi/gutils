# -*- coding: utf-8 -*-
""" gutils/test/test_image_processing """

import unittest

import numpy as np

from gutils.image_processing import get_slices_coords, get_patches


class Test_get_slices_coords(unittest.TestCase):

    def test_exact_slices(self):
        self.assertEqual(list(get_slices_coords(20, 5, 0)), [0, 5, 10, 15])

    def test_exact_slices_with_overlapping(self):
        self.assertEqual(list(get_slices_coords(20, 5, 1)), [0, 4, 8, 12, 15])

    def test_non_exact_slices(self):
        self.assertEqual(list(get_slices_coords(21, 5, 0)), [0, 5, 10, 15, 16])

    def test_non_exact_slices_with_overlapping(self):
        self.assertEqual(list(get_slices_coords(21, 5, 2)), [0, 3, 6, 9, 12, 15, 16])


class Test_get_patches(unittest.TestCase):

    def setUp(self):
        self.image = np.array([np.arange(20) for _ in range(20)])

    def test_exact_patches(self):
        width = height = 5

        patches = iter(get_patches(self.image, patch_width=5))

        for x in [0, 5, 10, 15]:
            for y in [0, 5, 10, 15]:
                self.assertTrue(np.array_equal(
                    self.image[x:x+width, y:y+height], next(patches)
                ))

    def test_exact_slices_with_overlapping(self):
        width = height = 5

        patches = iter(get_patches(self.image, patch_width=5, patch_overlapping=1))

        for x in [0, 4, 8, 12, 15]:
            for y in [0, 4, 8, 12, 15]:
                self.assertTrue(np.array_equal(
                    self.image[x:x+width, y:y+height], next(patches)
                ))

    def test_different_height_and_width(self):
        width = 4
        height = 5

        patches = iter(get_patches(self.image, patch_width=width, patch_height=height))

        for x in get_slices_coords(20, width, 0):
            for y in get_slices_coords(20, height, 0):
                self.assertTrue(np.array_equal(
                    self.image[x:x+width, y:y+height], next(patches)
                ))

    def test_different_height_and_width_with_overlapping(self):
        width = 4
        height = 5
        overlap = 2

        patches = iter(get_patches(
            self.image, patch_width=width, patch_height=height, patch_overlapping=overlap))

        for x in get_slices_coords(20, width, overlap):
            for y in get_slices_coords(20, height, overlap):
                self.assertTrue(np.array_equal(
                    self.image[x:x+width, y:y+height], next(patches)
                ))

    def test_non_exact_patches(self):
        width = height = 5
        self.image = np.random.rand(21, 21)

        patches = iter(get_patches(self.image, patch_width=5))

        for x in [0, 5, 10, 15, 16]:
            for y in [0, 5, 10, 15, 16]:
                self.assertTrue(np.array_equal(
                    self.image[x:x+width, y:y+height], next(patches)
                ))

    def test_non_exact_patches_with_overlapping(self):
        width = height = 5
        overlap = 2
        self.image = np.random.rand(21, 21)

        patches = iter(get_patches(self.image, patch_width=5, patch_overlapping=overlap))

        for x in [0, 3, 6, 9, 12, 15, 16]:
            for y in [0, 3, 6, 9, 12, 15, 16]:
                self.assertTrue(np.array_equal(
                    self.image[x:x+width, y:y+height], next(patches)
                ))

    def test_non_exact_patches_different_height_and_width_with_overlapping(self):
        width = 4
        height = 5
        overlap = 2

        self.image = np.random.rand(21, 23)

        patches = iter(get_patches(
            self.image, patch_width=width, patch_height=height, patch_overlapping=overlap))

        for x in get_slices_coords(self.image.shape[0], width, overlap):
            for y in get_slices_coords(self.image.shape[1], height, overlap):
                self.assertTrue(np.array_equal(
                    self.image[x:x+width, y:y+height], next(patches)
                ))


if __name__ == '__main__':
    unittest.main()
