# -*- coding: utf-8 -*-
""" gutils/numpy_/test/test_images """

import unittest

import numpy as np


from gutils.numpy_.images import ZeroPadding
from gutils.numpy_.test.test_numpy_ import MatrixMixin


class Test_ZeroPadding(MatrixMixin, unittest.TestCase):

    def test_get_padding_values_1(self):
        before_padding, after_padding = ZeroPadding.get_padding_values(6, 3)
        self.assertEqual(before_padding, 1)
        self.assertEqual(after_padding, 2)

    def test_get_padding_values_2(self):
        before_padding, after_padding = ZeroPadding.get_padding_values(8, 2)
        self.assertEqual(before_padding, 3)
        self.assertEqual(after_padding, 3)

    def test_get_padding_values_3(self):
        with self.assertRaises(AssertionError):
            ZeroPadding.get_padding_values(7, 2)

    def test_get_padding_values_4(self):
        with self.assertRaises(AssertionError):
            ZeroPadding.get_padding_values(6, 8)

    def test_ZeroPadding_functor(self):
        zero_padded_img = ZeroPadding(np.full([3, 3], 5), 6, 8)()
        self.assertEqual(zero_padded_img.shape, (6, 8))
