# -*- coding: utf-8 -*-
""" utils/test/test_numpy_ """

import unittest

import numpy as np
from scipy import linalg

from g_utils.numpy_ import colnorms_squared_new, normcols


class MatrixMixin:

    def setUp(self):
        self.matrix = np.random.rand(3, 3)


class Test_colnorms_squared_new(MatrixMixin, unittest.TestCase):

    def test_function(self):
        self.assertTrue(
            np.array_equal(np.sum(self.matrix**2, axis=0), colnorms_squared_new(self.matrix))
        )


class Test_normcols(MatrixMixin, unittest.TestCase):

    def test_normcols(self):
        self.assertTrue(np.array_equal(
            self.matrix/linalg.norm(self.matrix, axis=0), normcols(self.matrix)))


if __name__ == '__main__':
    unittest.main()
