# -*- coding: utf-8 -*-
""" gutils/numpy_/test/test_numpy_ """

import unittest

import numpy as np
from scipy import linalg

from gutils.numpy_.numpy_ import colnorms_squared_new, normcols, LabelMatrixManager


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


class Test_LabelMatrixManager(unittest.TestCase):

    def setUp(self):
        self.labels_1d = np.array([0, 1, 0, 2, 2])
        self.labels_2d = np.array([[1, 0, 1, 0, 0], [0, 1, 0, 0, 0], [0, 0, 0, 1, 1]])

    def test_get_2d_matrix_from_1d_array(self):
        self.assertTrue(np.array_equal(
            LabelMatrixManager.get_2d_matrix_from_1d_array(self.labels_1d),
            self.labels_2d
        ))

    def test_get_2d_matrix_from_1d_array_with_sorted_labels_args(self):
        sorted_labels = [2, 0, 1]
        output_label_matrix = np.array([[0, 0, 0, 1, 1], [1, 0, 1, 0, 0], [0, 1, 0, 0, 0]])

        self.assertFalse(np.array_equal(
            LabelMatrixManager.get_2d_matrix_from_1d_array(self.labels_1d),
            output_label_matrix
        ))
        self.assertTrue(np.array_equal(
            LabelMatrixManager.get_2d_matrix_from_1d_array(self.labels_1d, sorted_labels),
            output_label_matrix
        ))

    def test_get_1d_array_from_2d_matrix(self):
        self.assertTrue(np.array_equal(
            LabelMatrixManager.get_1d_array_from_2d_matrix(self.labels_2d), self.labels_1d))


if __name__ == '__main__':
    unittest.main()
