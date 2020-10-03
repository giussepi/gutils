# -*- coding: utf-8 -*-
""" gutils/test/test_numpy_ """

import unittest

import numpy as np
from scipy import linalg

from gutils.numpy_ import colnorms_squared_new, normcols, format_label_matrix


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


class Test_format_label_matrix(unittest.TestCase):

    def test_format_label_matrix(self):
        input_label_matrix = np.array([0, 1, 0, 2, 2])
        output_label_matrix = np.array([[1, 0, 1, 0, 0], [0, 1, 0, 0, 0], [0, 0, 0, 1, 1]])
        self.assertTrue(np.array_equal(
            format_label_matrix(input_label_matrix), output_label_matrix
        ))

    def test_sorted_labels_args(self):
        input_label_matrix = np.array([0, 1, 0, 2, 2])
        sorted_labels = [2, 0, 1]
        output_label_matrix = np.array([[0, 0, 0, 1, 1], [1, 0, 1, 0, 0], [0, 1, 0, 0, 0]])
        self.assertFalse(np.array_equal(
            format_label_matrix(input_label_matrix), output_label_matrix))
        self.assertTrue(np.array_equal(
            format_label_matrix(input_label_matrix, sorted_labels),
            output_label_matrix
        ))


if __name__ == '__main__':
    unittest.main()
