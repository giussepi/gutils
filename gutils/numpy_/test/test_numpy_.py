# -*- coding: utf-8 -*-
""" gutils/numpy_/test/test_numpy_ """

import unittest

import numpy as np
from scipy import linalg

from gutils.numpy_.numpy_ import colnorms_squared_new, normcols, LabelMatrixManager, \
    scale_using_general_min_max_values, split_numpy_array


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
            LabelMatrixManager.get_2d_matrix_from_1d_array(self.labels_1d, 3),
            self.labels_2d
        ))

    def test_get_2d_matrix_from_1d_array_with_empty_labels(self):
        labels_1d = np.array([0, 0, 3, 2, 3, 0, 5])
        labels_2d = np.array([
            [1, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 1, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1],
        ])

        self.assertTrue(np.array_equal(
            LabelMatrixManager.get_2d_matrix_from_1d_array(labels_1d, 6),
            labels_2d
        ))

    def test_get_1d_array_from_2d_matrix(self):
        self.assertTrue(np.array_equal(
            LabelMatrixManager.get_1d_array_from_2d_matrix(self.labels_2d), self.labels_1d))


class Test_scale_using_general_min_max_values(unittest.TestCase):

    def setUp(self):
        self.vector = np.array([1, 2, 3])
        self.matrix = np.array([[4, 5, 6], [7, 8, 9]])

    def test_vector_default_args(self):
        scaled_vector = (self.vector - self.vector.min()) / (self.vector.max() - self.vector.min())
        self.assertTrue(np.array_equal(
            scaled_vector,
            scale_using_general_min_max_values(self.vector)
        ))

    def test_vector_dtype(self):
        scaled_vector = (self.vector - self.vector.min()) / (self.vector.max() - self.vector.min())
        self.assertEqual(
            np.float32,
            scale_using_general_min_max_values(self.vector, dtype=np.float32).dtype
        )

    def test_vector_range(self):
        feats_range = [10, 20]
        scaled_vector = (self.vector - self.vector.min()) / (self.vector.max() - self.vector.min())
        scaled_vector *= feats_range[1] - feats_range[0]
        scaled_vector += feats_range[0]
        self.assertTrue(np.array_equal(
            scaled_vector,
            scale_using_general_min_max_values(self.vector, feats_range=feats_range)
        ))

    def test_vector_min_max_range(self):
        min_val = -10
        max_val = 10
        feats_range = [10, 20]
        scaled_vector = (self.vector - min_val) / (max_val - min_val)
        scaled_vector *= feats_range[1] - feats_range[0]
        scaled_vector += feats_range[0]
        self.assertTrue(np.array_equal(
            scaled_vector,
            scale_using_general_min_max_values(self.vector, min_val, max_val, feats_range)
        ))

    def test_matrix_default_args(self):
        scaled_matrix = (self.matrix - self.matrix.min()) / (self.matrix.max() - self.matrix.min())
        self.assertTrue(np.array_equal(
            scaled_matrix,
            scale_using_general_min_max_values(self.matrix)
        ))

    def test_matrix_dtype(self):
        scaled_matrix = (self.matrix - self.matrix.min()) / (self.matrix.max() - self.matrix.min())
        self.assertEqual(
            np.float64,
            scale_using_general_min_max_values(self.matrix, dtype=np.float64).dtype
        )

    def test_matrix_range(self):
        feats_range = [10, 20]
        scaled_matrix = (self.matrix - self.matrix.min()) / (self.matrix.max() - self.matrix.min())
        scaled_matrix *= feats_range[1] - feats_range[0]
        scaled_matrix += feats_range[0]
        self.assertTrue(np.array_equal(
            scaled_matrix,
            scale_using_general_min_max_values(self.matrix, feats_range=feats_range)
        ))

    def test_matrix_min_max_range(self):
        min_val = -10
        max_val = 10
        feats_range = [10, 20]
        scaled_matrix = (self.matrix - min_val) / (max_val - min_val)
        scaled_matrix *= feats_range[1] - feats_range[0]
        scaled_matrix += feats_range[0]
        self.assertTrue(np.array_equal(
            scaled_matrix,
            scale_using_general_min_max_values(self.matrix, min_val, max_val, feats_range)
        ))


class Test_split_numpy_array(unittest.TestCase):

    def test_1D_array(self):
        array = np.array(np.random.rand(10))
        bit1, bit2 = split_numpy_array(array, .3, 0, False)
        self.assertTrue((3, 7), (bit1.shape[0], bit2.shape[0]))
        self.assertTrue(np.array_equal(bit1, array[:3]))
        self.assertTrue(np.array_equal(bit2, array[3:]))

    def test_1D_array_with_shuffle(self):
        array = np.array(np.random.rand(10))
        bit1, bit2 = split_numpy_array(array, .3, 0, True)
        self.assertTrue((3, 7), (bit1.shape[0], bit2.shape[0]))
        self.assertFalse(np.array_equal(bit1, array[:3]))
        self.assertFalse(np.array_equal(bit2, array[3:]))

    def test_2D_array_axis_0(self):
        axis = 0
        array_2D = np.random.rand(20, 10)
        bit1, bit2 = split_numpy_array(array_2D, .3, axis, False)
        self.assertTrue((6, 14), (bit1.shape[axis], bit2.shape[axis]))
        self.assertEqual((6, 10), bit1.shape)
        self.assertEqual((14, 10), bit2.shape)
        self.assertTrue(np.array_equal(bit1, array_2D[:6, :]))
        self.assertTrue(np.array_equal(bit2, array_2D[6:, :]))

    def test_2D_array_axis_0_with_shuffle(self):
        axis = 0
        array_2D = np.random.rand(20, 10)
        bit1, bit2 = split_numpy_array(array_2D, .3, axis, True)
        self.assertTrue((6, 14), (bit1.shape[axis], bit2.shape[axis]))
        self.assertEqual((6, 10), bit1.shape)
        self.assertEqual((14, 10), bit2.shape)
        self.assertFalse(np.array_equal(bit1, array_2D[:6, :]))
        self.assertFalse(np.array_equal(bit2, array_2D[6:, :]))

    def test_2D_array_axis_1(self):
        axis = 1
        array_2D = np.random.rand(20, 10)
        bit1, bit2 = split_numpy_array(array_2D, .3, axis, False)
        self.assertTrue((3, 7), (bit1.shape[axis], bit2.shape[axis]))
        self.assertEqual((20, 3), bit1.shape)
        self.assertEqual((20, 7), bit2.shape)
        self.assertTrue(np.array_equal(bit1, array_2D[:, :3]))
        self.assertTrue(np.array_equal(bit2, array_2D[:, 3:]))

    def test_2D_array_axis_1_with_shuffle(self):
        axis = 1
        array_2D = np.random.rand(20, 10)
        bit1, bit2 = split_numpy_array(array_2D, .3, axis, True)
        self.assertTrue((3, 7), (bit1.shape[axis], bit2.shape[axis]))
        self.assertEqual((20, 3), bit1.shape)
        self.assertEqual((20, 7), bit2.shape)
        self.assertFalse(np.array_equal(bit1, array_2D[:, :3]))
        self.assertFalse(np.array_equal(bit2, array_2D[:, 3:]))


if __name__ == '__main__':
    unittest.main()
