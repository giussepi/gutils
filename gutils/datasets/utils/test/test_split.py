# -*- coding: utf-8 -*-
""" gutils/datasets/test/test_split """

import unittest

import numpy as np

from gutils.datasets.utils.split import calculate_train_val_percentages, TrainValTestSplit
from gutils.numpy_.numpy_ import LabelMatrixManager


class Test_calculate_train_val_percentages(unittest.TestCase):

    def setUp(self):
        self.train_size = .65
        self.val_size = .10
        self.test_size = .25
        self.num_samples = 1000

    def test_decimal_arguments(self):

        self.assertEqual(
            (self.train_size*self.num_samples, self.val_size*self.num_samples),
            tuple(
                map(
                    lambda x: x*(self.num_samples*(1-self.test_size)),
                    calculate_train_val_percentages(self.val_size, self.test_size)
                )
            )
        )

    def test_integer_arguments(self):
        self.assertEqual(
            (self.train_size*self.num_samples, self.val_size*self.num_samples),
            tuple(
                map(
                    lambda x: x*(self.num_samples*(1-self.test_size)),
                    calculate_train_val_percentages(int(self.val_size*100), int(self.test_size*100))
                )
            )
        )


class Test_TrainValTestSplit(unittest.TestCase):

    def setUp(self):
        self.train_size = .65
        self.val_size = .10
        self.test_size = .25
        self.num_samples = 100

    def test_1D_targets(self):
        samples = np.random.rand(self.num_samples, 3)
        targets = np.array(range(self.num_samples))
        # targets = np.array([0, 1]*5)

        x_train, x_val, x_test, y_train, y_val, y_test = TrainValTestSplit(
            samples, targets, self.val_size, self.test_size)()

        self.assertEqual(x_train.shape[0], 65)
        self.assertEqual(x_val.shape[0], 10)
        self.assertEqual(x_test.shape[0], 25)

        self.assertEqual(y_train.shape[0], 65)
        self.assertEqual(y_val.shape[0], 10)
        self.assertEqual(y_test.shape[0], 25)

    def test_1D_targets_with_stratify(self):
        samples = np.random.rand(self.num_samples, 3)
        targets = np.array([0, 1]*50)

        x_train, x_val, x_test, y_train, y_val, y_test = TrainValTestSplit(
            samples, targets, self.val_size, self.test_size, stratify=targets)()

        self.assertEqual(x_train.shape[0], 65)
        self.assertEqual(x_val.shape[0], 10)
        self.assertEqual(x_test.shape[0], 25)

        self.assertEqual(y_train.shape[0], 65)
        self.assertEqual(y_val.shape[0], 10)
        self.assertEqual(y_test.shape[0], 25)

    def test_1D_targets_with_stratify_error(self):
        samples = np.random.rand(self.num_samples, 3)
        targets = np.array(range(self.num_samples))

        error_msg = ("The least populated class in y has only 1 member, which is too few. "
                     "The minimum number of groups for any class cannot be less than 2.")

        with self.assertRaisesRegex(ValueError, error_msg):
            TrainValTestSplit(
                samples, targets, self.val_size, self.test_size, stratify=targets)()

    def test_2D_targets(self):
        samples = np.random.rand(self.num_samples, 3)
        targets = np.array([0, 1]*50)
        targets = LabelMatrixManager.get_2d_matrix_from_1d_array(targets, 2).T

        x_train, x_val, x_test, y_train, y_val, y_test = TrainValTestSplit(
            samples, targets, self.val_size, self.test_size)()

        self.assertEqual(x_train.shape[0], 65)
        self.assertEqual(x_val.shape[0], 10)
        self.assertEqual(x_test.shape[0], 25)

        self.assertEqual(y_train.shape[0], 65)
        self.assertEqual(y_val.shape[0], 10)
        self.assertEqual(y_test.shape[0], 25)

    def test_2D_targets_with_stratify(self):
        samples = np.random.rand(self.num_samples, 3)
        targets = np.array([0, 1]*50)
        targets = LabelMatrixManager.get_2d_matrix_from_1d_array(targets, 2).T

        x_train, x_val, x_test, y_train, y_val, y_test = TrainValTestSplit(
            samples, targets, self.val_size, self.test_size, stratify=targets)()

        self.assertEqual(x_train.shape[0], 65)
        self.assertEqual(x_val.shape[0], 10)
        self.assertEqual(x_test.shape[0], 25)

        self.assertEqual(y_train.shape[0], 65)
        self.assertEqual(y_val.shape[0], 10)
        self.assertEqual(y_test.shape[0], 25)

    def test_2D_targets_with_stratify_error(self):
        samples = np.random.rand(self.num_samples, 3)
        targets = np.array(range(self.num_samples))
        targets = LabelMatrixManager.get_2d_matrix_from_1d_array(targets, self.num_samples).T

        error_msg = ("The least populated class in y has only 1 member, which is too few. "
                     "The minimum number of groups for any class cannot be less than 2.")

        with self.assertRaisesRegex(ValueError, error_msg):
            TrainValTestSplit(samples, targets, self.val_size, self.test_size, stratify=targets)()
