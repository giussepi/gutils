# -*- coding: utf-8 -*-
""" gutils/numpy_/images """

import numpy as np


class ZeroPadding:
    """
    Returns a zero-padded 2D np.array

    Usage:
        ZeroPadding(img, target_rows, target_cols)()
    """

    def __init__(self, image, target_rows, target_cols):
        """
        Initialized the object instance

        Args:
            image  (np.ndarray): 2D numpy array
            target_rows   (int): desired number of rows
            target_cols   (int): desired number of cols
        """
        self.image = image
        self.target_rows = target_rows
        self.target_cols = target_cols

    def __call__(self):
        """ functor call """
        return self.get_zero_padded_image()

    @staticmethod
    def get_padding_values(target_length, current_length):
        """
        Calculates the before and after padding values

        Args:
            target_length  (int): objective length
            current_length (int): original length

        Returns:
            before_padding (int), after_padding (int)
        """
        assert isinstance(target_length, int)
        assert isinstance(current_length, int)
        assert target_length % 2 == 0
        assert target_length >= current_length

        padding1 = (target_length - current_length) // 2

        if current_length + 2*padding1 == target_length:
            return padding1, padding1

        return padding1, padding1 + 1

    def get_img_padding_values(self):
        """
        Calculates the padding values for a 2D numpy array

        Returns:
            (top_padding (int), bottom_padding (int)), (left_padding (int), right_padding (int))
        """
        assert isinstance(self.image, np.ndarray)
        assert len(self.image.shape) == 2

        top_rows, bottom_rows = self.get_padding_values(self.target_rows, self.image.shape[0])
        left_cols, right_cols = self.get_padding_values(self.target_cols, self.image.shape[1])

        return (top_rows, bottom_rows), (left_cols, right_cols)

    def get_zero_padded_image(self):
        """
        Returns a zero-padded image considering the target rows and cols provided

        Returns:
            np.array with shape (target_rows, target_cols)
        """
        return np.pad(self.image, self.get_img_padding_values(), 'constant', constant_values=(0,))
