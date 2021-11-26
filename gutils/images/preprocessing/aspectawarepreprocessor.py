# -*- coding: utf-8 -*-
""" gutils/images/preprocessing/aspectawarepreprocessor """

import cv2
import imutils
import numpy as np


class AspectAwarePreprocessor:
    """
    Resizes an image to a fixed width and height considering the image
    aspect ratio. (No image distortion)

    Usage:
        preprocessor = AspectAwarePreprocessor(100, 200)
        resized_img = preprocessor.preprocess(image)

        # or just use the functor call
        resized_img = AspectAwarePreprocessor(100, 200)(image)
    """

    def __init__(self, width, height, inter=cv2.INTER_AREA):
        """ Initialized the instance """
        self.width = width
        self.height = height
        self.inter = inter

    def __call__(self, image):
        """ functor call """
        return self.preprocess(image)

    def preprocess(self, image):
        """
        Resizes and crops the image considering its aspect ratio
        Args:
            image (np.ndarray): image

        return image (np.ndarray)
        """
        assert isinstance(image, np.ndarray)

        h, w = image.shape[:2]
        d_w = d_h = 0

        if w < h:
            image = imutils.resize(image, width=self.width, inter=self.inter)
            d_h = int((image.shape[0] - self.height) / 2.)
        else:
            image = imutils.resize(image, height=self.height, inter=self.inter)
            d_w = int((image.shape[1] - self.width) / 2.)

        h, w = image.shape[:2]
        image = image[d_h:h-d_h, d_w:w-d_w]

        return cv2.resize(image, (self.width, self.height), interpolation=self.inter)
