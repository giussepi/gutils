# -*- coding: utf-8 -*-
""" gutils/images/postprocessing/remove_bg """

from typing import Callable

import numpy as np
from PIL import Image
from skimage.filters import threshold_otsu


class RemoveBG:
    """
    Removes the background from an PIL image

    Usage:
        RemoveBG()(img)
    """

    def __init__(self, bg_color=255, threshold=threshold_otsu):
        """
        Initialzes the object instance

        Args:
            bg_color                   <int>: background colour. Default 255
            threshold <int, float, Callable>: threshold value or callable to calculate it.
                                              Default threshold_otsu
        """
        assert isinstance(bg_color, int), type(bg_color)
        assert isinstance(threshold, (int, float, Callable)), type(threshold)

        self.bg_color = bg_color
        self.threshold = threshold

    def __call__(self, *args):
        """
        Args:
            img <PIL image>: PIL image input
        """
        return self.process(*args)

    @staticmethod
    def remove_transparency(img, bg_colour=(255, 255, 255)):
        """
        Args:
            img <PIL image>: PIL image input

        Returns:
            img <PIL image>

        Source: https://stackoverflow.com/questions/44997339/convert-python-image-to-single-channel-from-rgb-using-pil-or-scipy#answer-48624958
        """
        # Only process if image has transparency
        if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):

            # Need to convert to RGBA if LA format due to a bug in PIL
            alpha = img.convert('RGBA').split()[-1]

            # Create a new background image of our matt color.
            # Must be RGBA because paste requires both images have the same format

            bg = Image.new("RGBA", img.size, bg_colour + (255,))
            bg.paste(img, mask=alpha)

            return bg

        return img

    @classmethod
    def convert_to_single_channel_gray(cls, img):
        """
        Args:
            img <PIL image>: image input

        Returns:
           img <PIL image>
        """
        return cls.remove_transparency(img).convert('L')

    @staticmethod
    def get_cleaned_img(img):
        """
        Returns a RGB numpy array

        Args:
            img <PIL image>: PIL image input

        Returns:
            img <np.ndarray>
        """
        if img.mode != 'RGB':
            img = img.convert('RGB')

        return np.array(img)

    def get_bg_RGB_mask(self, img):
        """
        Calculates the background threshold and returns a cleaned numpy array of the img
        and a MxNx3 numpy array with the indexes of the background

        Args:
            img <PIL image>: PIL image input

        Returns:
            img <np.ndarray>, rgb_mask <np.ndarray>
        """
        gray = np.array(self.convert_to_single_channel_gray(img))
        threshold = self.threshold(gray) if callable(self.threshold) else self.threshold
        print(f'threshold: {threshold}')
        img = self.get_cleaned_img(img)
        binary = gray > threshold
        rgbmask = np.moveaxis(np.stack([binary]*3, axis=0), [0, 1, 2], [2, 0, 1])

        return img, rgbmask

    def process(self, img, bg_rgbmask=None):
        """
        Replaces the background using the self.bg_color and returns the result as a PIL Image

        Args:
            img <PIL image>: PIL image input

        Returns:
            img <np.ndarray>
        """
        if bg_rgbmask is None:
            img, bg_rgbmask = self.get_bg_RGB_mask(img)
        else:
            img = self.get_cleaned_img(img)

        img[bg_rgbmask] = self.bg_color

        return Image.fromarray(img)
