# -*- coding: utf-8 -*-
""" gutils/images/augmentation """

import os
import re
from collections import namedtuple
from itertools import combinations

import albumentations as A
import cv2
import numpy as np
from logzero import logger
from tqdm import tqdm

from gutils.constants import AugmentationType


AugmentType = namedtuple('AugmentType', ['type', 'transform', 'id'])


class ImageAugmentationProcessor:
    """
    Offline data augmentation
    Creates and saves a user defined number of images/masks applying some image transformations

    Usage:
        image_path = "/media/giussepi/2_0_TB_Hard_Disk/CRLM/annotations_masks/F_f001_r-1_a00365_c00363.ann.tiff"
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ImageAugmentationProcessor()(n=8, image=image, image_name=os.path.basename(image_path))

        # including masks
        mask_path = "/media/giussepi/2_0_TB_Hard_Disk/CRLM/annotations_masks/F_f001_r-1_a00365_c00363.mask.png"
        mask = cv2.imread(mask_path)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        ImageAugmentationProcessor()(
            n=3, original=True, image=image, image_name=os.path.basename(image_path), mask=mask,
            mask_name=os.path.basename(mask_path)
        )
    """
    # Pixel-level transforms ##################################################
    # CLAHE = AugmentType(AugmentationType.PIXEL_LEVEL, A.CLAHE(1.005, p=1), 'CLAHE')
    BRIGHT_CONTRAST = AugmentType(AugmentationType.PIXEL_LEVEL, A.ColorJitter(.1, .15, 0, 0, p=1), 'BC')
    DOWNSCALE = AugmentType(AugmentationType.PIXEL_LEVEL, A.Downscale(.9, .9, p=1), 'DWS')
    NOISE = AugmentType(
        AugmentationType.PIXEL_LEVEL,
        A.OneOf([
            A.GaussNoise((10.0, 30.0), p=.33),
            A.ISONoise((.01, .05), (.1, .5), p=.33),  # Apply camera sensor noise
            A.MultiplicativeNoise((.9, 1.1), p=.33),
        ], p=1),
        'NOISE'
    )
    HUE_SATURATION = AugmentType(AugmentationType.PIXEL_LEVEL, A.HueSaturationValue(3, 6, 3, p=1), 'HUE')
    COMPRESSION = AugmentType(AugmentationType.PIXEL_LEVEL, A.ImageCompression(95, 100, p=1), 'CMP')
    RAND_GAMMA = AugmentType(AugmentationType.PIXEL_LEVEL, A.RandomGamma(p=1), 'RANDG')

    # Spatial-level transforms ################################################
    DISTORTION = AugmentType(
        AugmentationType.SPATIAL_LEVEL,
        A.OneOf([
            A.ElasticTransform(1, 50, 50, p=.33),
            A.GridDistortion(5, .2, p=.33),
            A.OpticalDistortion(.2, .2, p=.33),
        ], p=1),
        'DIST'
    )
    HORIZONTAL_FLIP = AugmentType(AugmentationType.SPATIAL_LEVEL, A.HorizontalFlip(p=1), 'HF')
    VERTICAL_FLIP = AugmentType(AugmentationType.SPATIAL_LEVEL, A.VerticalFlip(p=1), 'VF')
    ROTATION = AugmentType(
        AugmentationType.SPATIAL_LEVEL,
        A.OneOf([
            A.Rotate((90, 90), p=.33),
            A.Rotate((180, 180), p=.33),
            A.Rotate((270, 270), p=.33)
        ], p=1),
        'RO'
    )
    NO_OP = AugmentType(AugmentationType.SPATIAL_LEVEL, A.NoOp(), 'ORIG')

    ALL_TRANSFORMS = (
        ROTATION, VERTICAL_FLIP, HORIZONTAL_FLIP, DISTORTION, RAND_GAMMA, COMPRESSION, HUE_SATURATION,
        NOISE, DOWNSCALE, BRIGHT_CONTRAST
    )

    def __call__(self, **kwargs):
        """ functor call """
        return self.create_n_images(**kwargs)

    @property
    def num_transforms(self):
        """ returns the number of transforms available """
        return len(self.ALL_TRANSFORMS)

    def apply_transforms_and_save(self, **kwargs):
        """
        Applies the transformations provided and saves the images/masks.

        Kwags:
            augmentations_list <list, tuple>: list of AugmentType instances
            image   <np.ndarray>: image to be transformed. Channels order must be RGB or RGBA
            image_name     <str>: full image name including extension
            image_filename <str>: image filename without extension
            mask    <np.ndarray>: mask to be transformed. Could be None.
            mask_name      <str>: full mask name including extension. Default ''
            mask_filename  <str>: image filename without extension
            saving_path    <str>: path to folder where to save the augmented data.
                                  Default 'augmented_dataset'
            img_format     <str>: Saving image extension. Default 'tiff'
            mask_format    <str>: Saving mask format. Default 'png'
        """
        augmentations_list = kwargs.get('augmentations_list')
        image = kwargs.get('image')
        image_name = kwargs.get('image_name')
        image_filename = kwargs.get('image_filename')
        mask = kwargs.get('mask', None)
        mask_name = kwargs.get('mask_name', '')
        mask_filename = kwargs.get('mask_filename', '')
        saving_path = kwargs.get('saving_path', 'augmented_dataset')
        img_format = kwargs.get('img_format', 'tiff')
        mask_format = kwargs.get('mask_format', 'png')

        assert isinstance(augmentations_list, (list, tuple)), type(augmentations_list)
        assert len(augmentations_list) > 0, len(augmentations_list)
        assert isinstance(augmentations_list[0], AugmentType), type(augmentations_list[0])
        assert isinstance(image, np.ndarray), type(image)
        assert isinstance(image_name, str), type(image_name)
        assert isinstance(image_filename, str), type(image_filename)
        assert image_filename != '', image_filename
        assert isinstance(mask_name, str), type(mask_name)

        if mask is not None:
            assert isinstance(mask, np.ndarray), type(mask)
            assert mask_name != '', mask_name
            assert mask_filename != '', mask_filename
            assert isinstance(mask_format, str), type(mask_format)
            assert mask_format != '', mask_format

        assert os.path.isdir(saving_path), saving_path
        assert isinstance(img_format, str), type(img_format)
        assert img_format != '', img_format

        # NOTE: some transforms like ImageCompression returns only RGB images
        #       so to avoid errors when having RGBA images it is necessary
        #       to transform them into RGB images before any processing
        if image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

        transform = A.Compose([augmentation.transform for augmentation in augmentations_list])
        augmented = transform(image=image, mask=mask) if mask is not None else transform(image=image)
        transforms_ids = "-".join([str(augmentation.id) for augmentation in augmentations_list])
        augmented_path = os.path.join(
            saving_path, image_filename + '_{}.ann.{}'.format(transforms_ids, img_format))
        cv2.imwrite(augmented_path, cv2.cvtColor(augmented['image'], cv2.COLOR_RGB2BGR))

        if mask is not None:
            augmented_path = os.path.join(
                saving_path, mask_filename + '_{}.mask.{}'.format(transforms_ids, mask_format))
            if 'mask' in augmented.keys():
                cv2.imwrite(augmented_path, cv2.cvtColor(augmented['mask'], cv2.COLOR_RGB2BGR))
            else:
                cv2.imwrite(augmented_path, cv2.cvtColor(mask, cv2.COLOR_RGB2BGR))

    def transforms_iterator(self, original=True, min_length=1):
        """
        Transforms list iterator based on the incremental combination of
        ALL_TRANSFORMS elements

        Args:
            original  <bool>: whether or not include the original image. Default True
            min_length <int>: minimum combinations length to start with

        Returns:
            augmentations <tuple>
        """
        assert isinstance(original, bool), type(original)
        assert isinstance(min_length, int), type(min_length)
        assert 1 <= min_length <= self.num_transforms, min_length

        if original:
            yield (self.NO_OP, )

        for combination_length in range(min_length, self.num_transforms+1):
            for combination in combinations(self.ALL_TRANSFORMS, combination_length):
                yield combination

    def create_n_images(self, **kwargs):
        """
        Applies incremental combinations of image transformations to create the desired number of
        image augmentations

        Kwags:
            n              <int>: number of images to create
            min_transforms <int>: minimum number of transforms to apply. Default 1
            original      <bool>: whether or not include the original image. Default True
            image   <np.ndarray>: image to be transformed. Channels order must be RGB or RGBA
            image_name     <str>: full image name including extension
            image_reg      <str>: regular expression to extract the name from the image_name.
                                  Default r'(?P<filename>.+).ann.tiff'
            mask    <np.ndarray>: mask to be transformed. Could be None.
            mask_name      <str>: full mask name including extension. Default ''
            mask_reg       <str>: regular expression to extract the name from the mask_name.
                                  Default r'(?P<filename>.+).mask.png'
            saving_path    <str>: path to folder where to save the augmented data.
                                  Default 'augmented_dataset'
            img_format     <str>: Saving image extension. Default 'tiff'
            mask_format    <str>: Saving mask format. Default 'png'
        """
        n = kwargs.get('n')
        min_transforms = kwargs.get('min_transforms', 1)
        original = kwargs.get('original', True)
        image = kwargs.get('image')
        image_name = kwargs.get('image_name')
        image_reg = kwargs.get('image_reg', r'(?P<filename>.+).ann.tiff')
        mask = kwargs.get('mask', None)
        mask_name = kwargs.get('mask_name', '')
        mask_reg = kwargs.get('mask_reg', r'(?P<filename>.+).mask.png')
        saving_path = kwargs.get('saving_path', 'augmented_dataset')
        img_format = kwargs.get('img_format', 'tiff')
        mask_format = kwargs.get('mask_format', 'png')

        assert isinstance(n, int), type(n)
        assert n >= 0, n
        assert isinstance(min_transforms, int), type(min_transforms)
        assert 1 <= min_transforms <= self.num_transforms, min_transforms
        assert isinstance(original, bool), type(original)
        assert isinstance(image, np.ndarray), type(image)
        assert isinstance(image_name, str), type(image_name)
        assert isinstance(image_reg, str), type(image_reg)
        assert isinstance(mask_name, str), type(mask_name)
        assert isinstance(mask_reg, str), type(mask_reg)

        if mask is not None:
            assert isinstance(mask, np.ndarray), type(mask)
            assert mask_name != '', mask_name
            assert mask_reg != '', mask_reg
            assert isinstance(mask_format, str), type(mask_format)
            assert mask_format != '', mask_format

        assert isinstance(saving_path, str), type(saving_path)
        assert isinstance(img_format, str), type(img_format)
        assert img_format != '', img_format

        if n == 0:
            logger.info("n has been set to 0; thus, no image was created")
            return

        if not os.path.isdir(saving_path):
            os.makedirs(saving_path)

        image_pattern = re.compile(image_reg)
        mask_pattern = re.compile(mask_reg) if mask is not None else ''
        counter = 0

        try:
            img_filename = image_pattern.fullmatch(image_name).groupdict()['filename']
        except AttributeError as err:
            logger.error(err)
            return

        if mask is not None:
            try:
                mask_filename = mask_pattern.fullmatch(mask_name).groupdict()['filename']
            except AttributeError as err:
                logger.error(err)
                return
        else:
            mask_filename = ''

        logger.info("Applying augmentation transforms")

        for augmentations in tqdm([*self.transforms_iterator(original, min_transforms)]):
            self.apply_transforms_and_save(
                augmentations_list=augmentations,
                image=image,
                image_name=image_name,
                image_filename=img_filename,
                mask=mask,
                mask_name=mask_name,
                mask_filename=mask_filename,
                saving_path=saving_path,
                img_format=img_format,
                mask_format=mask_format
            )

            counter += 1

            if counter == n:
                break
        else:
            if counter < n:
                logger.warning(
                    "Only {} out of {} were created using the combinations of the {} transformations."
                    .format(counter, n, self.num_transforms)
                )
