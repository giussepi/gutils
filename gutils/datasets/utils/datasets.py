# -*- coding: utf-8 -*-
""" gutils/datasets/utils/datasets """

import os
import shutil

import cv2
import numpy as np
from logzero import logger
from PIL import Image
from tqdm import tqdm

from gutils.images.files import list_images
from gutils.images.preprocessing import AspectAwarePreprocessor


def resize_dataset(
        dataset_path, output_directory, percentage=1., fixed_dims=None,
        img_extension='', masks=False, aspect_aware=True
):
    """
    Resize the images from a directory and saves them into output_directory

    Args:
        dataset_path       (str): dataset path
        output_directory   (str): resized dataset output directory path
        percentage       (float): percentage of dimensions (must be != 1 to work) e.g. 0.8
        fixed_dims (list, tuple): new dimensions to apply e.g.: (width, height)
        img_extension     (list): file image extension e.g.: ['.gif', '.ndpi'] to look for. Accepted
                                  formats are defined on imutils.paths.image_types
                                  (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
        masks             (bool): Whether or not working with masks
        aspect_aware      (bool): Where or not consider the aspect ratio when resizing

    Usage:
        resize_dataset('data/imgs/', 'data/imgs_0.5/', percentage=.5)
        resize_dataset('data/imgs/', 'data/imgs_960_640/', fixed_dims=(960, 640), aspect_aware=True)
        resize_dataset('data/masks/', 'data/masks_0.5/', percentage=.5, img_extension='.gif', masks=True)
        resize_dataset('data/masks/', 'data/masks_960_640/', fixed_dims=(960, 640), aspect_aware=True, masks=True)
    """
    assert float(percentage)
    assert percentage > 0
    assert os.path.exists(dataset_path)
    assert isinstance(img_extension, str)
    assert isinstance(aspect_aware, bool)

    if os.path.exists(output_directory):
        shutil.rmtree(output_directory)
    os.makedirs(output_directory)

    if fixed_dims is not None:
        assert isinstance(fixed_dims, (tuple, list))

    print("Creating new dataset at: {}".format(output_directory))

    for img_path in tqdm(list(list_images(dataset_path, [img_extension]))):
        extension = img_path.split('.')[-1]

        try:
            if extension == 'gif':
                cap = cv2.VideoCapture(img_path)
                ret, image = cap.read()
                cap.release()

                if ret:
                    img = np.array(Image.fromarray(image, 'RGB'))

                    if masks:
                        img_path = img_path.replace('.gif', '.png')
                else:
                    raise Exception()
            else:
                img = cv2.imread(img_path)
        except Exception:
            logger.warning("{} could not be openned".format(img_path))
            continue

        if percentage != 1:
            fixed_dims = (int(img.shape[1]*percentage), int(img.shape[0]*percentage))

        if aspect_aware:
            preprocessor = AspectAwarePreprocessor(*fixed_dims, inter=cv2.INTER_AREA)
            resized = preprocessor.preprocess(img)
            del preprocessor
        else:
            resized = cv2.resize(img, fixed_dims, interpolation=cv2.INTER_AREA)

        if masks:
            resized = resized[:, :, 0]

            if not np.array_equal(np.unique(resized), np.array([0, 255])):
                resized[resized > 0] = 255

        cv2.imwrite(os.path.join(output_directory, os.path.basename(img_path)), resized)
