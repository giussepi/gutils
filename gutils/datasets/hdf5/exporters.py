# -*- coding: utf-8 -*-
""" gutils/datasets/hdf5/exporters """

import os
import re

import numpy as np
from logzero import logger
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from gutils.datasets.hdf5.writers import HDF5DatasetWriter


class Images2HDF5:
    """
    General handler to export a classification/segmentation dataset into an HDF5 file

    Note:
        * When using masks the images (e.g. *.ann.tiff) and masks (e.g. *.mask.png)
          must have the same filename and only differ in the extension.
        * When using PyTorch, you can load the exported dataset using the class
          gtorch_utils.datasets.segmentation.hdf5.HDF5Dataset
          (see https://github.com/giussepi/gtorch_utils)

    Usage:
        # Classification dataset
        Images2HDF5(images_path=<path_to_dataset>)(
            data_dims=(1135, 640, 640, 4),
            labels_dims=(1135, )
        )

        # Segmentation dataset
        Images2HDF5(images_path=<path_to_dataset>)(
            data_dims=(1135, 640, 640, 4),
            masks_dims=(1135, 640, 640),
            labels_dims=(1135, )
        )
    """

    def __init__(self, **kwargs):
        """
        Initializes the object instance

        Kwargs:
            images_path     <str>: path to the folder for containing the images.
            masks_path      <str>: path to folder for containing the masks. If not provided,
                                   the images_path is used instead
            image_extension <str>: image extension. Default '.ann.tiff'
            mask_extension  <str>: mask extension. Default '.mask.png'
            image_reg       <str>: regular expression to get the label from the image name.
                                   Default r'(?P<label>[A-Z])_f(?P<file>\d+)_r(?P<roi>\-?\d+)_a(?P<annotation>\d+)_c(?P<counter>\d+)_?(?P<transforms>[A-Z\-]*).ann.tiff'
            saving_file     <str>: full path to the HDF5 file where all the data will be saved.
                                   Default 'dataset.hdf5'
            batch_size      <int>: Number images to be passed to the HDF5DatasetWriter.
                                   Default 32
        """
        self.images_path = kwargs.get('images_path')
        self.masks_path = kwargs.get('masks_path', self.images_path)
        self.image_extension = kwargs.get('image_extension', '.ann.tiff')
        self.mask_extension = kwargs.get('mask_extension', '.mask.png')
        self.image_reg = kwargs.get(
            'filename_reg',
            r'(?P<label>[A-Z])_f(?P<file>\d+)_r(?P<roi>\-?\d+)_a(?P<annotation>\d+)_c(?P<counter>\d+)_?(?P<transforms>[A-Z\-]*).ann.tiff',
        )
        self.saving_file = kwargs.get('saving_file', 'dataset.hdf5')
        self.batch_size = kwargs.get('batch_size', 32)

        assert isinstance(self.images_path, str)
        assert os.path.isdir(self.images_path)
        assert isinstance(self.masks_path, str)
        assert os.path.isdir(self.masks_path)
        assert isinstance(self.image_extension, str), type(self.image_extension)
        assert isinstance(self.mask_extension, str), type(self.mask_extension)
        assert isinstance(self.image_reg, str), type(self.image_reg)
        assert isinstance(self.saving_file, str), type(self.saving_file)

        basedir = os.path.dirname(self.saving_file)

        if basedir and not os.path.isdir(basedir):
            os.makedirs(basedir)

        assert isinstance(self.batch_size, int), type(self.batch_size)

    def __call__(self, **kwargs):
        """ functor call """
        return self.process(**kwargs)

    def preprocess(self, batch_images, batch_labels, batch_masks):
        """
        Overwrite this method to perform modifications before saving the data

        Args:
            batch_images       <list>: list of images
            batch_labels <np.ndarray>: list of labels
            batch_masks        <list>: list of masks

        Returns:
            batch_images <list>, batch_labels <list>, batch_masks <list>
        """
        assert isinstance(batch_images, list), type(batch_images)
        assert isinstance(batch_labels, np.ndarray), type(batch_labels)
        assert isinstance(batch_masks, list), type(batch_masks)

        batch_images = [np.array(i) for i in batch_images]
        batch_masks = [np.array(i) for i in batch_masks]

        return batch_images, batch_labels, batch_masks

    def process(self, **kwargs):
        """
        Iterates over the dataset and saves all its elements in a HDF5 file

        Kwargs:
            data_dims   <tuple>: dimensions of the data to be saved.
                                 E.g. (<num_items>, <height>, <width>, <channels>)
            masks_dims  <tuple>: dimensions of the masks to be saved. E.g. (<num_items>, <height>, <width>).
                                 Set it to None (Default) when there are no masks.
            labels_dims <tuple>: dimensions of the label to be saved. E.g. (<num_items>, )
            data_key      <str>: name of the hdf5 dataset holding the data. Default 'images'
            data_dtype       <>: data type to be used to store the data
                                 Default h5py.h5t.STD_U8BE
            masks_key     <str>: name of the hdf5 dataset holding the masks. Default 'masks'
            masks_dtype      <>: data type to be used to store the masks
                                 Default h5py.h5t.STD_U8BE
            labels_key    <str>: name of the hdf5 dataset holding the labels. Default 'labels'
            labels_dtype     <>: data type to be used to store the targets.
                                 Default 'int8'
            buffer_size   <int>: number of objects to be stored in memory before being sent to disk.
                                 Default 1024
        """
        image_list = list(filter(
            lambda x: x.endswith(self.image_extension), os.listdir(self.images_path)))
        pattern = re.compile(self.image_reg)

        labels = [pattern.fullmatch(i).groupdict()['label'] for i in image_list]
        label_encoder = LabelEncoder()
        labels = label_encoder.fit_transform(labels)

        if 'output_path' in kwargs:
            kwargs.pop('output_path')

        db_writer = HDF5DatasetWriter(output_path=self.saving_file, **kwargs)
        db_writer.store_label_names(label_encoder.classes_)

        logger.info(f'Saving dataset into {self.saving_file}')
        masks_dims = kwargs.get('masks_dims', None)

        for i in tqdm(range(0, len(image_list), self.batch_size), unit='batch'):
            batch_labels = labels[i:i+self.batch_size]
            batch_images = image_list[i:i+self.batch_size]

            if masks_dims:
                batch_masks = [Image.open(os.path.join(self.images_path, i.replace(
                    self.image_extension, self.mask_extension))) for i in batch_images]
            else:
                batch_masks = None

            batch_images = [Image.open(os.path.join(self.images_path, i)) for i in batch_images]
            db_writer.add(*self.preprocess(batch_images, batch_labels, batch_masks))

        db_writer.close()
