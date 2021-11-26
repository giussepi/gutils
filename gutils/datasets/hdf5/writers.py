# -*- coding: utf-8 -*-
""" gutils/datasets/hdf5/writers """

import os

import h5py
import numpy as np


class HDF5DatasetWriter:
    """
    General handler to create an HDF5 database

    Usage:
        db = HDF5DatasetWriter(
            data_dims=(1135, 640, 640, 4),
            masks_dims=(1135, 640, 640),
            labels_dims=(1135, )
        )
    """

    def __init__(self, **kwargs):
        """
        Kwargs:
            data_dims   <tuple>: dimensions of the data to be saved.
                                 E.g. (<num_items>, <height>, <width>, <channels>)
            masks_dims  <tuple>: dimensions of the masks to be saved. E.g. (<num_items>, <height>, <width>).
                                 Set it to None (Default) when there are no masks.
            labels_dims <tuple>: dimensions of the label to be saved. E.g. (<num_items>, )
            output_path   <str>: full path to the HDF5 file where the data will be written.
                                 Default 'mydataset.hdf5'
            data_key      <str>: name of the HDF5 dataset holding the data. Default 'images'
            data_dtype       <>: data type to be used to store the data
                                 Default h5py.h5t.STD_U8BE
            masks_key     <str>: name of the HDF5 dataset holding the masks. Default 'masks'
            masks_dtype      <>: data type to be used to store the masks
                                 Default h5py.h5t.STD_U8BE
            labels_key    <str>: name of the HDF5 dataset holding the labels. Default 'labels'
            labels_dtype     <>: data type to be used to store the targets.
                                 Default 'int8'
            buffer_size   <int>: number of objects to be stored in memory before being sent to disk.
                                 Default 1024
        """
        data_dims = kwargs.get('data_dims')
        masks_dims = kwargs.get('masks_dims', None)
        labels_dims = kwargs.get('labels_dims')
        output_path = kwargs.get('output_path', 'mydataset.hdf5')
        self.data_key = kwargs.get('data_key', 'images')
        data_dtype = kwargs.get('data_dtype', h5py.h5t.STD_U8BE)
        self.masks_key = kwargs.get('masks_key', 'masks')
        masks_dtype = kwargs.get('masks_dtype', h5py.h5t.STD_U8BE)
        self.labels_key = kwargs.get('labels_key', 'labels')
        labels_dtype = kwargs.get('labels_dtype', 'int8')
        self.buffer_size = kwargs.get('buffer_size', 1024)

        assert isinstance(data_dims, (tuple)), type(data_dims)

        if masks_dims:
            assert isinstance(masks_dims, tuple), type(masks_dims)

        assert isinstance(labels_dims, (tuple)), type(labels_dims)

        if os.path.isfile(output_path):
            raise ValueError(f"{output_path} already exists and cannot be overwritten."
                             " Manually delete the file before continuing.")

        basedir = os.path.dirname(output_path)

        if basedir and not os.path.isdir(basedir):
            os.makedirs(basedir)

        assert isinstance(self.data_key, str), type(self.data_key)
        assert isinstance(self.masks_key, str), type(self.masks_key)
        assert isinstance(self.labels_key, str), type(self.labels_key)
        assert isinstance(self.buffer_size, int), type(self.buffer_size)

        self.db = h5py.File(output_path, "w")
        self.data = self.db.create_dataset(self.data_key, data_dims, dtype=data_dtype)

        if masks_dims:
            self.masks = self.db.create_dataset(self.masks_key, masks_dims, dtype=masks_dtype)

        self.labels = self.db.create_dataset(self.labels_key, labels_dims, dtype=labels_dtype)
        self.buffer = {self.data_key: [], self.masks_key: [], self.labels_key: []}
        self.idx = 0

    def add(self, data, labels, masks=None):
        """
        Kwargs:
            data   <list>: list of NumPy arrays containing images data
            masks  <list>: list of NumPy arrays containing masks data
            labels <list>: list integers containing the encoded labels
        """
        self.buffer[self.data_key].extend(data)
        self.buffer[self.labels_key].extend(labels)

        if masks:
            self.buffer[self.masks_key].extend(masks)

        if len(self.buffer[self.data_key]) >= self.buffer_size:
            self.flush()

    def flush(self):
        """ Writes the buffers to disk then reset the buffer """
        i = self.idx + len(self.buffer[self.data_key])
        self.data[self.idx:i] = self.buffer[self.data_key]

        if hasattr(self, 'masks'):
            self.masks[self.idx:i] = self.buffer[self.masks_key]

        self.labels[self.idx:i] = self.buffer[self.labels_key]
        self.idx = i
        self.buffer = {self.data_key: [], self.masks_key: [], self.labels_key: []}

    def store_label_names(self, label_names):
        """
        Creates a dataset to store the actual label names, then store the class labels

        Args:
            label_names <np.ndarray>: Label names
        """
        assert isinstance(label_names, np.ndarray), type(label_names)

        label_set = self.db.create_dataset(
            "label_names", (len(label_names),), dtype=h5py.special_dtype(vlen=str))
        label_set[:] = label_names

    def close(self):
        """
        Checks to see if there are any other entries in the buffer that need to be flushed to disk,
        then closes the dataset
        """
        if len(self.buffer[self.data_key]) > 0:
            self.flush()

        self.db.close()
