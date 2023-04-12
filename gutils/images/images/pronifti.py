# -*- coding: utf-8 -*-
""" gutils/images/images/pronifti """

from typing import Union

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from mpl_toolkits.axes_grid1 import ImageGrid

from gutils.images.images.dicom import DICOM
from gutils.images.images.nifti import NIfTI


__all__ = ['ProNIfTI']


class ProNIfTI(NIfTI):
    """
    Processed NIfTI class
    Creates and provides methods for basic operations over processed NIfTI (pro.nii.gz) files created
    from DICOM files
    """

    EXTENSION = 'pro.nii.gz'

    @staticmethod
    def create_save(dcm_list: Union[list, tuple], /, *, processing: dict = None,
                    saving_path: str = 'new_pronifti.pro.nii.gz',
                    dicom_min_val: Union[int, float] = 0, dicom_max_val: Union[int, float] = 0):
        """
        Saves all the DICOMs from the dcm_list as a single ProNIfTI file

        Kwargs:
            dcm_list <list, tuple>: iterable containing the DICOM paths
            processing      <dict>: dictionary with all the DICOM->methods and kwargs to be used.
                                    e.g. dict(resize={'target':(256,256)}). By default all methods receive
                                    inplace=True as keyword argument.
                                    Default None
            saving_path      <str>: full path to save the image. Default new_pronifti.pro.nii.gz
            dicom_min_val <int, float>: minimum value of the DICOM file. Default 0
            dicom_max_val <int, float>: maximum value of the DICOM file. Default 0
        """
        assert isinstance(dcm_list, (list, tuple)), type(dcm_list)
        assert len(dcm_list) > 0, len(dcm_list)
        assert isinstance(dicom_min_val, (int, float)), type(dicom_min_val)
        assert isinstance(dicom_max_val, (int, float)), type(dicom_max_val)
        assert dicom_max_val >= dicom_min_val

        if processing:
            assert isinstance(processing, dict), type(processing)
            for method, kwargs in processing.items():
                assert isinstance(method, str), type(method)
                assert isinstance(kwargs, dict), type(kwargs)
        else:
            processing = {}
        assert isinstance(saving_path, str), type(saving_path)

        data = []
        for filepath in dcm_list:
            dicom = DICOM(filepath, dicom_min_val, dicom_max_val)
            for method, kwargs in processing.items():
                kwargs['inplace'] = True  # making sure all operations are made in place
                getattr(dicom, method)(**kwargs)
            data.append(dicom.ndarray[:, :, np.newaxis])

        data = np.concatenate(data, axis=2)
        new_nifti = nib.Nifti1Image(data, affine=np.eye(4), dtype=np.int64)
        nib.save(new_nifti, saving_path)

    def plot(self, rows: int = 2, cols: int = 2, /):
        """
        Plots a grid of rows * cols images takend from the loaded data

        Kwargs:
            rows <int>: number of rows. Default 2
            cols <int>: number of columns. Default 2
        """
        assert isinstance(rows, int), type(rows)
        assert isinstance(cols, int), type(cols)

        fig = plt.figure()
        grid = ImageGrid(fig, 111, nrows_ncols=(rows, cols), axes_pad=0.1)

        for axis, img_idx in zip(grid, range(self.shape[-1])):
            axis.imshow(self.ndarray[..., img_idx], cmap='gray')

        plt.show()

    def plot_3d_ndarray(self):
        raise NotImplementedError

    def clean_3d_ndarray(self, *, height: int = -1, inplace: bool = False):
        raise NotImplementedError
