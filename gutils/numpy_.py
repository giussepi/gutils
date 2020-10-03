# -*- coding: utf-8 -*-
""" gutils/numpy_ """

import scipy as sp
import numpy as np


def colnorms_squared_new(matrix, blocksize=-1):
    """
    Calculate and returns the norms of the columns.
    If blocksize>0 the computation is performed in blocks to conserve memory

    Args:
        matrix (np.ndarray): matrix

    Returns:
        cols_norms (np.ndarray)
    """
    assert isinstance(matrix, np.ndarray)
    assert isinstance(blocksize, int)
    assert blocksize != 0

    if blocksize <= 0:
        return np.sum(matrix**2, axis=0)

    cols_norms = np.zeros(matrix.shape[1])
    blocksize = 2000

    for i in range(0, matrix.shape[1], blocksize):
        blockids = list(range(min(i+blocksize-1, matrix.shape[1])))
        cols_norms[blockids] = sum(matrix[:, blockids]**2)

    return cols_norms


def get_unique_rows(matrix):
    """
    Returns a matrix containing only unique rows

    Source: https://www.w3resource.com/python-exercises/numpy/python-numpy-exercise-87.php

    Args:
        matrix (np.ndarray)
    Returns:
        np.ndarray
    """
    assert isinstance(matrix, np.ndarray)

    y = np.ascontiguousarray(matrix).view(
        np.dtype((np.void, matrix.dtype.itemsize * matrix.shape[1])))
    _, idx = np.unique(y, return_index=True)

    return matrix[idx]


def normcols(matrix):
    """
    Returns an array with columns normalised

    Args:
        matrix (np.ndarray): matrix

    Returns:
        np.ndarray
    """
    assert isinstance(matrix, np.ndarray)

    return matrix/sp.linalg.norm(matrix, axis=0)


def format_label_matrix(label_matrix, sorted_labels=None):
    """
    Transforms the 1-d label_matrix to 2-D label_matrix and returns it

    Args:
        label_matrix   (np.ndarray): 1-D array
        sorted_labels (list, tuple): iterable with labels in the order to be used to create
                                     the 2-D label matrix

    Returns:
        label matrix (np.ndarray) with shape (num labels, num samples) e.g.:
        [0, 1, 0, 2, 2] is returned as:

        [[1 0 1 0 0],
         [0 1 0 0 0],
         [0 0 0 1 1]]
    """
    assert isinstance(label_matrix, np.ndarray)
    assert len(label_matrix.shape) == 1

    if sorted_labels is not None:
        assert isinstance(sorted_labels, (list, tuple))

    labels = sorted_labels if sorted_labels else sorted(set(label_matrix))
    lmatrix = np.zeros([len(labels), label_matrix.shape[0]], dtype=np.int32)

    for label, row in zip(labels, range(len(labels))):
        lmatrix[row][label_matrix == label] = 1

    return lmatrix
