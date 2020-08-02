# -*- coding: utf-8 -*-
""" gutils/folders """

import os
import shutil


def remove_folder(folder_path):
    """
    Removes a directory if it exists
    Args:
        folder_path (str): path to the folder to be removed
    """
    assert isinstance(folder_path, str)

    if os.path.isdir(folder_path):
        shutil.rmtree(folder_path)


def clean_create_folder(folder_path):
    """
    Removes the folder and recreates it
    Args:
        folder_path (str): path to the folder to be (re)-created
    """
    remove_folder(folder_path)
    os.makedirs(folder_path)
