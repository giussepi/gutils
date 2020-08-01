# -*- coding: utf-8 -*-
""" utils/files """

import os
from utils.utils import get_random_string


def get_filename_and_extension(string_):
    """
    Extracts and returns the file name and extension from a file path or file name
    Args:
        string_ (str): File path or file name
    Returns:
        '<filename>', '<extension>'
    """
    assert isinstance(string_, str)
    assert bool(string_), 'Empty strings are not allowed'

    bits = os.path.basename(string_).split('.')

    if len(bits) > 2:
        return '.'.join(bits[:-1]), bits[-1]
    if len(bits) == 2:
        return bits[0], bits[1]

    return bits[0], ''


def clean_json_filename(filename):
    """
    Verifies and returns a filename, or random string if not provided, with the .json extension.
    Args:
        filename (str): file name
    Returns
        '<filname or random string>.json'
    """
    assert isinstance(filename, str)

    if filename:
        assert filename.endswith('.json'), 'The filename does not have a .json extension'

    if not filename:
        filename = get_random_string() + '.json'

    return filename
