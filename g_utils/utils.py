# -*- coding: utf-8 -*-
""" utils/utils """

import random
import string


def get_random_string(length=15):
    """
    Returns a random string with the length especified
    Args:
        length (int): string length
    Returns:
        str
    """
    assert isinstance(length, int)
    assert length > 0

    return ''.join(random.choices(string.ascii_letters+string.digits, k=length))
