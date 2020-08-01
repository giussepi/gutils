# -*- coding: utf-8 -*-
""" utils/decorators """

from functools import wraps
from time import time


def timing(function):
    """
    Times execution time in seconds and prints it
    Args:
        function: Function or method to be timed
    Note: The first argument of the function or method must contain an attribute
          timeit = False to disable the time trancking and printing.
    """
    @wraps(function)
    def wrap(*args, **kw):
        start = time()
        result = function(*args, **kw)
        end = time()
        name = getattr(
            function, 'py_func.__qualname__', getattr(function, '__name__', function.__str__()))
        print('func:{} processed in {:.4f} seconds'.format(name, end-start))

        return result

    return wrap
