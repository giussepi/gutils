# -*- coding: utf-8 -*-
""" gutils/settings """

try:
    import settings as global_settings
except ModuleNotFoundError:
    global_settings = None


QUICK_TESTS = getattr(global_settings, 'GUTILS_QUICK_TESTS', True)
