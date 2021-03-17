# -*- coding: utf-8 -*-
""" gutils/mock """


def notqdm(iterable, *args, **kwargs):
    """
    Replacement for tqdm that just passes back the iterable
    useful to silence `tqdm` in tests

    Use it along with mock.patch decorator. E.g.:
    @patch('Data.Prepare_patches.CRLM.tqdm', notqdm)
    def myfunc(*args, **kwars):

    Source: https://stackoverflow.com/questions/37091673/silence-tqdms-output-while-running-tests-or-running-the-code-via-cron#answer-46689485
    """
    return iterable
