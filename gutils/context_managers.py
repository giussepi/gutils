# -*- coding: utf-8 -*-
""" gutils/context_managers """

import contextlib

import joblib


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """
    Context manager to patch joblib to report into tqdm progress bar given as argument

    Source: https://stackoverflow.com/questions/24983493/tracking-progress-of-joblib-parallel-execution#answer-58936697

    Usage:
        form math import sqrt
        from tqdm import tqdm
        from joblib import Parallel, delayed

        with tqdm_joblib(tqdm(desc="My calculation", total=10)) as progress_bar:
            Parallel(n_jobs=16)(delayed(sqrt)(i**2) for i in range(10))
    """
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback

    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()
