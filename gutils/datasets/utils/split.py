# -*- coding: utf-8 -*-
""" gutils/datasets/utils/split """

import numpy as np
from sklearn.model_selection import train_test_split


def calculate_train_val_percentages(val_size, test_size):
    """
    Generaly, when using sklearn.model_selection.train_test_split we get the train and test splits.
    Thus, the validation split use to be calculated by running a second train_test_split over the
    train split so we end with train_train, train_val and test splits. In this was they val_size is
    calculated over the train split and not over the whole datastet.

    This function returns the right train_train_size and train_val_size to be used to properly split
    the train split in a way that val_size is taken from the whole datastet. E.g.:

    total = test_size * total + val_size * total + train_train_size * (total*(1-test_size))

    Thus, if you want to split your dataset in train 65%, val 10% and test 25% the following
    will be true:

    TOTAL * VAL_SIZE == TOTAL * (1-TEST_SIZE) * TRAIN_VAL_SIZE
    TOTAL * VAL_SIZE == TOTAL * (1-TEST_SIZE) * VAL_SIZE/(1-TEST_SIZE)

    E.g. If we had a dataset with 100 samples:
     100  *   0.1    == 100   *   (1-0.25)    *   0.1 / (1-0.25)
                  10 == 10

    Usage:
        # first we get the final test split
        x_train_val, x_test, y_train_val, y_test = train_test_split(myfeatures, mylabels,
            test_size=.25, ramdom_state=42, stratify=mylabels)

        # second calculate the equivalent percentages in the train_val split
        train_train_size, train_val_size = calculate_train_val_subdataset_percentage(.1, .25)

        # Finally, we get the final train and validation splits
        # option 1
        x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val,
            train_size=train_train_size, ramdom_state=42, stratify=y_train_val)
        # option 2
        x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val,
            test_size=train_val_size, ramdom_state=42, stratify=y_train_val)
    """
    assert isinstance(val_size, (float, int))
    assert isinstance(test_size, (float, int))
    assert 0 < val_size < 100
    assert 0 < test_size < 100

    val_size, test_size = list(map(lambda x: x/100 if x >= 1 else x, (val_size, test_size)))

    train_val_size = val_size/(1-test_size)
    train_train_size = 1 - train_val_size

    return train_train_size, train_val_size


class TrainValTestSplit:
    """
    Splits the dataset into train, validation and test subdatasets

    Usage:
        x_train, x_val, x_test, y_train, y_val, y_test = TrainTestSplit(samples, targets, val_size, test_size)()
    """

    def __init__(self, samples, targets, val_size=.1, test_size=.2, **kwargs):
        """
        Initializes the instance

        Args:
            samples (np.ndarray): numpy array with shape [<num samples>, <num features>]
            targets (np.ndarray): numpy array with shape [<num_samples>] or [<num_samples>, <num classes>]
            val_size     (float): validation dataset size in range [0, 1]
            test_size    (float): test dataset size in range [0, 1]

        Kwargs:
            random_stat   e (int): Controls the shuffling applied to the data before applying the split.
            shuffle        (bool): Whether or not to shuffle the data before splitting.
            stratify (np.ndarray): If not None, data is split in a stratified fashion, using this as the class labels.
        """
        self.samples = samples
        self.targets = targets
        self.val_size = val_size
        self.test_size = test_size
        self.random_state = kwargs.get('random_state', 42)
        self.shuffle = kwargs.get('shuffle', True)
        self.stratify = kwargs.get('stratify', None)

        assert isinstance(self.samples, np.ndarray)
        assert isinstance(self.targets, np.ndarray)
        assert isinstance(self.val_size, (float, int))
        assert isinstance(self.test_size, (float, int))
        assert isinstance(self.random_state, int)
        assert isinstance(self.shuffle, bool)
        assert self.samples.shape[0] == self.targets.shape[0]
        if self.stratify is not None:
            assert isinstance(self.stratify, np.ndarray)

    def __call__(self):
        """ Functor call """
        return self.__split_dataset()

    def __split_dataset(self):
        """
        Splits the dataset into train, validation and test datasets and returns them as
        NumPy arrays

        Returns:
            x_train, x_val, x_test, y_train, y_val, y_test
        """

        x_train_val, x_test, y_train_val, y_test = train_test_split(
            self.samples, self.targets, test_size=self.test_size,
            random_state=self.random_state, stratify=self.stratify
        )

        train_val_size = calculate_train_val_percentages(self.val_size, self.test_size)[1]

        x_train, x_val, y_train, y_val = train_test_split(
            x_train_val, y_train_val, test_size=train_val_size,
            random_state=self.random_state,
            stratify=y_train_val if self.stratify is not None else None
        )

        return x_train, x_val, x_test, y_train, y_val, y_test
