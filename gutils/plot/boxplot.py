# -*- coding: utf-8 -*-
""" gutils/plot/boxplot """

import matplotlib.pyplot as plt
import numpy as np


__all__  = ['BoxPlot']


class BoxPlot:
    """
    Provides methods to plot a boxplot and get the values of its components
    """

    def __init__(self, data: np.ndarray, /):
        """ Initializes the object instance """
        assert isinstance(data, np.ndarray), type(data)

        self.data = data.ravel()

    def plot(self, save: bool = False, save_filename: str = 'boxplot.png', dpi: int = 200):
        """
        Plots the boxplot using matplotlib.pyplot. If save is True, the figure is saved to disk

        Kwargs:
            save         <bool>: If True, saves the figure to disk. Default False
            save_filename <str>: filename for the saving the figure. Default 'boxplot.png'
            dpi           <int>: Dpi for saved image
        """
        plt.boxplot(self.data)

        if save:
            plt.savefig(save_filename, dpi=dpi)
        else:
            plt.show()

    def find_quantiles_median_iqr_wiskers(self):
        """
        source: https://www.geeksforgeeks.org/finding-the-outlier-points-from-matplotlib/

        Returns:
            q1<float>, q3<float>, med<float>, iqr<float>, lower_bound<float>, upper_bound<float>
        """
        # finding the 1st quartile
        q1 = np.quantile(self.data, 0.25)

        # finding the 3rd quartile
        q3 = np.quantile(self.data, 0.75)
        med = np.median(self.data)

        # finding the iqr region
        iqr = q3-q1

        # finding upper and lower whiskers
        upper_bound = q3+(1.5*iqr)
        lower_bound = q1-(1.5*iqr)

        return q1, q3, med, iqr, lower_bound, upper_bound

    def find_outliers(self, lower_bound: float = None, upper_bound: float = None):
        """
        source: https://www.geeksforgeeks.org/finding-the-outlier-points-from-matplotlib/

        Kwargs:
            lower_bound <float>: boxplot lower bound. If not provided it is computed
            upper_bound <float>: boxplot upper bound. If not provided it is computed

        Returns:
            lower_outliers<np.ndarray>, upper_outliers<np.ndarray>
        """
        if lower_bound is None or upper_bound is None:
            lower_bound, upper_bound = self.find_quantiles_median_iqr_wiskers()[-2:]

        assert isinstance(lower_bound, float), type(lower_bound)
        assert isinstance(upper_bound, float), type(upper_bound)

        # outliers = self.data[(self.data <= lower_bound) | (self.data >= upper_bound)]
        lower_outliers = self.data[self.data <= lower_bound]
        upper_outliers = self.data[self.data >= upper_bound]
        lower_outliers[::-1].sort()
        upper_outliers.sort()

        return np.unique(lower_outliers), np.unique(upper_outliers)
