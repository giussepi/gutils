# -*- coding: utf-8 -*-
""" gutils/plots/test/test_tables """

import unittest

import matplotlib.colors as mcolors
from matplotlib.figure import Figure

from gutils.plots.tables import plot_color_table


class Test_plot_color_table(unittest.TestCase):

    def test_function(self):
        self.assertTrue(
            isinstance(plot_color_table(mcolors.BASE_COLORS, 'BASE COLOURS'), Figure)
        )


if __name__ == '__main__':
    unittest.main()
