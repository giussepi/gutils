# -*- coding: utf-8 -*-
""" gutils/plot/tables """

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def plot_color_table(colors, title, sort_colors=True, emptycols=0):
    """
    Plots a table of colours

    Source: https://matplotlib.org/stable/gallery/color/named_colors.html

    Args:
        colors      <dict>: Dictionary with colour names and rgba values as keys and
                            values respectively. See matplotlib.colors.BASE_COLORS as
                            reference
        title        <str>: Table title
        sort_colors <bool>: Whether or not sort colors by hue, saturation, value and name.
                            Default True
        emptycols    <int>: Number of empty columns (columns number = 4 - emptycols).
                            Default 0

    Usage:
        import matplotlib.colors as mcolors

        plot_colortable(mcolors.BASE_COLORS, "Base Colors", sort_colors=False, emptycols=1)
        plt.show()
    """
    assert isinstance(colors, dict), type(colors)
    assert isinstance(title, str), type(str)
    assert isinstance(sort_colors, bool), type(sort_colors)
    assert isinstance(emptycols, int), type(int)
    assert 0 <= emptycols < 4

    cell_width = 212
    cell_height = 22
    swatch_width = 48
    margin = 12
    topmargin = 40

    if sort_colors is True:
        by_hsv = sorted((tuple(mcolors.rgb_to_hsv(mcolors.to_rgb(color))),
                         name)
                        for name, color in colors.items())
        names = [name for hsv, name in by_hsv]
    else:
        names = list(colors)

    n = len(names)
    ncols = 4 - emptycols
    nrows = n // ncols + int(n % ncols > 0)

    width = cell_width * 4 + 2 * margin
    height = cell_height * nrows + margin + topmargin
    dpi = 72

    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
    fig.subplots_adjust(margin/width, margin/height,
                        (width-margin)/width, (height-topmargin)/height)
    ax.set_xlim(0, cell_width * 4)
    ax.set_ylim(cell_height * (nrows-0.5), -cell_height/2.)
    ax.yaxis.set_visible(False)
    ax.xaxis.set_visible(False)
    ax.set_axis_off()
    ax.set_title(title, fontsize=24, loc="left", pad=10)

    for i, name in enumerate(names):
        row = i % nrows
        col = i // nrows
        y = row * cell_height

        swatch_start_x = cell_width * col
        text_pos_x = cell_width * col + swatch_width + 7

        ax.text(text_pos_x, y, name, fontsize=14,
                horizontalalignment='left',
                verticalalignment='center')

        ax.add_patch(
            Rectangle(xy=(swatch_start_x, y-9), width=swatch_width,
                      height=18, facecolor=colors[name], edgecolor='0.7')
        )

    return fig
