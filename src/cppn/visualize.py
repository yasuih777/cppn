# !/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np


def canvas_heatmap(map_vector: np.ndarray, axes: plt.Axes) -> plt.Axes:
    axes.imshow(map_vector, cmap="Greys")
    axes.axis("off")

    return axes


def create_figure(**args) -> tuple[plt.Figure, plt.Axes]:
    fig, axes = plt.subplots(**args)

    return fig, axes
