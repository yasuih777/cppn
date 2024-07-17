# !/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def norm_uint8(vector: np.ndarray) -> np.ndarray:
    min_value = np.min(vector)
    max_value = np.max(vector)

    return (255 * (vector - min_value) / (max_value - min_value)).astype(np.uint8)


def canvas_heatmap(map_vectors: list[np.ndarray], axes: plt.Axes) -> plt.Axes:
    map_vectors = [norm_uint8(vector) for vector in map_vectors]
    if len(map_vectors) == 1:
        image = Image.fromarray(map_vectors[0])
        cmap = "Greys"
    else:
        map_vector = np.dstack(map_vectors)
        image = Image.fromarray(map_vector)
        cmap = None

    axes.imshow(image, cmap=cmap)
    axes.axis("off")

    return axes


def create_figure(**args) -> tuple[plt.Figure, plt.Axes]:
    fig, axes = plt.subplots(**args)

    return fig, axes
