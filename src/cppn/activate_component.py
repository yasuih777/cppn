# !/usr/bin/env python3

import numpy as np

from src.utils import typing


def activate(name: str, vector: np.ndarray) -> np.ndarray:
    if name == "linear":
        return vector
    if name == "abs":
        return np.abs(vector)
    if name == "sin":
        return np.sin(vector)
    if name == "cos":
        return np.cos(vector)
    if name == "tan":
        return np.tan(vector)
    if name == "sign":
        return np.sign(vector)
    if name == "sigmoid":
        return 1.0 / (1.0 + np.exp(-vector))
    if name == "gauss":
        return np.exp(-((vector) ** 2) / 2)
    else:
        raise ValueError(f"{name} is undifined activate key")


def layer_transform(layer: typing.SingleLayer, vector: np.ndarray) -> np.ndarray:
    vector = vector @ layer["edges"]
    reduce_node = {
        name: number for name, number in layer["nodes"].items() if number > 0
    }

    vector_split = np.hsplit(vector.copy(), np.cumsum(list(reduce_node.values()))[:-1])
    vector_split = [
        activate(name, vec) for name, vec in zip(reduce_node.keys(), vector_split)
    ]

    return np.hstack(vector_split)
