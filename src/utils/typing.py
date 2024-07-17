# !/usr/bin/env python3

from typing import TypedDict

import numpy as np


class InputNode(TypedDict, total=False):
    x_point: int
    y_point: int
    radius: int
    time: int


class ActiveNode(TypedDict):
    linear: int
    abs: int
    sin: int
    cos: int
    tan: int
    sign: int
    sigmoid: int
    gauss: int


class SingleLayer(TypedDict):
    name: str
    nodes: InputNode | ActiveNode
    edges: np.ndarray | None
