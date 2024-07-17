# !/usr/bin/env python3

from itertools import product

import numpy as np

from src.cppn.activate_component import layer_transform
from src.cppn.layer_factory import LayerFactory


class CPPNModeler:
    def __init__(
        self,
        scale: float,
        cycle: float,
        layer_factoty: LayerFactory,
        length: int = 512,
        is_radius: bool = False,
        is_time: bool = False,
        is_color: bool = False,
    ) -> None:
        self.length = length
        self.space: tuple[float, float] = (-scale, scale)
        self.cycle = cycle
        self.is_radius = is_radius
        self.is_time = is_time
        self.color: int = 3 if is_color else 1

        self.layer_factory = layer_factoty

        self.canvas: list[np.ndarray] = []

    def vec2canvas(self) -> None:
        for _ in range(self.color):
            self.layer_factory.create_layer()
            output_vector = self.layer_propagation()
            self.canvas.append(output_vector.reshape(self.length, self.length))

    def canvas2vec(self) -> np.ndarray:
        space = np.linspace(self.space[0], self.space[1], self.length)
        space_point = space.copy()
        if self.cycle != 0:
            space = np.sin(np.pi * self.cycle * space / 2)
        input_vec = np.array(list(product(space, space)), dtype=np.float32)

        if self.is_radius:
            point_vec = np.array(
                list(product(space_point, space_point)), dtype=np.float32
            )
            input_vec = np.hstack(
                (input_vec, np.linalg.norm(point_vec, ord=2, axis=1).reshape(-1, 1)),
                dtype=np.float32,
            )

        return input_vec

    def layer_propagation(self) -> np.ndarray:
        x_vector = self.canvas2vec()
        for layer in self.layer_factory.layers[1:]:
            x_vector = layer_transform(layer, x_vector)

        return x_vector
