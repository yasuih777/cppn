# !/usr/bin/env python3

import numpy as np

from src.utils import typing


class LayerFactory:
    def __init__(self, init_node_n: int = 3, seed: int | None = None) -> None:
        self.layers: list[typing.SingleLayer] = []
        self.rng = np.random.default_rng(seed)

        # input layer
        self.is_radius: bool = False
        self.is_time: bool = False

        # active layer
        self.init_node_n: int = init_node_n

    def create_layers_init(self, layer_num: int) -> None:
        self.layers.append(
            {"name": "input_layer", "nodes": self.set_input_node(), "edges": None}
        )
        for number in range(1, layer_num + 1):
            self.layers.append(
                {
                    "name": f"hidden_layer_{number}",
                    "nodes": self.set_activate_node(),
                    "edges": None,
                }
            )

        self.layers.append(
            {
                "name": "output_layer",
                "nodes": self.set_activate_node(output=True),
                "edges": None,
            }
        )

    def create_layer(self) -> None:
        edge_shapes: list[tuple[int, int]] = []
        for from_layer, to_layer in zip(self.layers[:-1], self.layers[1:]):
            edge_shapes.append(
                (
                    self.get_node_n(from_layer["nodes"]),
                    self.get_node_n(to_layer["nodes"]),
                )
            )

        for layer, edge_shape in zip(self.layers[1:], edge_shapes):
            layer["edges"] = 2 * self.rng.random(size=edge_shape) - 1

    @staticmethod
    def get_node_n(nodes: typing.InputNode | typing.ActiveNode) -> tuple[int, int]:
        return np.sum([value for value in nodes.values()])

    def insert_node(
        self, name: str, node: typing.ActiveNode | None = None, input: bool = False
    ) -> typing.SingleLayer:
        if input:
            node = self.set_input_node

        return {"name": name, "nodes": node, "edges": None}

    @property
    def set_input_node(self) -> typing.InputNode:
        node: typing.InputNode = {"x_point": 1, "y_point": 1}
        if self.is_radius:
            node.update({"radius": 1})
        if self.is_time:
            node.update({"time": 1})

        return node
