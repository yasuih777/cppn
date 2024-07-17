# !/usr/bin/env python3# !/usr/bin/env python3

import platform
from inspect import get_annotations

import pkg_resources
import streamlit as st

from src.cppn import visualize
from src.cppn.layer_factory import LayerFactory
from src.cppn.model import CPPNModeler
from src.utils import logging, typing, utils


class AppBuilder:
    def __init__(self) -> None:
        self.logger = logging.set_logger("warning")

        self.layer_factory: LayerFactory
        self.modeler: CPPNModeler

        self.flag: bool = False

    def __call__(self) -> None:
        st.set_page_config(
            page_title="CPPN モデレーター",
            layout="wide",
            initial_sidebar_state="collapsed",
        )
        self.sidebar_component()
        self.body_components()

    def sidebar_component(self) -> None:
        st.sidebar.subheader("リンク")
        st_col = st.sidebar.columns(2)
        with st_col[0]:
            st.link_button("X(Twitter)", "https://twitter.com/yugetsubiostat")
        with st_col[1]:
            st.link_button(
                "Github", "https://github.com/yasuih777/simulation_significant_test"
            )

        st.sidebar.subheader("使い方")
        st.sidebar.markdown("")

        st.sidebar.subheader("Licence")
        st.sidebar.link_button("MIT Licence", "https://opensource.org/license/mit/")

        with st.sidebar.expander("Python environment"):
            st.text(f"Python version: {platform.python_version()}")
            st.text(f"Python pkgs:")
            for pkg in pkg_resources.working_set:
                st.text(pkg)

    def body_components(self) -> None:
        st.title("CPPNモデレーター")

        body = st.columns(2)
        with body[0]:
            st.header("1 CPPNモデルの定義")

            st.subheader("1.1 座標系の設定")
            with st.container(border=True):
                sparam_body = st.columns(3)
                with sparam_body[0]:
                    scale = st.number_input(
                        "空間座標の境界値",
                        min_value=1e-10,
                        value=1.0,
                        step=1.0,
                        key="space",
                    )
                with sparam_body[1]:
                    use_cycle = st.checkbox("周期的な模様を作成", key="is-cycle")
                    if use_cycle:
                        cycle = st.number_input(
                            "周期数", min_value=0.0, value=4.0, step=1.0, key="cycle"
                        )
                    else:
                        cycle = 0
                with sparam_body[2]:
                    fix_seed = st.checkbox("乱数を固定", key="is_seed")
                    if fix_seed:
                        seed = st.number_input(
                            "乱数シード", min_value=0, value=42, step=1, key="seed"
                        )
                    else:
                        seed = None

            self.layer_factory = LayerFactory(seed=seed)

            st.subheader("1.2 ネットワークグラフの設定")
            with st.container(border=True):
                layer_num = st.number_input("中間層の数", min_value=0, value=3, step=1)
                nparam_tab = st.tabs(
                    ["入力層"]
                    + [f"中間層{idx}" for idx in range(1, layer_num + 1)]
                    + ["出力層"]
                )

                with nparam_tab[0]:
                    is_radius, _ = self.input_layer_param()

                self.layer_factory.is_radius = is_radius
                self.layer_factory.layers.append(
                    self.layer_factory.insert_node("input_layer", input=True)
                )

                for idx, tab in enumerate(nparam_tab[1:-1]):
                    with tab:
                        node = self.activate_layer_param(idx)

                    self.layer_factory.layers.append(
                        self.layer_factory.insert_node(
                            f"hidden_layer_{idx + 1}", node=node
                        )
                    )

                with nparam_tab[-1]:
                    node = self.activate_layer_param(len(nparam_tab), output=True)

                self.layer_factory.layers.append(
                    self.layer_factory.insert_node("output_layer", node=node)
                )

            with body[1]:
                st.header("2 CPPNモデルによる描画")
                go_canvas = st.button("描画を開始", key="is_canvas")
                if go_canvas:
                    # make canvas
                    self.layer_factory.create_layer()

                    self.modeler = CPPNModeler(
                        scale, cycle, self.layer_factory.layers, is_radius=is_radius
                    )
                    self.modeler.vec2canvas()

                    fig, axes = visualize.create_figure(
                        nrows=1, ncols=1, figsize=(2.75, 2.75), tight_layout=True
                    )
                    visualize.canvas_heatmap(self.modeler.canvas, axes)
                    st.pyplot(fig, use_container_width=False)

    @staticmethod
    def input_layer_param() -> tuple[bool, bool]:
        lparam_body = st.columns(2)
        with lparam_body[0]:
            is_radius = st.checkbox("中心性を追加", key="is-radius")
        # with lparam_body[1]:
        #     is_time = st.checkbox("時間位相を加える")

        return is_radius, False

    def activate_layer_param(
        self, i_layer: int, output: bool = False, split: int = 4
    ) -> typing.ActiveNode:
        names = list(get_annotations(typing.ActiveNode).keys())
        node: typing.ActiveNode = {name: 0 for name in names}

        if output:
            name = st.radio(
                "出力層に使用するノードを選択",
                options=names,
                horizontal=True,
                index=names.index("sign"),
                key=f"output_layer",
            )
            node[name] = 1

        else:
            node["tan"] = 3
            for split_names in utils.split_container(names, split):
                lparam_body = st.columns(split)

                for idx, tab in enumerate(lparam_body):
                    with tab:
                        try:
                            name = split_names[idx]
                            node[name] = st.number_input(
                                f"{name}ノード数",
                                min_value=0,
                                value=node[name],
                                step=1,
                                key=f"{name}-{i_layer}",
                            )
                        except:
                            pass

        return node
