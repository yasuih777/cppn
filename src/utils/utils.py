# !/usr/bin/env python3

import random
from typing import Generator, Optional

import numpy as np


def set_seed(seed: Optional[int]) -> None:
    random.seed(seed)
    np.random.seed(seed)


def split_container(containers: list, split: int = 5) -> Generator[list, None, None]:
    for idx in range(0, len(containers), split):
        yield containers[idx : idx + split]
