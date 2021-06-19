"""
Holds some per-process state, such as Random Number Generators.
"""

import numpy as np
import torch
from pathlib import Path


rng: np.random.Generator = np.random.default_rng()
torch_device = torch.device("cpu")
carla_host = "localhost"
carla_world_port = 2000
num_torch_threads = 1

# Episode collection settings:
#: Number of episodes to collect
num_episodes: int = 1

#: Episode data will be stored here
data_path = Path()

#: On running an episode, the town and weather are chosen randomly from these lists.
towns = ["Town01"]
weathers = ["ClearNoon"]


def init():
    torch.set_num_threads(num_torch_threads)
