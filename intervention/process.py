"""
Holds some per-process state, such as Random Number Generators.
"""

import numpy as np
import torch


rng = np.random.default_rng()
torch_device = torch.device("cpu")
