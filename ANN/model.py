import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))


import numpy as np
from activation import Activations, ACTIVATION_REGISTRY

x = np.array([-2.0, -0.5, 0.0, 1.0, 3.0])

y_relu = Activations.relu(x)
y_swish = Activations.swish(x)

act_fn = ACTIVATION_REGISTRY["gelu"]
y_gelu = act_fn(x)