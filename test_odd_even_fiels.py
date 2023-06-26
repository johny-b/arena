# %%
import numpy as np
import torch as t
from torch import nn
import os

from procgen_tools.imports import load_model
from procgen_tools import maze, visualization
from procgen_tools.models import human_readable_action, human_readable_actions, MAZE_ACTION_DELTAS
from procgen_tools.rollout_utils import rollout_video_clip, get_predict
device = t.device("cuda")

# %%

policy, hook = load_model()

# %%
MAZE_SIZE = 25

all_squares = [(x, y) for x in range(MAZE_SIZE) for y in range(MAZE_SIZE)]
even_odd_squares = [square for square in all_squares if not (square[0] % 2) and (square[1] % 2)]
odd_even_squares = [square for square in all_squares if (square[0] % 2) and not (square[1] % 2)]

# %%
def get_seed(maze_size: int) -> int:
    """Returns a random seed that creates maze of a given size."""
    while True:
        seed = np.random.randint(0, 100000000)
        if maze.get_inner_grid_from_seed(seed=seed).shape[0] == maze_size:
            return seed

def get_activations_sum(seed, layer_name, positions):
    """Returns mean activation absolute value per channel, for given layer_name and mouse positions"""
    venv = maze.create_venv(num=1, start_level=seed, num_levels=1)
    venv_all, (legal_mouse_positions, _) = maze.venv_with_all_mouse_positions(venv)
    with t.no_grad():
        hook.run_with_input(venv_all.reset().astype('float32'))
        
    raw_activations = hook.values_by_label[layer_name]
    assert len(raw_activations.shape) == 4, "This layer has wrong shape"
    activations = raw_activations.abs().sum(dim=-1).sum(dim=-1)

    data = []
    for mouse_pos, activation in zip(legal_mouse_positions, activations):
        if mouse_pos in positions:
            data.append(activation)
            
    data = t.stack(data)
    data = data.mean(dim=0)
    
    return data

# %%

seed = get_seed(MAZE_SIZE)
layer_name = "embedder.relu3_out"
even_odd = get_activations_sum(seed, layer_name, even_odd_squares)
odd_even = get_activations_sum(seed, layer_name, odd_even_squares)
diff = (even_odd - odd_even)

even_odd_high = diff.topk(5)
odd_even_high = diff.topk(5, largest=False)
print(f"(Even, odd) squares have high values in channels {even_odd_high.indices.tolist()}, diff {even_odd_high.values.round().tolist()}")
print(f"(Odd, even) squares have high values in channels {odd_even_high.indices.tolist()}, diff {odd_even_high.values.round().tolist()}")


# %%
