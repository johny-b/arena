# %%:
import numpy as np
import torch as t

from procgen_tools.imports import load_model
from procgen_tools import maze, visualization

device = t.device("cpu")

# %%
#   Load the model. This function has some arguments that might be useful,
#   but I don't know yet what they are doing.
#
#   Policy is a procgen_tools.models.CategoricalPolicy (extends nn.Module), our model.
#   Hook gathers activations.
policy, hook = load_model(num_actions=15)

# %%
#   Create a virtual environment
def get_maze_venv(size):
    seed = np.random.randint(0, 100000)
    while maze.get_inner_grid_from_seed(seed=seed).shape[0] != size:
        seed = np.random.randint(0, 100000)
    venv = maze.create_venv(num=1, start_level=seed, num_levels=1)
    return venv

venv = get_maze_venv(size=5)

#   Print a maze
#   100 - corridor
#    51 - wall
#    25 - mouse
#     2 - cheese
state = maze.state_from_venv(venv)
print(state.inner_grid())
visualization.visualize_venv(venv, render_padding=False)



#   Run the model on the maze. Return logist of the 15 actions.
#   Q: why 15?? I guessed there are 5 (left/right/up/down/no-action)
#   A: I've no idea ;( I'm gonna dig into this.

batched_obs = t.tensor(venv.reset(), dtype=t.float32, device=device)
with t.no_grad():
    categorical, value = policy(batched_obs)
    
print(categorical.logits)
print(value)

# %%
