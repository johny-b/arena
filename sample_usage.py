# %%:
import numpy as np
import torch as t

from procgen_tools.imports import load_model
from procgen_tools import maze, visualization
from procgen_tools.models import human_readable_action, human_readable_actions, MAZE_ACTION_DELTAS

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
    
print(human_readable_action(categorical.logits.argmax()))
print(human_readable_actions(categorical.logits))
print(value)

# %%
# Perform a rollout (doesn't stop on cheese)
max_steps = 10

for i in range(max_steps):

    state = maze.state_from_venv(venv)
    current_pos = state.mouse_pos
    
    batched_obs = t.tensor(venv.reset(), dtype=t.float32, device=device)
    with t.no_grad():
        categorical, _ = policy(batched_obs)
    
    action = human_readable_action(categorical.logits.argmax())
    step = MAZE_ACTION_DELTAS[action]
    new_pos = (current_pos[0] + step[0], current_pos[1] + step[1])
    state.set_mouse_pos(*new_pos)
    venv = maze.venv_from_grid(state.inner_grid())
    print(f"MOVE FROM {current_pos} to {new_pos}")

    visualization.visualize_venv(venv, render_padding=False)