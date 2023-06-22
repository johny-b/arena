# %%:
import numpy as np
import torch as t
 
import os
os.chdir('/home/janbet/arena/t1')

from procgen_tools.imports import load_model
from procgen_tools import maze, visualization
from procgen_tools.models import human_readable_action, human_readable_actions, MAZE_ACTION_DELTAS
from procgen_tools.rollout_utils import rollout_video_clip, get_predict

# %%
policy, hook = load_model()

# %%
SEED=84451

def get_logits(obs: np.array):
    with t.no_grad():
        categorical, _ = policy(t.tensor(obs, dtype=t.float32))
        return categorical.logits

venv = maze.create_venv(num=1, start_level=SEED, num_levels=1)

# print(maze.state_from_venv(venv).inner_grid())
#   Determine the first action
first_obs = venv.reset()
logits = get_logits(first_obs)
action = logits.argmax().unsqueeze(0).numpy()

#   Make step via venv_1.step()
new_obs, *_ = venv.step(action)
print(maze.state_from_venv(venv).inner_grid())

#   Copy grid, create new venv
state = maze.state_from_venv(venv)
grid = state.inner_grid()
new_venv = maze.venv_from_grid(grid)
print(maze.state_from_venv(new_venv).inner_grid())
#   Just to make sure
assert np.array_equal(
    grid,
    maze.state_from_venv(new_venv).inner_grid(),
)

print(new_obs.mean(), new_venv.reset().mean())
    
# print(get_logits(new_obs))
# print(get_logits(venv.reset()))
# print(get_logits(new_venv.reset()))
# %%
import matplotlib.pyplot as plt

print("FIRST OBS")
plt.imshow(t.from_numpy(first_obs[0]).permute(1, 2, 0))
# %%
print("OLD VENV")
plt.imshow(t.from_numpy(new_obs[0]).permute(1, 2, 0))
# %%
print("NEW VENV")
plt.imshow(t.from_numpy(new_venv.reset()[0]).permute(1, 2, 0))
# %%
