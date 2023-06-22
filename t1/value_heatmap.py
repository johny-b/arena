# %%
import numpy as np
import torch as t
 
import os
os.chdir('/home/janbet/arena/t1')

from procgen_tools.imports import load_model
from procgen_tools import maze, visualization
from procgen_tools.models import human_readable_action, human_readable_actions, MAZE_ACTION_DELTAS
from procgen_tools.rollout_utils import rollout_video_clip, get_predict
device = t.device("cpu")

# %%
policy, hook = load_model()

# %%
venv = maze.create_venv(num=1, start_level=0, num_levels=1)

# grid = maze.state_from_venv(venv).inner_grid()
# grid[grid == 2] = 100
# venv = maze.venv_from_grid(grid)

visualization.visualize_venv(venv, render_padding=False) 

# %%
vf = visualization.vector_field(venv, policy)
visualization.plot_vf(vf, ax=None)

# %%
venv_all, (legal_mouse_positions, grid) = maze.venv_with_all_mouse_positions(venv)
with t.no_grad():
    actions, values = policy(t.tensor(venv_all.reset(), dtype=t.float32))

out_data = np.empty_like(grid, dtype=float)
out_data[:, :] = np.nan

for mouse_pos, value in zip(legal_mouse_positions, values.numpy()):
    out_data[-1 * mouse_pos[0] - 1, mouse_pos[1]] = value
    # print("POS", mouse_pos, "VAL", value)



# print(out_data)
import matplotlib.pyplot as plt
import numpy as np


# out_data[out_data == 0] = values.min() - 0.5

import matplotlib
my_cmap = matplotlib.cm.get_cmap('hot')
my_cmap.set_bad('green')

a = np.random.random((16, 16))
ax = plt.gca()
x = plt.imshow(out_data, cmap=my_cmap, interpolation='nearest')
cbar = ax.figure.colorbar(x, ax=ax)


plt.show()

# %%
