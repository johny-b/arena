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
venv = maze.create_venv(num=1, start_level=20, num_levels=1)
visualization.visualize_venv(venv, render_padding=False) 
# %%
# SEED = 84451
for SEED in range(16, 30):
    MAX_STEPS = 20
    print("SEED", SEED)

    venv = maze.create_venv(num=1, start_level=SEED, num_levels=1)
    grid = maze.state_from_venv(venv).inner_grid()
    # grid[1][2] = 51
    # grid[3][4] = 51
    # print(grid)
    venv = maze.venv_from_grid(grid)
    visualization.visualize_venv(venv, render_padding=False) 

    def hr_logits(logits):
        return [
            logits[0, :3].mean().item(),
            logits[0, 3].item(),
            logits[0, 5].item(),
            logits[0, 6:9].mean().item(),
        ]

    obs = venv.reset()
    for i in range(MAX_STEPS):
        # visualization.visualize_venv(venv, render_padding=False) 

        obs = t.tensor(obs, dtype=t.float32)
        current_pos = maze.state_from_venv(venv).mouse_pos
        
        with t.no_grad():
            categorical, value = policy(obs)
        value = round(value.item(), 3)
        # print("\n", categorical.logits)
        # p = t.nn.functional.softmax(categorical.logits, dim=1)
        # print(p)
        # print(hr_logits(categorical.logits))
        
        action = categorical.logits.argmax().unsqueeze(0).numpy()
        hr_action = human_readable_action(action)
        
        print(f"CURRENT POS {current_pos} VALUE {value} ACTION {hr_action}")
        
        obs, rewards, dones, infos = venv.step(action)
        if dones[0]:
            print("GOT CHEESE")
            break
# %%

visualization.visualize_venv(venv, render_padding=False) 

# %%