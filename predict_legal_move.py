# %%
import numpy as np
import torch as t
 
import os

from procgen_tools.imports import load_model
from procgen_tools import maze, visualization
from procgen_tools.models import human_readable_action, human_readable_actions, MAZE_ACTION_DELTAS
from procgen_tools.rollout_utils import rollout_video_clip, get_predict
device = t.device("cpu")

from train_predict_legal_move import LegalActionProbe

# %%
policy, hook = load_model()

# %%
probe = LegalActionProbe()
state_dict = t.load('/root/arena/probe_last_layer_1000_9.pth')
state_dict = {
    'linear.weight': state_dict["model.linear.weight"],
    'linear.bias': state_dict["model.linear.bias"],
}
probe.load_state_dict(state_dict)

# %%
# venv = maze.create_venv(num=1, start_level=20, num_levels=1)
# visualization.visualize_venv(venv, render_padding=False) 
# %%
# UP, RIGHT, LEFT, DOWN
SEED = 84452
MAX_STEPS = 5

venv = maze.create_venv(num=1, start_level=SEED, num_levels=1)
visualization.visualize_venv(venv, render_padding=False)

obs = venv.reset()
for i in range(MAX_STEPS):
    # visualization.visualize_venv(venv, render_padding=False) 
    print()
    print("POS", maze.state_from_venv(venv).mouse_pos)
    
    obs = t.tensor(obs, dtype=t.float32)
    current_pos = maze.state_from_venv(venv).mouse_pos
    
    with t.no_grad():
        categorical, _ = policy(obs)
        print(probe(categorical.logits).round(decimals=2))

    action = categorical.logits.argmax().unsqueeze(0).numpy()
    
    obs, rewards, dones, infos = venv.step(action)
    if dones[0]:
        print("GOT CHEESE")
        break

# %%

visualization.visualize_venv(venv, render_padding=False) 

# %%