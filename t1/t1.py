# %%:
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
#   Load the model. This function has some arguments that might be useful,
#   but I don't know yet what they are doing.
#
#   Policy is a procgen_tools.models.CategoricalPolicy (extends nn.Module), our model.
#   Hook gathers activations.
policy, hook = load_model()


# %%
#   Create a virtual environment
def get_maze_venv(size, seed=None):
    if seed is None:
        seed = np.random.randint(0, 100000)
        while maze.get_inner_grid_from_seed(seed=seed).shape[0] != size:
            seed = np.random.randint(0, 100000)
    venv = maze.create_venv(num=1, start_level=seed, num_levels=1)
    return seed, venv

def get_decision_square_maze_venv(size, seed=None):   
    assert size % 2, "size must be an odd number"                               
                                                                                
    while True:
        if seed is None:                                                                 
            seed = np.random.randint(100000000)                                     
            while maze.get_inner_grid_from_seed(seed=seed).shape[0] != size:        
                seed = np.random.randint(100000000)                                 
                                                                                
        venv = maze.create_venv(num=1, start_level=seed, num_levels=1)          
        state_bytes = venv.env.callmethod("get_state")[0]                       
                                                                                
        if seed is not None or maze.maze_has_decision_square(state_bytes):                          
            return seed, venv

# %%
MAZE_SIZE=9
SEED=84451
MAX_STEPS=30


for i in range(1):
    seed, venv = get_decision_square_maze_venv(size=MAZE_SIZE, seed=SEED)

    print("SEED", seed)
    visualization.visualize_venv(venv, render_padding=False)

    state = maze.state_from_venv(venv)
    outer_grid = state.full_grid()
    cheese = next(zip(*(np.where(np.array(outer_grid) == 2))))

    print("CHEESE", cheese)
    print("MOUSE", state.mouse_pos)
    visited = set()
    
    obs = t.tensor(venv.reset(), dtype=t.float32)
    for i in range(MAX_STEPS):

        state = maze.state_from_venv(venv)
        current_pos = state.mouse_pos
        
        with t.no_grad():
            categorical, value = policy(obs)
        value = round(value.item(), 3)
        
        val = categorical.logits.argmax().unsqueeze(0).numpy()
        action = human_readable_action(val[0])
        new_obs, *_ = venv.step(val)
        obs = t.from_numpy(new_obs).to(t.float32)
        
        new_grid = maze.state_from_venv(venv).inner_grid()
        copied_venv = maze.venv_from_grid(new_grid)
        assert np.array_equal(
            new_grid,
            maze.state_from_venv(copied_venv).inner_grid(),
        )
        new_obs = t.tensor(copied_venv.reset(), dtype=t.float32)
        with t.no_grad():
            new_categorical, _ = policy(new_obs)
        
        print(f"Current pos {current_pos} has value {value}, action {action}")
        print("OLD", categorical.logits)
        print("NEW", new_categorical.logits)

        if current_pos == cheese:
            print("GOT CHEESE")
            break
        if current_pos not in visited:
            visited.add(current_pos)
        else:
            print("REPEATED POS")
            break
# %%
visualization.visualize_venv(venv, render_padding=False) 

# %%
predict = get_predict(policy)
x = rollout_video_clip(predict, 12)

x[1].write_videofile('/home/janbet/Desktop/t1.mp4')
# %%
