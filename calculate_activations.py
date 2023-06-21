# %%
import json
import numpy as np
import torch as t

from collections import defaultdict
import gc

from procgen_tools import maze, imports, models

device = t.device("cpu")
# %%

def next_step_to_cheese(grid):
    grid = np.array(grid)
    graph = maze.maze_grid_to_graph(grid)
    venv = maze.venv_from_grid(grid)
    mr, mc = maze.state_from_venv(venv).mouse_pos
    padding = maze.get_padding(grid)
    mr_inner, mc_inner = mr - padding, mc - padding                 
    path_to_cheese = maze.get_path_to_cheese(grid, graph, (mr_inner, mc_inner))
    next_step_x, next_step_y = path_to_cheese[1]

    next_step_x, next_step_y = next_step_x + padding, next_step_y + padding
    
    diff = (next_step_x - mr, next_step_y - mc)
    action = next(key for key, val in models.MAZE_ACTION_DELTAS.items() if val == diff)
    
    return action
    
# %%
for mazes_ix in range(1, 11):
    in_fname = f'/home/janbet/arena/data/mazes_{mazes_ix}.json'
    with open(in_fname, 'r') as f:
        grids = json.load(f)

    data = defaultdict(list)
    for ix, (seed, grid) in enumerate(grids):
        print(mazes_ix, ix)
        venv = maze.venv_from_grid(np.array(grid))
        policy, hook = imports.load_model()

        batched_obs = t.tensor(venv.reset(), dtype=t.float32, device=device).numpy()
        with t.no_grad():
            hook.run_with_input(batched_obs)
            
        #   Where the mouse will go?
        categorical, _ = hook.values_by_label['_out']
        action = models.human_readable_action(categorical.logits.argmax())
        
        #   Where is the cheese?
        cheese = next(zip(*(np.where(np.array(grid) == 25))))
        
        #   Are we going to the cheese?
        action_to_cheese = next_step_to_cheese(grid)

        for layer_name, val in hook.values_by_label.items():
            if layer_name == '_out':
                continue
            data = (seed, layer_name, action, action_to_cheese, ' '.join([str(x) for x in cheese]), val.tolist())
            data = ' '.join([str(x) for x in data])

            out_fname = f'/home/janbet/arena/activations_2/mazes_{seed}_{layer_name}.txt'
            with open(out_fname, 'w') as f:
                f.write(data)
            
        #   Some of this stuff reduces memory leak (although doesn't fix it fully)
        del policy
        del hook
        del venv
        gc.collect()
    
    
# %%
