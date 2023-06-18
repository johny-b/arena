# %%
import json
import numpy as np
import torch as t
import pickle
from collections import defaultdict
import gc

from procgen_tools import maze, imports, models

device = t.device("cpu")
# %%
for mazes_ix in range(1, 101):
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

        for layer_name, val in hook.values_by_label.items():
            if any(x in layer_name for x in ("resadd_out", "maxpool_out", "relu3_out", "relufc_out")):
                data[layer_name].append((seed, layer_name, action, cheese, val))
        
        #   Some of this stuff reduces memory leak (although doesn't fix it fully)
        del policy
        del hook
        del venv
        gc.collect()
    
    for key, val in data.items():
        out_fname = f'/home/janbet/arena/activations/mazes_{mazes_ix}_{key}.pickle'
        with open(out_fname, 'wb') as f:
            pickle.dump(val, f, protocol=pickle.HIGHEST_PROTOCOL)
            
    
# %%
