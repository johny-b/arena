# %%
import pickle
import os
from typing import List
import json

import numpy as np
from torch.utils.data import Dataset
# %%
class ActivationsDataset(Dataset):
    def __init__(self, layer_name: str, file_ids: List[int]):
        self.layer_name = layer_name
        self.file_ids = file_ids
        self.act_files = self._get_act_files()
        self.maze_files = self._get_maze_files()
        
        self._current_act_data = None
        self._current_maze_data = None
        self._current_file_ix = None
        
    def __len__(self):
        return len(self.file_ids) * 1000
    
    def __getitem__(self, idx):
        file_ix = idx // 1000
        sample_ix = idx % 1000
        
        if self._current_file_ix is None or self._current_file_ix != file_ix:
            self._open_files(file_ix)
        
        act_data = self._current_act_data[sample_ix]
        maze_data = self._current_maze_data[sample_ix]

        assert act_data[0] == maze_data[0]
        
        
        act = act_data[2]
        grid = maze_data[1]
        
        next_dir = self._get_next_mouse_dir(grid)
        cheese_coord = self._get_cheese_coord(grid)
        
        return next_dir, cheese_coord, act
    
    def _get_mouse_dir(self, grid):
        return np.random.randint(4)
    
    def _get_cheese_coord(self, grid):
        return (np.random.randint(19), np.random.randint(19))
        
    def _open_files(self, file_ix):
        with open(self.act_files[file_ix], 'rb') as f:
            self._current_act_data = pickle.load(f)
        
        with open(self.maze_files[file_ix], 'rb') as f:
            self._current_maze_data = json.load(f)
        
        self._current_file_ix = file_ix
            
    def _get_act_files(self):
        fnames = []
        for id_ in self.file_ids:
            fname = f'/home/janbet/arena/activations/mazes_{id_}_{self.layer_name}.pickle'
            if not os.path.isfile(fname):
                raise ValueError(f"File {fname} doesn't exist")
            fnames.append(fname)
        return fnames
        
    def _get_maze_files(self):
        fnames = []
        for id_ in self.file_ids:
            fname = f'/home/janbet/arena/data/mazes_{id_}.json'
            if not os.path.isfile(fname):
                raise ValueError(f"File {fname} doesn't exist")
            fnames.append(fname)
        return fnames
    
    
# %%
dataset = ActivationsDataset('embedder.relufc_out', [1,2,3])
        
# %%
