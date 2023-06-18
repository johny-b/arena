# %%
import pickle
import os
from typing import List
import json
import pandas as pd

import numpy as np
import torch as t
from torch.utils.data import Dataset
from torch import nn

import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger

device = t.device("cuda")
MAIN = __name__ == '__main__'
# %%
class ActivationsDataset(Dataset):
    def __init__(self, layer_name: str, file_ids: List[int]):
        self.layer_name = layer_name
        self.file_ids = file_ids
        self.act_files = self._get_act_files()
        self.maze_files = self._get_maze_files()
        
        self._action_map = ["LEFT", "UP", "RIGHT", "DOWN", "NOOP"]
        
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
        
        seed, layer, action, cheese, act = act_data
        assert layer == self.layer_name
        action_ix = self._action_map.index(action)
        
        #   TODO what about cheese?
        return act, action_ix
    
    def _get_next_mouse_dir(self, grid):
        return np.random.randint(4)
    
    def _get_cheese_coord(self, grid):
        vals = np.where(np.array(grid) == 25)
        return vals[0][0], vals[1][0]
        
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

class NextActionProbe(nn.Module):
    def __init__(self, layer_name):
        super().__init__()
        self.layer_name = layer_name
        self._in_size = self._calc_in_size()
        self._out_size = 4
        self.linear = nn.Linear(self._in_size, self._out_size)
        
    def forward(self, x):
        return self.linear(x.flatten(start_dim=1))
        
    def _calc_in_size(self):
        dataset = ActivationsDataset(self.layer_name, [1])
        sample_input = dataset[0][0]
        return sample_input.flatten().shape[0]


# %%
class LitModel(pl.LightningModule):
    def __init__(self, model, batch_size: int, max_epochs: int, trainset: ActivationsDataset, valset: ActivationsDataset):
        super().__init__()
        self.model = model
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.trainset = trainset
        self.valset = valset

    def forward(self, x: t.Tensor) -> t.Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx: int) -> t.Tensor:
        logits, labels = self._shared_train_val_step(batch)
        loss = t.nn.functional.cross_entropy(logits, labels)
        self.log("train_loss", loss)
        return loss

    def _shared_train_val_step(self, batch):
        in_, labels = batch
        in_ = in_.to(device)
        labels = labels.to(device)
        logits = self(in_)
        return logits, labels
    
    def validation_step(self, batch, batch_idx: int) -> None:
        logits, labels = self._shared_train_val_step(batch)
        classifications = logits.argmax(dim=1)
        accuracy = t.sum(classifications == labels) / len(classifications)
        self.log("accuracy", accuracy)

    def configure_optimizers(self):
        return t.optim.Adam(self.parameters())
    
    def train_dataloader(self):
        return t.utils.data.DataLoader(self.trainset, batch_size=self.batch_size, shuffle=False)

    def val_dataloader(self):
        return t.utils.data.DataLoader(self.valset, batch_size=self.batch_size, shuffle=False)

# %%
if MAIN:# %%
    LAYER_NAME = "embedder.block2.res2.resadd_out"
    TRAIN_FILES = [1,2,3,4,5]
    VAL_FILES = [6]

    batch_size = 32
    max_epochs = 10
    train_dataset = ActivationsDataset(layer_name=LAYER_NAME, file_ids=TRAIN_FILES)
    test_dataset = ActivationsDataset(layer_name=LAYER_NAME, file_ids=VAL_FILES)
    
    probe = NextActionProbe(LAYER_NAME)
    model = LitModel(probe, batch_size, max_epochs, train_dataset, test_dataset).to(device)
    assert str(model.device).startswith(str(device)), f"model has device {model.device}"
    
    # Get a logger, to record metrics during training
    logger = CSVLogger(save_dir=os.getcwd() + "/logs", name="t1")

    # Train the model (hint: here are some helpful Trainer arguments for rapid idea iteration)
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        logger=logger,
        log_every_n_steps=1,
    )
    trainer.fit(model=model)
    # print(sorted(model.model.freqs.tolist()))

# %%
if MAIN:
    metrics = pd.read_csv(f"{trainer.logger.log_dir}/metrics.csv") 
    import sys
    sys.path.append('/home/janbet/ARENA_2.0/chapter0_fundamentals/exercises')

    from plotly_utils import plot_train_loss_and_test_accuracy_from_metrics   
    plot_train_loss_and_test_accuracy_from_metrics(metrics, "TTTT")
# %%
