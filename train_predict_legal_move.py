# %%
import os

import numpy as np
import torch as t
from torch.utils.data import Dataset
from torch import nn
import pandas as pd
import pickle

from procgen_tools.imports import load_model
from procgen_tools import maze, visualization
device = t.device("cuda")

import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger

import sys
sys.path.append('/root/ARENA_2.0/chapter0_fundamentals/exercises')

from plotly_utils import plot_train_loss_and_test_accuracy_from_metrics   

MAIN = __name__ == '__main__'

# %%
if MAIN:
    policy, hook = load_model()

# %%
class LogitDataset(Dataset):
    def __init__(self, num_mazes: int, maze_size: int, ratio: float = 1):
        self.num_mazes = num_mazes
        self.maze_size = maze_size
        self.ratio = ratio
        self.layer_name = "embedder.block1.res2.resadd_out"
        self._data = self._create_data()
        
    def save(self, prefix):
        fname = prefix + '.pickle'
        with open(fname, 'wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    @classmethod
    def load(cls, fname):
        with open(fname, 'rb') as f:
            return pickle.load(f)
        
    def __len__(self):
        return len(self._data)
    
    def __getitem__(self, index):
        logits, (pos, labels) = self._data[index]
        return logits, t.tensor(labels, dtype=t.float32)
        
    def _create_data(self):
        seeds = set()
        while len(seeds) < self.num_mazes:
            while True:
                seed = np.random.randint(0, 100000000)
                if maze.get_inner_grid_from_seed(seed=seed).shape[0] == self.maze_size:
                    break
            seeds.add(seed)
            print(f"CREATE SEEDS: {len(seeds)}/{self.num_mazes}")
            
        data = []
        for ix, seed in enumerate(seeds):
            data += self._get_data_from_seed(seed)
            print(f"PROCESS SEEDS: {ix + 1}/{self.num_mazes}")
        return data
    
    @property
    def input_shape(self) -> int:
        return self[0][0].flatten().shape[0]
    
    def _get_data_from_seed(self, seed):
        venv = maze.create_venv(num=1, start_level=seed, num_levels=1)
        venv_all, (legal_mouse_positions, grid) = maze.venv_with_all_mouse_positions(venv)
        
        with t.no_grad():
            hook.run_with_input(venv_all.reset().astype('float32'))
        
        activations = hook.values_by_label[self.layer_name]
    
        # activations = hook.values_by_label['_out'][0].logits
        
        data = []
        for mouse_pos, activation in zip(legal_mouse_positions, activations):
            label = self._get_label(grid, mouse_pos)
            if np.random.random() < self.ratio:
                data.append([activation, label])
        return data

    @staticmethod
    def _get_label(grid, mouse_pos):
        x, y = mouse_pos
        neighbours = [
            (x + 1, y),
            (x, y + 1),
            (x - 1, y),
            (x, y - 1)
        ]
        labels = []
        for neighbour in neighbours:
            if any(x < 0 for x in neighbour):
                val = False
            elif any(x > grid.shape[0] - 1 for x in neighbour):
                val = False
            else:
                val = grid[neighbour] != 51
            labels.append(val)
            
        # x = maze.venv_from_grid(grid)
        # state = maze.state_from_venv(x)
        # state.set_mouse_pos(mouse_pos[0] + 7 + 1, mouse_pos[1] + 7 + 1)
        
        # print(labels)
        # visualization.visualize_venv(maze.venv_from_grid(state.inner_grid()))
                
        assert any(labels), "This should not be possible"
        return [mouse_pos, labels]
# %%
if MAIN:
    train_dataset = LogitDataset(4000, 9, ratio=0.1)
    val_dataset = LogitDataset(100, 9, ratio=1)

    # train_dataset = LogitDataset.load('train_t1.pickle')
    # val_dataset = LogitDataset.load('val_t1.pickle')
    train_dataset.save('train_large')
    # val_dataset.save('val_large')

# %%
class LegalActionProbe(nn.Module):
    def __init__(self, in_size: int):
        super().__init__()
        self.linear = nn.Linear(in_size, 4).to(t.device("cpu"))
        
    def forward(self, x: t.Tensor):
        x = x.flatten(start_dim=1)
        val = self.linear(x)
        val = nn.functional.sigmoid(val)
        return val

# %%
class LitModel(pl.LightningModule):
    def __init__(self, model, batch_size: int, max_epochs: int, trainset: LogitDataset, valset: LogitDataset):
        super().__init__()
        self.model = model.to(device)
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.trainset = trainset
        self.valset = valset

    def forward(self, x: t.Tensor) -> t.Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx: int) -> t.Tensor:
        preds, labels = self._shared_train_val_step(batch)
        loss = t.nn.functional.cross_entropy(preds, labels)
        self.log("train_loss", loss)
            
        return loss

    def _shared_train_val_step(self, batch):
        in_, labels = batch
        in_ = in_.to(device)
        labels = labels.to(device)
        logits = self(in_)
        return logits, labels
    
    def validation_step(self, batch, batch_idx: int) -> None:
        preds, labels = self._shared_train_val_step(batch)

        classifications = (preds > 0.5)
        
        accuracy = (classifications == labels).to(t.float32).mean() 
        self.log("accuracy", accuracy)

    def configure_optimizers(self):
        return t.optim.Adam(self.parameters())
    
    def train_dataloader(self):
        return t.utils.data.DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return t.utils.data.DataLoader(self.valset, batch_size=self.batch_size, shuffle=False)

# %%

if MAIN:# %%
    batch_size = 128
    max_epochs = 500
    print(len(train_dataset))
    
    probe = LegalActionProbe(train_dataset.input_shape).to(device)

    model = LitModel(probe, batch_size, max_epochs, train_dataset, val_dataset)
    model = model.to(device)
    assert str(model.device).startswith(str(device)), f"model has device {model.device}"
    
    # Get a logger, to record metrics during training
    logger = CSVLogger(save_dir="/home/janbet/logs", name="legal_action_probe")

    # Train the model (hint: here are some helpful Trainer arguments for rapid idea iteration)
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        logger=logger,
        log_every_n_steps=1,
    )
    trainer.fit(model=model)

    metrics = pd.read_csv(f"{trainer.logger.log_dir}/metrics.csv") 
    plot_train_loss_and_test_accuracy_from_metrics(metrics, "ttt")


# %%
if MAIN:
    plot_train_loss_and_test_accuracy_from_metrics(metrics, "ttt")
# %%

if MAIN:
    size = len(train_dataset._data)
    maze_size = train_dataset.maze_size
    layer = train_dataset.layer_name
    name = f'model_{layer}_{maze_size}_{size}_{max_epochs}.pth'
    t.save(model.state_dict(), name)
    print("SAVED", name)

# print(train_dataset._data[0])
# %%
# UP, RIGHT, LEFT, DOWN
# venv = maze.create_venv(num=1, start_level=84451, num_levels=1)
# state = maze.state_from_venv(venv)
# state.set_mouse_pos(10, 10)
# venv = maze.venv_from_grid(state.inner_grid())
# visualization.visualize_venv(venv, render_padding=False)

# with t.no_grad():
#     categorical, _ = policy(t.tensor(venv.reset(), dtype=t.float32))
    
# logits = categorical.logits
# x = t.tensor([[-1.5644, -1.0427, -0.9620, -5.5040, -5.4606, -4.1507, -6.4315, -7.1552,
#          -7.1789, -6.3089, -4.7793, -4.8320, -5.7904, -6.2919, -5.1482]])
# print(model(x))
# # %%
