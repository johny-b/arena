# %%
import os
os.chdir('/home/janbet/arena/t1')
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import torch as t
from torch.utils.data import Dataset
from torch import nn
import pandas as pd

from procgen_tools.imports import load_model
from procgen_tools import maze, visualization
device = t.device("cpu")

import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger

import sys
sys.path.append('/home/janbet/ARENA_2.0/chapter0_fundamentals/exercises')

from plotly_utils import plot_train_loss_and_test_accuracy_from_metrics   

MAIN = __name__ == '__main__'

# %%
policy, hook = load_model()

# %%
class LogitDataset(Dataset):
    def __init__(self, num_mazes: int, maze_size: int, ratio: float = 1):
        self.num_mazes = num_mazes
        self.maze_size = maze_size
        self.ratio = ratio
        self._data = self._create_data()
        
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
            print(f"PROCESS SEEDS: {ix}/{self.num_mazes}")
        return data
    
    def _get_data_from_seed(self, seed):
        venv = maze.create_venv(num=1, start_level=seed, num_levels=1)
        venv_all, (legal_mouse_positions, grid) = maze.venv_with_all_mouse_positions(venv)
        with t.no_grad():
            categorical, _ = policy(t.tensor(venv_all.reset(), dtype=t.float32))
        
        data = []
        for mouse_pos, logits in zip(legal_mouse_positions, categorical.logits):
            label = self._get_label(grid, mouse_pos)
            if np.random.random() < self.ratio:
                data.append([logits, label])
        return data

    @staticmethod
    def _get_label(grid, mouse_pos):
        x, y = mouse_pos
        neighbours = [
            (x + 1, y),
            (x, y + 1),
            (x, y - 1),
            (x - 1, y)
        ]
        labels = []
        for neighbour in neighbours:
            try:
                labels.append(grid[neighbour] != 51)
            except IndexError:
                labels.append(False)
                
        assert any(labels), "This should not be possible"
        return [mouse_pos, labels]
# %%
train_dataset = LogitDataset(200, 9, ratio=0.1)
val_dataset = LogitDataset(20, 9, ratio=0.1)

# %%
class LegalActionProbe(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(15, 4).to(t.device("cpu"))
        
    def forward(self, x):
        val = self.linear(x)
        val_min = val.min(-1).values
        val = val - val_min.unsqueeze(1)
        val_max = val.max(-1).values
        val = val / val_max.unsqueeze(1)
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
        loss = (preds - labels).square().sum()
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
result = []
if MAIN:# %%
    batch_size = 128
    max_epochs = 100
    print(len(train_dataset))
    
    probe = LegalActionProbe().to(device)

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
# plot_train_loss_and_test_accuracy_from_metrics(metrics, "ttt")
# %%

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
