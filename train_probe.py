# %%

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import glob
import pickle
from typing import Callable, List

import pandas as pd

from functools import lru_cache

import torch as t
from torch.utils.data import Dataset
from torch import nn

import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger

import sys
sys.path.append('/home/janbet/ARENA_2.0/chapter0_fundamentals/exercises')

from plotly_utils import plot_train_loss_and_test_accuracy_from_metrics   


device = t.device("cpu")
MAIN = __name__ == '__main__'
# %%
class ActivationsDataset(Dataset):
    def __init__(self, layer_name: str, selected_seed: Callable[[int], bool]):
        self.layer_name = layer_name
        self.files = self._get_files(selected_seed)
        
        self._action_map = ["LEFT", "UP", "RIGHT", "DOWN", "NOOP"]
        
    def _get_files(self, selected_seed: Callable[[int], bool]) -> List[int]:
        dir_ = '/home/janbet/arena/activations_1/'
        all_files = glob.glob(dir_ + f'*{self.layer_name}.pickle')
        selected_files = []
        for fname in all_files:
            seed = int(fname.split('_')[2])
            if selected_seed(seed):
                selected_files.append(fname)
        return sorted(selected_files)
        
    def __len__(self):
        return len(self.files)
    
    @lru_cache
    def __getitem__(self, idx):
        with open(self.files[idx], 'rb') as f:
            data = pickle.load(f)
        
        seed, layer_name, action, action_to_cheese, cheese, act = data    
        action_ix = self._action_map.index(action)
        
        return act, action_ix
    

# %%

class NextActionProbe(nn.Module):
    def __init__(self, layer_name):
        super().__init__()
        self.layer_name = layer_name
        self._in_size = self._calc_in_size()
        self._out_size = 5
        self.linear = nn.Linear(self._in_size, self._out_size).to(t.device("cpu"))
        
    def forward(self, x):
        return self.linear(x.flatten(start_dim=1))
        
    def _calc_in_size(self):
        dataset = ActivationsDataset(self.layer_name, lambda seed: True)
        sample_input = dataset[0][0]
        return sample_input.flatten().shape[0]


# %%
class LitModel(pl.LightningModule):
    def __init__(self, model, batch_size: int, max_epochs: int, trainset: ActivationsDataset, valset: ActivationsDataset):
        super().__init__()
        self.model = model.to(device)
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
        return t.utils.data.DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True, num_workers=4, persistent_workers=True)

    def val_dataloader(self):
        return t.utils.data.DataLoader(self.valset, batch_size=self.batch_size, shuffle=False, num_workers=2, persistent_workers=True)

# %%
if MAIN:# %%

    batch_size = 32
    max_epochs = 30

    for LAYER_NAME in (
        # "embedder.block1.maxpool_out",
        # "embedder.block1.res1.resadd_out",
        # "embedder.block1.res2.resadd_out",
        # "embedder.block2.maxpool_out",
        "embedder.block2.res1.resadd_out",
        "embedder.block2.res2.resadd_out",
        "embedder.block3.maxpool_out",
        # "embedder.block3.res1.resadd_out",
        # "embedder.block3.res2.resadd_out",
        # "embedder.relu3_out",
        # "embedder.relufc_out",
    ):
        train_dataset = ActivationsDataset(layer_name=LAYER_NAME, selected_seed=lambda seed: seed % 100 < 80)
        test_dataset = ActivationsDataset(layer_name=LAYER_NAME, selected_seed=lambda seed: seed % 100 >= 80)
        print(len(train_dataset))
        
        probe = NextActionProbe(LAYER_NAME).to(device)

        model = LitModel(probe, batch_size, max_epochs, train_dataset, test_dataset)
        model = model.to(device)
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

        metrics = pd.read_csv(f"{trainer.logger.log_dir}/metrics.csv") 
        plot_train_loss_and_test_accuracy_from_metrics(metrics, LAYER_NAME)
# %%
