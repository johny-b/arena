# %%

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch as t
from torch import nn
import random

MAIN = '__main__'
N_FREQ = 64

device = 'cpu'

# %%
# Implementation of (a + b) % p according to section 3.1 of https://arxiv.org/pdf/2301.05217.pdf
def freq_2_logit(
    sin_a: t.Tensor, 
    cos_a: t.Tensor, 
    sin_b: t.Tensor, 
    cos_b: t.Tensor,
    freqs: t.Tensor, 
    p: int,
):
    sin_sum = sin_a * cos_b + cos_a * sin_b
    cos_sum = cos_a * cos_b - sin_a * sin_b

    c = t.arange(p, dtype=float)
    sin_c = t.sin(c)
    cos_c = t.cos(c)

    x = c.unsqueeze(0)
    freqs = freqs.unsqueeze(0)
    cos_c = t.cos(x.T @ freqs)
    sin_c = t.sin(x.T @ freqs)

    cos_a_plus_b_minus_c = cos_sum * cos_c + sin_sum * sin_c

    return cos_a_plus_b_minus_c.sum(dim=-1)
                                             
def sum_mod_p(a: int, b: int, p: int, size: int = 64) -> int:
    """Returns (a + b) % p using freq_2_logit"""
    freqs = (2  * t.pi / p) * t.randint(1, 10 ** 5, (size,))
    freqs = freqs.to(float)
    
    sin_a = t.sin(a * freqs)
    cos_a = t.cos(a * freqs)

    sin_b = t.sin(b * freqs)
    cos_b = t.cos(b * freqs)

    logits = freq_2_logit(sin_a, cos_a, sin_b, cos_b, freqs, p)
    return logits.argmax()
 
# if MAIN:
#    for a, b, p in t.randint(1, 1000, (100, 3)):
#         a, b, p = int(a), int(b), int(p)
#         assert sum_mod_p(a, b, p, size=15).item() == (a + b) % p, f"NOPE: {a} {b} {p}"

# %%
class SingleLayer(nn.Module):
    def __init__(self, p, n_freqs=50):
        super().__init__()
        self.p = p
        self.n_freqs = n_freqs

        freqs = (2  * t.pi / p) * t.randint(1, 10 ** 5, (n_freqs,))
        self.freqs = freqs.to(float)

        self.sin = nn.Linear(n_freqs, n_freqs, bias=False).double()
        self.cos = nn.Linear(n_freqs, n_freqs, bias=False).double()

    def forward(self, x):
        print(x.device)
        a = x[:, 0]
        b = x[:, 1]

        print(x.shape)
        print(a.shape)
        print(b.shape)
        sin_a = self.sin(a * self.freqs)
        cos_a = self.cos(a * self.freqs)

        sin_b = self.sin(b * self.freqs)
        cos_b = self.cos(b * self.freqs)
        
        print(sin_a.shape)

        return freq_2_logit(sin_a, cos_a, sin_b, cos_b, self.freqs, self.p)

# %%
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger

# %%

class ModDataset(t.utils.data.Dataset):
    """Face Landmarks dataset."""

    def __init__(self, p: int, nth: int):
        self.p = p
        self.nth = nth

        self.data = self.get_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
  
    def get_data(self):
        all_data = []
        for i in range(self.p):
            for j in range(self.p):
                input = t.tensor([i, j], dtype=float)
                label = (i + j) % self.p
                all_data.append((input, label))
    
        return [x for i, x in enumerate(all_data) if not i % self.nth]

# %%

def get_datasets(p):
    trainset = ModDataset(p, 7)
    testset = ModDataset(p, 3)

    return trainset, testset

# %%
class LitSingleLayer(pl.LightningModule):
    def __init__(self, p: int, batch_size: int, max_epochs: int):
        super().__init__()
        self.p = p
        self.model = SingleLayer(p)
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.trainset, self.testset = get_datasets(p)

    def forward(self, x: t.Tensor) -> t.Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx: int) -> t.Tensor:
        in_, labels = batch
        in_ = in_.to(device)
        labels = labels.to(device)
        logits = self(in_)
        print(len(logits))
        print(len(labels))
        loss = t.nn.functional.cross_entropy(logits, labels)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        '''
        Choose what optimizers and learning-rate schedulers to use in your optimization.
        '''
        return t.optim.Adam(self.parameters())
    
    def train_dataloader(self):
        '''
        Return the training dataloader.
        '''
        return t.utils.data.DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True)

# %%

# Create the model & training system

if MAIN:
    p = 113
    batch_size = 64
    max_epochs = 3
    model = LitSingleLayer(p, batch_size, max_epochs).to(device)
    assert str(model.device) == device, f"model has device {model.device}"
    
    # Get a logger, to record metrics during training
    logger = CSVLogger(save_dir=os.getcwd() + "/logs", name="t1")
    
    # Train the model (hint: here are some helpful Trainer arguments for rapid idea iteration)
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        logger=logger,
        log_every_n_steps=1,
    )
    trainer.fit(model=model)

# %%
import pandas as pd

if MAIN:
    metrics = pd.read_csv(f"{trainer.logger.log_dir}/metrics.csv")
    
    metrics.head()

# %%


if MAIN:
    print(metrics["train_loss"].values)
    from plotly_utils import line
    line(
        metrics["train_loss"].values,
        x=metrics["step"].values,
        yaxis_range=[0, metrics["train_loss"].max() + 0.1],
        labels={"x": "Batches seen", "y": "Cross entropy loss"},
        title="ConvNet training on MNIST",
        width=800,
        hovermode="x unified",
        template="ggplot2", # alternative aesthetic for your plots (-:
    )

# %%
