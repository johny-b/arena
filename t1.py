# %%

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch as t
from torch import nn
import random

MAIN = '__main__'
N_FREQ = 64

device = 'cpu'

from freq_2_logit import batched_freq_2_logit

# %%
class SingleLayer(nn.Module):
    def __init__(self, p, n_freqs=50):
        super().__init__()
        self.p = p
        self.n_freqs = n_freqs

        self.freqs = nn.Linear(1, n_freqs).double()
        # self.freqs = (2  * t.pi / p) * t.randint(1, 10 ** 5, (64,)).to(float)

        # freqs = (2  * t.pi / p) * t.randint(1, 10 ** 5, (n_freqs,))
        # self.freqs = freqs.to(float)

        # self.sin = nn.Linear(n_freqs, n_freqs, bias=False).double()
        # self.cos = nn.Linear(n_freqs, n_freqs, bias=False).double()

    def forward(self, x):
        a = x[:, 0]
        b = x[:, 1]

        sin_a = t.sin(self.freqs(a.unsqueeze(0).T))
        cos_a = t.cos(a.unsqueeze(0).T * self.freqs)
        sin_b = t.sin(b.unsqueeze(0).T * self.freqs)
        cos_b = t.cos(b.unsqueeze(0).T * self.freqs)


        logits = batched_freq_2_logit(sin_a, cos_a, sin_b, cos_b, self.freqs, p)
        return logits

if MAIN:
    p = 113
    a = t.randint(1, p, (100,))
    b = t.randint(1, p, (100,))

    mod = (a + b) % p

    model = SingleLayer(p)
    
    ab = t.stack((a, b), dim=1)
    logits = model.forward(ab)
    out = logits.argmax(-1)
    
    print(out, model)

    t.testing.assert_close(out, mod)

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
