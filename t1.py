# %%

import pandas as pd
import torch as t

import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger

import sys
sys.path.append('/home/janbet/ARENA_2.0/chapter0_fundamentals/exercises')

from plotly_utils import line, plot_train_loss_and_test_accuracy_from_metrics

MAIN = __name__ == '__main__' 

import models as m

from IPython import get_ipython
ipython = get_ipython()
ipython.run_line_magic("load_ext", "autoreload")
ipython.run_line_magic("autoreload", "2")


device = t.device('cuda' if t.cuda.is_available() else 'cpu')

# %%

class ModDataset(t.utils.data.Dataset):
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

def get_datasets(p):
    trainset = ModDataset(p, 7)
    testset = ModDataset(p, 3)

    return trainset, testset

# %%
class LitModel(pl.LightningModule):
    def __init__(self, model, batch_size: int, max_epochs: int):
        super().__init__()
        self.model = model
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.trainset, self.testset = get_datasets(model.p)

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
        '''
        Choose what optimizers and learning-rate schedulers to use in your optimization.
        '''
        return t.optim.Adam(self.parameters())
    
    def train_dataloader(self):
        '''
        Return the training dataloader.
        '''
        return t.utils.data.DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return t.utils.data.DataLoader(self.testset, batch_size=self.batch_size, shuffle=True)

# %%
if MAIN:
    p = 113
    batch_size = 64
    max_epochs = 20
    n_freqs=64

    x = m.FreqsSinParam(p, n_freqs=n_freqs)

    model = LitModel(x, batch_size, max_epochs).to(device)
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
    plot_train_loss_and_test_accuracy_from_metrics(metrics, "Training ConvNet on MNIST data")
    
# %%
