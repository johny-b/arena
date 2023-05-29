# %%
import torch as t
from torch import nn
import random
from jaxtyping import Float, Int, jaxtyped
import typeguard

MAIN = __name__ == '__main__'
N_FREQ = 64

from freq_2_logit import batched_freq_2_logit

# %%
class FreqsParam(nn.Module):
    def __init__(self, p, n_freqs=50):
        super().__init__()
        self.p = p
        self.n_freqs = n_freqs

        self.freqs = nn.Parameter(t.empty((n_freqs,), dtype=t.double))
        t.nn.init.normal_(self.freqs)

    def forward(self, x):
        a = x[:, 0].double()
        b = x[:, 1].double()

        sin_a = t.sin(a.unsqueeze(0).T * self.freqs)
        cos_a = t.cos(a.unsqueeze(0).T * self.freqs)
        sin_b = t.sin(b.unsqueeze(0).T * self.freqs)
        cos_b = t.cos(b.unsqueeze(0).T * self.freqs)

        logits = batched_freq_2_logit(sin_a, cos_a, sin_b, cos_b, self.freqs, self.p)
        return logits

def test_model(p, model):
    n_nums = 100
    a = t.randint(1, p, (n_nums,))
    b = t.randint(1, p, (n_nums,))

    mod = (a + b) % p
    ab = t.stack((a, b), dim=1)
    logits = model.forward(ab)
    out = logits.argmax(-1)

    t.testing.assert_close(out, mod)

if MAIN:
    p = 113
    n_freqs = 64

    #   Test FreqParams by setting freqs to a "correct" value
    model = FreqsParam(p, n_freqs=n_freqs)
    model.freqs = nn.Parameter((2  * t.pi / p) * t.randint(1, 10 ** 5, (n_freqs,)).double())
    test_model(p, model)

    
# %%
class FreqsSinParam(nn.Module):
    def __init__(self, p, n_freqs=50):
        super().__init__()
        self.p = p
        self.n_freqs = n_freqs

        self.freqs = nn.Parameter(t.empty((n_freqs,), dtype=t.double))
        self.sin = nn.Parameter(t.empty((n_freqs,), dtype=t.double))
        t.nn.init.normal_(self.freqs)
        t.nn.init.normal_(self.sin)

    def forward(self, x):
        a = x[:, 0].double()
        b = x[:, 1].double()

        sin_a = self.sin * (a.unsqueeze(0).T * self.freqs)
        cos_a = t.cos(a.unsqueeze(0).T * self.freqs)
        sin_b = self.sin * (b.unsqueeze(0).T * self.freqs)
        cos_b = t.cos(b.unsqueeze(0).T * self.freqs)

        logits = batched_freq_2_logit(sin_a, cos_a, sin_b, cos_b, self.freqs, self.p)
        return logits

def test_model(p, model):
    n_nums = 100
    a = t.randint(1, p, (n_nums,))
    b = t.randint(1, p, (n_nums,))

    mod = (a + b) % p
    ab = t.stack((a, b), dim=1)
    logits = model.forward(ab)
    out = logits.argmax(-1)

    t.testing.assert_close(out, mod)

if MAIN:
    p = 113
    n_freqs = 64

    #   Test FreqParams by setting freqs to a "correct" value
    model = FreqsParam(p, n_freqs=n_freqs)
    model.freqs = nn.Parameter((2  * t.pi / p) * t.randint(1, 10 ** 5, (n_freqs,)).double())
    test_model(p, model)