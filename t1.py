# %%

import torch as t
from torch import nn
import random

MAIN = '__main__'
N_FREQ = 64

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
 
if MAIN:
    for a, b, p in t.randint(1, 1000, (100, 3)):
        a, b, p = int(a), int(b), int(p)
        assert sum_mod_p(a, b, p, size=64).item() == (a + b) % p, f"NOPE: {a} {b} {p}"

# %%
