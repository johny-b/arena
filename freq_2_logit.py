# %%
import torch as t

MAIN = __name__ == '__main__'
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

# %%
# Test freq_2_logit
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

def batched_freq_2_logit(
    sin_a: t.Tensor,
    cos_a: t.Tensor,
    sin_b: t.Tensor,
    cos_b: t.Tensor,
    freqs: t.Tensor,
    p: int,
):
    sin_sum = sin_a * cos_b + cos_a * sin_b  # batch, freqs
    cos_sum = cos_a * cos_b - sin_a * sin_b  # batch, freqs

    c = t.arange(p, dtype=float)  # p
    sin_c = t.sin(c)  # p
    cos_c = t.cos(c)  # p

    x = c.unsqueeze(0)  # 1 p
    freqs = freqs.unsqueeze(0)  # 1 freqs
    cos_c = t.cos(x.T @ freqs)  # p freqs
    sin_c = t.sin(x.T @ freqs)  # p freqs

    cos_sum = cos_sum.unsqueeze(1)  # batch 1 freqs
    sin_sum = sin_sum.unsqueeze(1)  # batch 1 freqs

    cos_c = cos_c.unsqueeze(0)  # 1 p freqs
    sin_c = sin_c.unsqueeze(0)  # 1 p freqs

    cos_a_plus_b_minus_c = cos_sum * cos_c + sin_sum * sin_c  # batch, p, freqs

    return cos_a_plus_b_minus_c.sum(dim=-1)  # batch, p


def batched_sum_mod_p(a: t.Tensor, b: t.Tensor, p: int, size: int = 64):
    """Returns (a + b) % p using freq_2_logit"""
    freqs = (2  * t.pi / p) * t.randint(1, 10 ** 5, (size,))
    freqs = freqs.to(float)
    
    sin_a = t.sin(a.unsqueeze(0).T * freqs)
    cos_a = t.cos(a.unsqueeze(0).T * freqs)
    sin_b = t.sin(b.unsqueeze(0).T * freqs)
    cos_b = t.cos(b.unsqueeze(0).T * freqs)

    logits = batched_freq_2_logit(sin_a, cos_a, sin_b, cos_b, freqs, p)
    print(logits.shape)
    return logits.argmax(-1)

if MAIN:
    p = t.randint(1, 1000, (1,)).item()
    a = t.randint(1, p, (100,))
    b = t.randint(1, p, (100,))
    mod = (a + b) % p
    out = batched_sum_mod_p(a, b, p, 64)
    
    t.testing.assert_close(out, mod)


# %%
