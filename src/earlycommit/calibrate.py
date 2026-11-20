"""Temperature scaling that works from probabilities alone.

log p_i = logit_i - logsumexp(logit), so dividing log-probs by T and
renormalising is exactly logit temperature scaling. No access to raw logits
needed, which matters because Laya's API returns probabilities.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar


def apply_temperature(p: dict[str, float], T: float) -> dict[str, float]:
    keys = list(p)
    lp = np.log(np.clip([p[k] for k in keys], 1e-12, 1.0)) / T
    lp -= lp.max()
    e = np.exp(lp)
    e /= e.sum()
    return dict(zip(keys, e.tolist()))


def nll(dists: list[dict[str, float]], soft_labels: list[dict[str, float]], T: float) -> float:
    total = 0.0
    for p, y in zip(dists, soft_labels):
        q = apply_temperature(p, T)
        total -= sum(w * np.log(max(q[k], 1e-12)) for k, w in y.items())
    return total / len(dists)


def fit_temperature(dists, soft_labels) -> float:
    res = minimize_scalar(lambda lt: nll(dists, soft_labels, float(np.exp(lt))),
                          bounds=(-3, 3), method="bounded")
    return float(np.exp(res.x))


def reliability(dists, soft_labels, n_bins: int = 10):
    """Confidence of the top choice vs the probability that it is correct
    (expected under the soft label). Returns (ece, bins)."""
    conf, acc = [], []
    for p, y in zip(dists, soft_labels):
        k = max(p, key=p.get)
        conf.append(p[k])
        acc.append(y.get(k, 0.0))
    conf, acc = np.array(conf), np.array(acc)
    edges = np.linspace(0, 1, n_bins + 1)
    bins, ece = [], 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            bins.append((float(conf[m].mean()), float(acc[m].mean()), int(m.sum())))
            ece += m.mean() * abs(conf[m].mean() - acc[m].mean())
    return float(ece), bins
