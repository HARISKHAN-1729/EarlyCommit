"""Latency benchmark: warm-up first, then per-word timings on realistic prefixes.

  USE_TF=0 python -m earlycommit.bench --device cuda
  USE_TF=0 python -m earlycommit.bench --device cpu --n 30
"""
import argparse
import time

import numpy as np

from .core import make_episodes
from .judge import LayaJudge


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="convaiinnovations/laya")
    ap.add_argument("--subfolder", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=5)
    a = ap.parse_args()

    t0 = time.perf_counter()
    j = LayaJudge(a.repo, a.subfolder, a.device)
    print(f"load: {time.perf_counter() - t0:.1f} s")
    try:
        import torch
        p = next(j.agent.model.parameters()) if hasattr(j.agent, "model") else None
        print(f"torch cuda available: {torch.cuda.is_available()} | model on: {p.device if p is not None else 'unknown'}")
    except Exception as e:  # attribute names inside laya may differ
        print(f"(could not inspect device: {e})")

    eps = make_episodes(a.n, seed=7)
    calls = [(ep.scene, ep.utt.words[: max(2, len(ep.utt.words) // 2)]) for ep in eps]
    for sc, w in calls[: a.warmup]:
        j.predict(sc, w)
    lat = []
    for sc, w in calls:
        j.predict(sc, w)
        lat.append(j.last_latency_ms)
    lat = np.array(lat)
    print(f"device={a.device}  n={len(lat)}  p50={np.percentile(lat, 50):.1f} ms  "
          f"p95={np.percentile(lat, 95):.1f} ms  max={lat.max():.1f} ms")
    print(f"real-time budget per word is ~300 ms -> p95 {'OK' if np.percentile(lat, 95) < 300 else 'TOO SLOW'}")


if __name__ == "__main__":
    main()
