"""Judges map (scene, partial utterance) -> probability over scene objects.

LayaJudge  : the real thing (convaiinnovations/laya or your fine-tuned copy).
MockJudge  : oracle listener + noise + over-confidence. Lets the whole pipeline
             (calibration, commit policy, evaluation, plots) run before Laya is
             fine-tuned, and doubles as an upper-bound baseline in the paper.
"""
from __future__ import annotations

import math
import random
import time

from .language import consistent
from .scenes import Scene

QUESTION_TEXT = ("Which object on the table does the speaker mean, "
                 "given only the words said so far?")


def build_state(scene: Scene, prefix_words: list[str]) -> dict:
    return {
        "scene": [{"id": o.id, "color": o.color, "shape": o.shape,
                   "position": f"{o.column} {o.row}"} for o in scene.objects],
        "utterance_so_far": " ".join(prefix_words),
    }


def build_questions(scene: Scene) -> dict:
    # Neutral keys (obj_i) with descriptive criteria; avoids the noul label bug.
    return {"target": {"type": "choice", "instructions": QUESTION_TEXT,
                       "criteria": {o.id: o.describe() for o in scene.objects}}}


class Judge:
    def predict(self, scene: Scene, prefix_words: list[str]) -> dict[str, float]:
        raise NotImplementedError


class MockJudge(Judge):
    def __init__(self, noise: float = 0.15, overconfidence: float = 2.5, seed: int = 0):
        self.noise, self.sharpen, self.rng = noise, overconfidence, random.Random(seed)

    def predict(self, scene, prefix_words):
        cands = set(consistent(scene.objects, prefix_words))
        logits = {}
        for o in scene.objects:
            base = 1.0 if o.id in cands else 0.05
            logits[o.id] = self.sharpen * math.log(base) + self.rng.gauss(0, self.noise * self.sharpen)
        z = max(logits.values())
        exp = {k: math.exp(v - z) for k, v in logits.items()}
        s = sum(exp.values())
        return {k: v / s for k, v in exp.items()}


class CachedJudge(Judge):
    """Memoises predictions per (scene, prefix). The evaluation asks for the same
    prefix many times (calibration pass + once per tau); this computes each once."""

    def __init__(self, inner: Judge):
        self.inner, self.cache, self.hits, self.misses = inner, {}, 0, 0

    def predict(self, scene, prefix_words):
        key = (scene.id, tuple(o.id for o in scene.objects), tuple(prefix_words))
        if key in self.cache:
            self.hits += 1
        else:
            self.misses += 1
            self.cache[key] = self.inner.predict(scene, prefix_words)
        return self.cache[key]


class KeywordJudge(Judge):
    """Non-learned baseline: exact keyword matching against the scene (this is
    the oracle's own parser). Perfect on the training templates, brittle outside them."""

    def predict(self, scene, prefix_words):
        c = consistent(scene.objects, prefix_words) or [o.id for o in scene.objects]
        return {o.id: (1.0 / len(c) if o.id in c else 0.0) for o in scene.objects}


class LayaJudge(Judge):
    """Thin wrapper over the documented `agent.predict(state, questions)` API.

    Laya returns the full per-option distribution in answers[q]["probabilities"]
    (verified with laya 0.3.7). The fallback below is only for other versions.
    """
    DIST_KEYS = ("probabilities", "distribution", "probs", "scores", "options")

    def __init__(self, repo: str = "convaiinnovations/laya", subfolder: str | None = None,
                 device: str | None = None):
        import os
        import laya  # pip install -U laya ; set USE_TF=0 if loading hangs
        if os.path.isdir(repo):
            # Local fine-tuned checkpoint (folder with rl_agent_config.json, model.safetensors,
            # encoder/, tokenizer/), as written by colab_finetune_earlycommit.py.
            self.agent = laya.Agent(repo, device=device or "cuda")
        else:
            kw = {"subfolder": subfolder} if subfolder else {}
            if device:
                kw["device"] = device
            self.agent = laya.load(repo, **kw)
        self.last_latency_ms = None

    def raw(self, scene, prefix_words) -> dict:
        t0 = time.perf_counter()
        out = self.agent.predict(build_state(scene, prefix_words), build_questions(scene))
        self.last_latency_ms = (time.perf_counter() - t0) * 1000
        return out

    def predict(self, scene, prefix_words):
        ans = self.raw(scene, prefix_words)["answers"]["target"]
        ids = [o.id for o in scene.objects]
        for key in self.DIST_KEYS:
            d = ans.get(key)
            if isinstance(d, dict) and set(ids) <= set(d):
                s = sum(float(d[i]) for i in ids)
                return {i: float(d[i]) / s for i in ids}
        # Fallback: only argmax + confidence available -> spread the remainder.
        # This loses information; fix DIST_KEYS instead of relying on it.
        choice, conf = ans["choice"], float(ans.get("confidence", 1.0))
        rest = (1 - conf) / max(1, len(ids) - 1)
        return {i: (conf if i == choice else rest) for i in ids}


if __name__ == "__main__":
    import json
    from .scenes import random_scene
    sc = random_scene(random.Random(0), "probe")
    j = LayaJudge()
    print(json.dumps(j.raw(sc, "pick up the red".split()), indent=2, default=str))
    print(f"latency: {j.last_latency_ms:.1f} ms")
