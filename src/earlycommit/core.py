"""Episodes, the streaming commit policy, and the fine-tuning dataset builder."""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

from .judge import Judge, build_questions, build_state
from .language import Utterance, make_utterance
from .scenes import Scene, random_scene
from .calibrate import apply_temperature

SECONDS_PER_WORD = 0.30   # conversational English is ~2.5-3.5 words/s


@dataclass
class Episode:
    scene: Scene
    utt: Utterance

    def soft_label(self, i: int) -> dict[str, float]:
        c = self.utt.candidates_per_prefix[i]
        return {k: 1.0 / len(c) for k in c} if c else {}

    def to_dict(self) -> dict:
        u = self.utt
        return {"scene": self.scene.to_dict(), "words": u.words, "target": u.target,
                "candidates_per_prefix": u.candidates_per_prefix,
                "commit_index": u.commit_index, "ambiguous": u.ambiguous}


def make_episodes(n: int, seed: int, augment: float = 0.0, pool: str = "train") -> list[Episode]:
    """`augment` = fraction of episodes rewritten with word pool `pool`
    ("train" for training data, "calib" for temperature fitting)."""
    from .language_ood import perturb
    rng = random.Random(seed)
    aug_rng = random.Random(seed + 1000)      # separate stream: augment=0 reproduces old data
    eps = []
    while len(eps) < n:
        sc = random_scene(rng, f"s{seed}_{len(eps)}")
        utt = make_utterance(sc, rng)
        if augment and aug_rng.random() < augment:
            kind = aug_rng.choice(["synonym", "asr", "disfluency", "all"])
            utt = perturb(utt, [o.id for o in sc.objects], aug_rng, kind, pool=pool)
        eps.append(Episode(sc, utt))
    return eps


@dataclass
class CommitResult:
    committed_to: str | None
    commit_index: int | None      # word index at which we committed
    n_words: int
    judge_calls: int

    def time_saved_s(self) -> float:
        """Seconds between commit and end of utterance (0 if no commit)."""
        if self.commit_index is None:
            return 0.0
        return (self.n_words - 1 - self.commit_index) * SECONDS_PER_WORD


def run_streaming(ep: Episode, judge: Judge, tau: float, T: float = 1.0,
                  min_words: int = 2) -> CommitResult:
    """Feed words one at a time; commit when calibrated max prob >= tau.
    `min_words` skips the verb, which carries no referent information."""
    words = ep.utt.words
    for i in range(len(words)):
        if i + 1 < min_words:
            continue
        p = apply_temperature(judge.predict(ep.scene, words[: i + 1]), T)
        k = max(p, key=p.get)
        if p[k] >= tau:
            return CommitResult(k, i, len(words), i + 1)
    return CommitResult(None, None, len(words), len(words))


def write_finetune_jsonl(episodes: list[Episode], path: Path, seed: int = 0,
                         hard_samples: int = 1) -> int:
    """One row per (episode, prefix). Each row carries the soft label and
    `hard_samples` labels drawn from it: training a proper scoring rule on
    labels sampled from the soft target pushes the model toward the soft
    target itself, so ambiguous prefixes learn to stay uncertain."""
    rng = random.Random(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w") as f:
        for ep in episodes:
            for i in range(1, len(ep.utt.words)):
                soft = ep.soft_label(i)
                if not soft:
                    continue
                keys, w = zip(*soft.items())
                for _ in range(hard_samples):
                    f.write(json.dumps({
                        "state": build_state(ep.scene, ep.utt.words[: i + 1]),
                        "questions": build_questions(ep.scene),
                        "label": {"target": rng.choices(keys, weights=w)[0]},
                        "soft_label": {"target": soft},
                        "meta": {"scene_id": ep.scene.id, "prefix_len": i + 1},
                    }) + "\n")
                    n += 1
    return n
