"""Surface variation: synonyms, speech-recognition slips, disfluencies.

Every perturbation rewrites an utterance word by word while keeping the ground
truth: each surface word inherits the oracle candidate set of the original word
it came from, so labels stay exact.

Word pools are split three ways, sharing no surface form:
  TRAIN : used for training augmentation
  CALIB : used ONLY for fitting the temperature (variation the model never trained on)
  TEST  : used ONLY for evaluation
If the temperature is fitted on variation the model was trained on, the model
looks competent on it, T ~ 1, and there is no safety margin left for genuinely
new words. CALIB provides a held-out stand-in for "unfamiliar language".

"asr-confusable" is a separate stress test: slips that turn one attribute word
into ANOTHER valid attribute word (black -> "back", bowl -> "ball"). A listener
cannot recover the intent from the words alone, so errors here are largely
irreducible; it is reported separately and never used for training.
"""
from __future__ import annotations

import random

from .language import Utterance

SYN = {
    "train": {"red": ["scarlet"], "blue": ["navy"], "green": ["lime"], "yellow": ["golden"],
              "white": ["ivory"], "black": ["jet"], "cup": ["tumbler"], "block": ["brick"],
              "ball": ["orb"], "bowl": ["basin"], "bottle": ["jar"], "front": ["close"],
              "back": ["rear"]},
    "test": {"red": ["crimson"], "blue": ["azure"], "green": ["emerald"], "yellow": ["lemon"],
             "white": ["snowy"], "black": ["dark"], "cup": ["mug"], "block": ["cube"],
             "ball": ["sphere"], "bowl": ["dish"], "bottle": ["flask"], "front": ["near"],
             "back": ["far"]},
    "calib": {"red": ["ruby"], "blue": ["cobalt"], "green": ["olive"], "yellow": ["amber"],
              "white": ["pearl"], "black": ["ebony"], "cup": ["beaker"], "block": ["slab"],
              "ball": ["globe"], "bowl": ["pot"], "bottle": ["vial"], "front": ["nearby"],
              "back": ["distant"]},
}
# Slips into NON-attribute words only (recoverable from context in principle).
ASR = {
    "train": {"red": ["rad"], "blue": ["blu"], "green": ["greene"], "yellow": ["yello"],
              "white": ["wite"], "black": ["blak"], "cup": ["cap"], "ball": ["bawl"],
              "bowl": ["bole"], "block": ["blok"], "bottle": ["model"], "right": ["rite"],
              "left": ["laughed"], "front": ["frunt"], "back": ["pack"], "middle": ["midal"]},
    "test": {"red": ["read"], "blue": ["blew"], "green": ["grin"], "yellow": ["jello"],
             "white": ["whyte"], "black": ["flack"], "cup": ["cop"], "ball": ["bull"],
             "bowl": ["bold"], "block": ["clock"], "bottle": ["battle"], "right": ["write"],
             "left": ["lift"], "front": ["fronts"], "back": ["bach"], "middle": ["meddle"]},
    "calib": {"red": ["redd"], "blue": ["bloo"], "green": ["grean"], "yellow": ["yelo"],
              "white": ["wait"], "black": ["blac"], "cup": ["cub"], "ball": ["baul"],
              "bowl": ["boal"], "block": ["blog"], "bottle": ["bottel"], "right": ["rigt"],
              "left": ["lefft"], "front": ["frant"], "back": ["bak"], "middle": ["midle"]},
}
CONFUSABLE = {"black": ["back"], "ball": ["bowl"], "bowl": ["ball"], "block": ["black"]}
FILLERS = {"train": ["uh", "um"], "test": ["er", "like", "you know"], "calib": ["hmm", "so"]}

KINDS = ["synonym", "disfluency", "asr", "all"]          # evaluated on the TEST pools
STRESS = ["asr-confusable"]                               # reported separately


def perturb(utt: Utterance, all_ids: list[str], rng: random.Random, kind: str,
            pool: str = "test", p_word: float = 0.6, p_filler: float = 0.25) -> Utterance:
    use_syn = kind in ("synonym", "all")
    use_asr = kind in ("asr", "all")
    use_dis = kind in ("disfluency", "all")
    use_conf = kind == "asr-confusable"
    out_words, out_cands = [], []
    before = list(all_ids)
    for i, w in enumerate(utt.words):
        cand_i = utt.candidates_per_prefix[i]
        if use_dis and i > 0 and rng.random() < p_filler:
            if rng.random() < 0.5:
                for f in rng.choice(FILLERS[pool]).split():
                    out_words.append(f); out_cands.append(before)
            else:
                out_words.append(out_words[-1]); out_cands.append(before)   # repetition
        options = []
        if use_syn:
            options += SYN[pool].get(w, [])
        if use_asr:
            options += ASR[pool].get(w, [])
        if use_conf:
            options += CONFUSABLE.get(w, [])
        s = rng.choice(options) if options and rng.random() < p_word else w
        out_words.append(s)
        out_cands.append(cand_i)
        before = cand_i
    commit = next((i for i, c in enumerate(out_cands) if len(c) == 1), None)
    return Utterance(out_words, utt.target, out_cands, commit, utt.ambiguous)


def _check_disjoint():
    from itertools import combinations
    pools = ["train", "calib", "test"]
    for name, d in [("SYN", SYN), ("ASR", ASR),
                    ("FILLERS", {k: {"_": v} for k, v in FILLERS.items()})]:
        sets = {k: {w for v in d[k].values() for w in v} for k in pools}
        for a, b in combinations(pools, 2):
            assert not sets[a] & sets[b], f"{name} {a}/{b} overlap: {sets[a] & sets[b]}"


_check_disjoint()
