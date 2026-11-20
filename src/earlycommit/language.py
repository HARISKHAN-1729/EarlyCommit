"""Templated referring expressions + an incremental oracle listener.

The oracle knows, after every word, which objects are still consistent with
what has been said. From that we get, for free and without annotation:
  * a soft label per prefix (uniform over the consistent set),
  * the oracle commit point (first word at which one object remains),
  * whether the full utterance is ambiguous (never resolves).
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from .scenes import COLORS, SHAPES, COLUMNS, ROWS, Obj, Scene

VERBS = ["pick up", "grab", "hand me", "give me", "can you get", "please take"]
# Trailing material people add after the referent is already clear.
TAILS = ["please", "for me", "over there", "for me please", "just there", "if you can",
         "the one I mean", "thanks"]

# Words that carry attribute information, mapped to (attribute, value).
_LEXICON: dict[str, tuple[str, str]] = {}
for c in COLORS:
    _LEXICON[c] = ("color", c)
for s in SHAPES:
    _LEXICON[s] = ("shape", s)
for c in COLUMNS:
    _LEXICON[c] = ("column", c)
for r in ROWS:
    _LEXICON[r] = ("row", r)
_LEXICON["center"] = ("column", "middle")


def _phrase(attr: str, obj: Obj) -> list[str]:
    """Surface forms for one attribute, chosen to be realistic but parseable."""
    if attr == "color":
        return [obj.color]
    if attr == "shape":
        return [obj.shape]
    if attr == "column":
        return ["in", "the", "middle"] if obj.column == "middle" else ["on", "the", obj.column]
    if attr == "row":
        return ["at", "the", obj.row]
    raise ValueError(attr)


@dataclass
class Utterance:
    words: list[str]
    target: str | None          # None when the utterance is ambiguous
    candidates_per_prefix: list[list[str]]  # consistent object ids after word i
    commit_index: int | None    # first i (0-based) with a single candidate
    ambiguous: bool

    @property
    def text(self) -> str:
        return " ".join(self.words)


def consistent(objects: list[Obj], words: list[str]) -> list[str]:
    constraints = [_LEXICON[w] for w in words if w in _LEXICON]
    return [o.id for o in objects
            if all(getattr(o, a) == v for a, v in constraints)]


def _distinguishing_attrs(scene: Scene, target: Obj) -> list[str]:
    """Greedy minimal-ish attribute set that singles out the target, if any."""
    attrs, remaining = [], [o for o in scene.objects if o.id != target.id]
    for a in ["color", "shape", "column", "row"]:
        if not remaining:
            break
        keep = [o for o in remaining if getattr(o, a) == getattr(target, a)]
        if len(keep) < len(remaining):
            attrs.append(a)
            remaining = keep
    return attrs if not remaining else ["color", "shape", "column", "row"]


def make_utterance(scene: Scene, rng: random.Random,
                   p_ambiguous: float = 0.2, p_overspecify: float = 0.5,
                   p_tail: float = 0.5, target_id: str | None = None) -> Utterance:
    # target_id=None keeps the original random draw, so generated datasets are unchanged.
    target = rng.choice(scene.objects) if target_id is None else \
        next(o for o in scene.objects if o.id == target_id)
    attrs = _distinguishing_attrs(scene, target)

    if rng.random() < p_ambiguous and len(attrs) > 1:
        attrs = attrs[:-1]                           # drop one -> may not resolve
    elif rng.random() < p_overspecify:
        extra = [a for a in ["color", "shape", "column", "row"] if a not in attrs]
        if extra:
            attrs = attrs + [rng.choice(extra)]      # redundant info, like people do

    # Word order matters for earliness. Two realistic orders:
    #   "the red cup on the left"          (pre-nominal colour, then location)
    #   "the cup on the left, the red one" (afterthought colour)
    locs = [a for a in attrs if a in ("column", "row")]
    rng.shuffle(locs)
    head = [target.shape] if "shape" in attrs else ["one"]
    words = rng.choice(VERBS).split() + ["the"]
    if "color" in attrs and rng.random() < 0.3 and head != ["one"]:
        words += head + [w for a in locs for w in _phrase(a, target)]
        words += ["the", target.color, "one"]
    else:
        if "color" in attrs:
            words += [target.color]
        words += head + [w for a in locs for w in _phrase(a, target)]
    if rng.random() < p_tail:
        words += rng.choice(TAILS).split()

    cands = [consistent(scene.objects, words[: i + 1]) for i in range(len(words))]
    assert cands[-1], f"contradictory utterance generated: {' '.join(words)}"
    commit = next((i for i, c in enumerate(cands) if len(c) == 1), None)
    ambiguous = len(cands[-1]) != 1
    return Utterance(words, None if ambiguous else target.id, cands, commit, ambiguous)
