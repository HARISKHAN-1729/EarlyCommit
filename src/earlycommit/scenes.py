"""Random tabletop scenes with discrete, language-groundable attributes.

Each object has a colour, a shape, and a coarse table region (column x row).
Continuous x/y positions are kept for the kinematic arm in the demo.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, asdict

COLORS = ["red", "blue", "green", "yellow", "white", "black"]
SHAPES = ["cup", "block", "ball", "bowl", "bottle"]
COLUMNS = ["left", "middle", "right"]   # along table x-axis (robot's view)
ROWS = ["front", "back"]                # along table y-axis

# Table extent in metres (centre of table = origin); used by the demo arm.
_X_EDGES = [-0.30, -0.10, 0.10, 0.30]
_Y_EDGES = [-0.20, 0.00, 0.20]


@dataclass(frozen=True)
class Obj:
    id: str
    color: str
    shape: str
    column: str
    row: str
    x: float
    y: float

    def describe(self) -> str:
        """Option text shown to the judge for this object."""
        return f"{self.color} {self.shape}, {self.column} {self.row} of the table"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Scene:
    id: str
    objects: list[Obj]

    def to_dict(self) -> dict:
        return {"id": self.id, "objects": [o.to_dict() for o in self.objects]}

    @staticmethod
    def from_dict(d: dict) -> "Scene":
        return Scene(id=d["id"], objects=[Obj(**o) for o in d["objects"]])


def random_scene(rng: random.Random, scene_id: str,
                 n_min: int = 3, n_max: int = 7,
                 distractor_bias: float = 0.5) -> Scene:
    """Sample a scene. `distractor_bias` > 0 makes objects share attributes more
    often, so that referring expressions need several words to resolve (which is
    exactly where early commitment is interesting)."""
    n = rng.randint(n_min, n_max)
    cells = [(c, r) for c in COLUMNS for r in ROWS]
    rng.shuffle(cells)
    # Allow at most 2 objects per cell so positions stay separable.
    cells = (cells * 2)[:n]

    palette = rng.sample(COLORS, k=max(2, round(len(COLORS) * (1 - distractor_bias))))
    shapes = rng.sample(SHAPES, k=max(2, round(len(SHAPES) * (1 - distractor_bias))))

    objs, used = [], set()
    for i, (col, row) in enumerate(cells):
        for _ in range(20):  # avoid exact duplicates in the same cell
            color, shape = rng.choice(palette), rng.choice(shapes)
            if (color, shape, col, row) not in used:
                break
        used.add((color, shape, col, row))
        ci, ri = COLUMNS.index(col), ROWS.index(row)
        x = rng.uniform(_X_EDGES[ci] + 0.03, _X_EDGES[ci + 1] - 0.03)
        y = rng.uniform(_Y_EDGES[ri] + 0.03, _Y_EDGES[ri + 1] - 0.03)
        objs.append(Obj(f"obj_{i}", color, shape, col, row, round(x, 3), round(y, 3)))
    return Scene(scene_id, objs)
