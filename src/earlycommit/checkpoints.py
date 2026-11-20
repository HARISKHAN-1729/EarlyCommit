"""Unpack a fine-tuned checkpoint zip and put it where the notebooks expect it.

  python -m earlycommit.checkpoints --zip checkpoints/laya-earlycommit.zip     --name laya-earlycommit
  python -m earlycommit.checkpoints --zip checkpoints/laya-earlycommit-aug.zip --name laya-earlycommit-aug

The zip may contain the model folder at any depth. The folder that holds
`rl_agent_config.json` is moved to checkpoints/<name>/ and its files are checked.
"""
from __future__ import annotations

import argparse
import shutil
import tempfile
import zipfile
from pathlib import Path

REQUIRED = ["rl_agent_config.json", "model.safetensors", "encoder", "tokenizer"]


def prepare(zip_path: str, name: str, root: str = "checkpoints") -> Path:
    zp, dest = Path(zip_path), Path(root) / name
    if not zp.exists():
        raise SystemExit(f"{zp} not found. Upload it first (see README, section 'Checkpoints').")
    with zipfile.ZipFile(zp) as z:
        bad = z.testzip()
        if bad is not None:
            raise SystemExit(f"{zp} is damaged (first bad member: {bad}). Re-download or re-upload it.")
        with tempfile.TemporaryDirectory() as tmp:
            z.extractall(tmp)
            hits = list(Path(tmp).rglob("rl_agent_config.json"))
            if not hits:
                raise SystemExit(f"no rl_agent_config.json inside {zp}; is this a Laya checkpoint?")
            if dest.exists():
                shutil.rmtree(dest)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(hits[0].parent), str(dest))
    missing = [f for f in REQUIRED if not (dest / f).exists()]
    if missing:
        raise SystemExit(f"{dest} is missing {missing}")
    print(f"ready: {dest} -> {sorted(p.name for p in dest.iterdir())}")
    return dest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--root", default="checkpoints")
    a = ap.parse_args()
    prepare(a.zip, a.name, a.root)


if __name__ == "__main__":
    main()
