"""Fine-tune Laya on EarlyCommit prefixes (the recipe used in the paper).

  python -m earlycommit.train --out checkpoints/laya-earlycommit                  # no augmentation
  python -m earlycommit.train --out checkpoints/laya-earlycommit-aug --augment 0.5  # train-pool augmentation

Steps: (1) generate the labelled prefixes, (2) tokenise them with Laya's own
sequence builder, (3) fine-tune the whole model with Laya's RLCD recipe
(policy gradient on a proper scoring rule + soft cross-entropy), (4) fit Laya's
internal temperature on held-out validation prefixes, (5) save a checkpoint
folder that `laya.Agent(folder)` loads directly.

Paper setting: 2 epochs, effective batch 64, AdamW (2.5e-5 encoder, 1e-4 head,
weight decay 0.01, cosine), clip 1.0, bfloat16, one A100. On GPUs without bfloat16
(e.g. T4) float16 with a gradient scaler is used instead.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path


def generate_data(out_dir: Path, augment: float):
    from .core import make_episodes, write_finetune_jsonl
    from .run import SPLITS
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, (n, seed) in SPLITS.items():
        if name == "test":
            continue
        aug = augment if name in ("train", "val") else 0.0
        eps = make_episodes(n, seed, augment=aug)
        rows = write_finetune_jsonl(eps, out_dir / f"{name}.jsonl", seed=seed)
        print(f"{name}: {n} instructions -> {rows} prefix examples (augment={aug})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output checkpoint folder")
    ap.add_argument("--augment", type=float, default=0.0, help="0.5 for the augmented model")
    ap.add_argument("--base", default="convaiinnovations/laya")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--micro-batch", type=int, default=32)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file, save_file
    from transformers import AutoTokenizer
    from laya.agent import _fix_tokenizer_config
    from laya.common import QTYPES, build_model, build_sequence, proper_reward, render_options

    assert torch.cuda.is_available(), "A GPU is required (Colab: Runtime -> Change runtime type -> GPU)."
    torch.manual_seed(args.seed); random.seed(args.seed)
    device = torch.device("cuda")
    use_bf16 = torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=not use_bf16)
    print(f"GPU: {torch.cuda.get_device_name(0)} | mixed precision: {'bf16' if use_bf16 else 'fp16'}")

    # 1. data
    data_dir = Path(args.out + "_data")
    generate_data(data_dir, args.augment)

    # 2. tokenise with Laya's own builder, keeping the checkpoint's max_len/head_max_len
    model_dir = snapshot_download(args.base)
    _fix_tokenizer_config(model_dir)
    tok = AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))
    cfg = json.load(open(os.path.join(model_dir, "rl_agent_config.json")))
    print("max_len", cfg["max_len"], "| head_max_len", cfg["head_max_len"])

    def to_items(path):
        items, dropped = [], 0
        for line in open(path):
            row = json.loads(line)
            q = row["questions"]["target"]; crit = q["criteria"]; keys = list(crit.keys())
            soft = row["soft_label"]["target"]
            target = [soft.get(k, 0.0) for k in keys]
            seq, markers = build_sequence(tok, row["state"], {"t": "choice", "ins": q["instructions"], "crit": crit},
                                          cfg["max_len"], cfg["head_max_len"])
            if len(markers) != len(render_options({"t": "choice", "crit": crit})):
                dropped += 1; continue
            items.append({"ids": seq, "markers": markers, "qtype": QTYPES["choice"], "target": target,
                          "label": target.index(max(target)), "resolved": max(target) == 1.0})
        return items, dropped

    train_items, d1 = to_items(data_dir / "train.jsonl")
    val_items, d2 = to_items(data_dir / "val.jsonl")
    print(f"train {len(train_items)} (dropped {d1}) | val {len(val_items)} (dropped {d2}) | "
          f"longest {max(len(it['ids']) for it in train_items)} tokens")
    assert d1 < 0.01 * len(train_items), "Many rows truncated: check max_len/head_max_len."

    # 3. model
    model = build_model(cfg, encoder_dir=os.path.join(model_dir, "encoder"))
    model.load_state_dict(load_file(os.path.join(model_dir, "model.safetensors")), strict=True)
    model.to(device)

    def collate(items):
        n, L = len(items), max(len(it["ids"]) for it in items)
        kmax = max(len(it["markers"]) for it in items)
        ids = torch.full((n, L), tok.pad_token_id, dtype=torch.long)
        att = torch.zeros((n, L), dtype=torch.long)
        mpos = torch.zeros((n, kmax), dtype=torch.long)
        mmask = torch.zeros((n, kmax), dtype=torch.bool)
        target = torch.zeros((n, kmax), dtype=torch.float32)
        for i, it in enumerate(items):
            ids[i, :len(it["ids"])] = torch.tensor(it["ids"]); att[i, :len(it["ids"])] = 1
            k = len(it["markers"])
            mpos[i, :k] = torch.tensor(it["markers"]); mmask[i, :k] = True
            target[i, :len(it["target"])] = torch.tensor(it["target"])
        return {"input_ids": ids, "attention_mask": att, "marker_pos": mpos, "marker_mask": mmask,
                "target": target, "qtype": torch.tensor([it["qtype"] for it in items])}

    def forward(b):
        with torch.autocast("cuda", dtype=amp_dtype):
            logits, act = model(b["input_ids"].to(device), b["attention_mask"].to(device),
                                b["marker_pos"].to(device), b["marker_mask"].to(device), b["qtype"].to(device))
        return logits.float(), act

    @torch.no_grad()
    def evaluate(items, bs=64):
        model.eval(); nll = 0.0; n = correct = n_res = 0
        for i in range(0, len(items), bs):
            chunk = items[i:i + bs]; b = collate(chunk); logits, _ = forward(b)
            mask = b["marker_mask"].to(device)
            logp = torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)
            nll += -(b["target"].to(device) * logp).sum(-1).sum().item(); n += len(chunk)
            pred = logits.masked_fill(~mask, -1e4).argmax(-1).cpu()
            for j, it in enumerate(chunk):
                if it["resolved"]:
                    n_res += 1; correct += int(pred[j].item() == it["label"])
        model.train()
        return nll / n, correct / max(1, n_res)

    print("before fine-tuning: val soft-NLL %.3f | resolved acc %.3f" % evaluate(val_items[:2000]))

    # 4. train (RLCD recipe)
    GROUP_SIZE, LR_ENC, LR_HEAD, S0, S1 = 4, 2.5e-5, 1.0e-4, 0.4, 0.1
    enc = [p for n_, p in model.named_parameters() if "encoder." in n_]
    head = [p for n_, p in model.named_parameters() if "encoder." not in n_]
    opt = torch.optim.AdamW([{"params": enc, "lr": LR_ENC}, {"params": head, "lr": LR_HEAD}], weight_decay=0.01)
    total = (len(train_items) // (args.micro_batch * args.grad_accum)) * args.epochs
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, total), eta_min=1e-6)
    model.train(); t0 = time.time()
    for epoch in range(args.epochs):
        random.Random(42 + epoch).shuffle(train_items)
        sigma = S0 + (S1 - S0) * (epoch / max(1, args.epochs - 1))
        opt.zero_grad(set_to_none=True); step = 0
        for bi in range(0, len(train_items), args.micro_batch):
            b = collate(train_items[bi:bi + args.micro_batch])
            logits, act = forward(b)
            mask = b["marker_mask"].to(device); k = mask.sum(-1, keepdim=True).float()
            target = b["target"].to(device)
            eps = torch.randn((GROUP_SIZE,) + logits.shape, device=device) * sigma * mask
            eps = (eps - eps.sum(-1, keepdim=True) / k) * mask
            z = logits.detach().unsqueeze(0) + eps
            q = torch.softmax(z.masked_fill(~mask, -1e4), -1)
            with torch.no_grad():
                r = proper_reward(q, target.unsqueeze(0), b["qtype"].to(device), mask, w_sph=0.75, w_rps=1.0)
                adv = r - r.mean(0, keepdim=True); adv = adv / (adv.std() + 1e-6)
            logp = -(((z - logits.unsqueeze(0)) ** 2) * mask).sum(-1) / (2 * sigma ** 2)
            loss_rl = -(adv * logp).mean()
            loss_ce = -(target * torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)).sum(-1).mean()
            loss = (loss_rl + loss_ce) / args.grad_accum + 0.0 * act.sum()
            scaler.scale(loss).backward(); step += 1
            if step % args.grad_accum == 0 or bi + args.micro_batch >= len(train_items):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update(); sched.step(); opt.zero_grad(set_to_none=True)
            if step % 100 == 0:
                print(f"epoch {epoch+1} step {step} | CE {loss_ce.item():.3f} | reward {r.mean().item():.3f} | {time.time()-t0:.0f}s")
        print(f"=== epoch {epoch+1}: val soft-NLL %.3f | resolved acc %.3f ===" % evaluate(val_items[:2000]))

    # 5. internal temperature on held-out validation prefixes (disjoint from the monitoring slice)
    @torch.no_grad()
    def collect_logits(items, bs=64):
        model.eval(); Z, T = [], []
        for i in range(0, len(items), bs):
            chunk = items[i:i + bs]; logits, _ = forward(collate(chunk))
            for j, it in enumerate(chunk):
                kk = len(it["markers"]); Z.append(logits[j, :kk].cpu()); T.append(torch.tensor(it["target"]))
        return Z, T
    Z, Tt = collect_logits(val_items[2000:3500])
    kmax = max(len(z) for z in Z)
    Zm = torch.full((len(Z), kmax), -1e4); Tm = torch.zeros((len(Z), kmax))
    for i, (z, t) in enumerate(zip(Z, Tt)):
        Zm[i, :len(z)] = z; Tm[i, :len(t)] = t
    log_t = torch.zeros(1, requires_grad=True)
    lb = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)
    def closure():
        lb.zero_grad(); l = -(Tm * torch.log_softmax(Zm / log_t.exp(), -1)).sum(-1).mean(); l.backward(); return l
    lb.step(closure)
    t_choice = float(torch.clamp(log_t.exp(), 0.5, 5.0))
    print("internal choice temperature:", round(t_choice, 3))

    # 6. save in Laya's format
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    save_file({k: v.half().contiguous().cpu() for k, v in model.state_dict().items()}, str(out / "model.safetensors"))
    model.encoder.config.save_pretrained(str(out / "encoder"))
    tok.save_pretrained(str(out / "tokenizer"))
    temps = cfg.get("temperature", [1.0, 1.0, 1.0])
    temps = list(temps) if isinstance(temps, (list, tuple)) else [float(temps)] * 3
    temps[QTYPES["choice"]] = t_choice
    cfg.update({"fine_tuned": True, "model_name": "laya-earlycommit" + ("-aug" if args.augment else ""),
                "temperature": temps})
    cfg.pop("temperature_by_options", None)
    json.dump(cfg, open(out / "rl_agent_config.json", "w"), indent=2)
    print("saved checkpoint to", out, sorted(os.listdir(out)))


if __name__ == "__main__":
    main()
