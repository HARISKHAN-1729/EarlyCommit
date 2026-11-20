"""End-to-end pipeline.

  python -m earlycommit.run data                 # write train/val/test JSONL
  python -m earlycommit.run eval --judge mock    # full eval with the mock judge
  python -m earlycommit.run eval --judge laya --repo <your-finetuned-repo>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .calibrate import apply_temperature, fit_temperature, reliability
from .core import SECONDS_PER_WORD, make_episodes, run_streaming, write_finetune_jsonl
from .judge import CachedJudge, KeywordJudge, LayaJudge, MockJudge

OUT = Path("outputs")
SPLITS = {"train": (4000, 1), "val": (500, 2), "test": (500, 3)}  # (n episodes, seed)


def cmd_data(args):
    out = OUT / args.out_dir
    for name, (n, seed) in SPLITS.items():
        aug = args.augment if name in ("train", "val") else 0.0   # test stays clean
        eps = make_episodes(n, seed, augment=aug)
        rows = write_finetune_jsonl(eps, out / f"{name}.jsonl", seed=seed)
        with (out / f"{name}_episodes.jsonl").open("w") as f:
            for ep in eps:
                f.write(json.dumps(ep.to_dict()) + "\n")
        amb = sum(e.utt.ambiguous for e in eps) / n
        print(f"{name}: {n} episodes, {rows} prefix rows, {amb:.0%} ambiguous, augment={aug}")


def collect(judge, eps):
    dists, labels = [], []
    for ep in eps:
        for i in range(1, len(ep.utt.words)):
            y = ep.soft_label(i)
            if y:
                dists.append(judge.predict(ep.scene, ep.utt.words[: i + 1]))
                labels.append(y)
    return dists, labels


def evaluate(judge, eps, T, taus):
    rows = []
    for tau in taus:
        res = [(ep, run_streaming(ep, judge, tau, T)) for ep in eps]
        clear = [(e, r) for e, r in res if not e.utt.ambiguous]
        amb = [(e, r) for e, r in res if e.utt.ambiguous]
        committed = [(e, r) for e, r in clear if r.committed_to is not None]
        rows.append({
            "tau": tau,
            "commit_rate": len(committed) / max(1, len(clear)),
            "accuracy_at_commit": np.mean([r.committed_to == e.utt.target
                                           for e, r in committed]) if committed else float("nan"),
            "mean_time_saved_s": float(np.mean([r.time_saved_s() for _, r in committed])) if committed else 0.0,
            "oracle_time_saved_s": float(np.mean(
                [(len(e.utt.words) - 1 - e.utt.commit_index) * SECONDS_PER_WORD for e, _ in committed])) if committed else 0.0,
            "false_commit_on_ambiguous": np.mean([r.committed_to is not None for _, r in amb]) if amb else float("nan"),
            "n_clear": len(clear), "n_ambiguous": len(amb),
            "n_committed": len(committed),
            "n_correct": int(sum(r.committed_to == e.utt.target for e, r in committed)),
            "n_false_commit_ambiguous": int(sum(r.committed_to is not None for _, r in amb)),
        })
    return rows


def cmd_eval(args):
    judge = CachedJudge(MockJudge() if args.judge == "mock" else LayaJudge(args.repo, args.subfolder, args.device))
    val, test = make_episodes(args.n_val, 2), make_episodes(args.n_test, 3)

    vd, vl = collect(judge, val)
    T = fit_temperature(vd, vl)
    td, tl = collect(judge, test)
    ece_raw, bins_raw = reliability(td, tl)
    ece_cal, bins_cal = reliability([apply_temperature(p, T) for p in td], tl)
    print(f"temperature T={T:.2f} | test ECE raw={ece_raw:.3f} -> calibrated={ece_cal:.3f}")

    taus = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    curves = {"raw": evaluate(judge, test, 1.0, taus), "calibrated": evaluate(judge, test, T, taus)}
    for name, rows in curves.items():
        print(f"\n[{name}]  tau  commit  acc@commit  saved_s(oracle)  false_commit_ambig")
        for r in rows:
            print(f"        {r['tau']:.2f}  {r['commit_rate']:.2f}    {r['accuracy_at_commit']:.3f}"
                  f"     {r['mean_time_saved_s']:.2f} ({r['oracle_time_saved_s']:.2f})"
                  f"       {r['false_commit_on_ambiguous']:.3f}")

    OUT.mkdir(exist_ok=True)
    json.dump({"T": T, "ece_raw": ece_raw, "ece_cal": ece_cal, "bins_raw": bins_raw,
               "bins_cal": bins_cal, "curves": curves},
              (OUT / f"results_{args.judge}.json").open("w"), indent=2, default=float)
    plot(bins_raw, bins_cal, curves, OUT / f"figures_{args.judge}.png")


def plot(bins_raw, bins_cal, curves, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
    ax[0].plot([0, 1], [0, 1], "k:", lw=1)
    for bins, style, lab in [(bins_raw, "s--", "raw"), (bins_cal, "o-", "calibrated")]:
        ax[0].plot([b[0] for b in bins], [b[1] for b in bins], style, label=lab)
    ax[0].set(xlabel="confidence", ylabel="empirical accuracy", title="Reliability")
    ax[0].legend()
    for name, style in [("raw", "s--"), ("calibrated", "o-")]:
        r = curves[name]
        ax[1].plot([x["mean_time_saved_s"] for x in r], [x["accuracy_at_commit"] for x in r],
                   style, label=name)
        for x in r:
            ax[1].annotate(f"{x['tau']}", (x["mean_time_saved_s"], x["accuracy_at_commit"]), fontsize=7)
    ax[1].set(xlabel="time saved before utterance ends (s)", ylabel="accuracy at commit",
              title="Earliness vs accuracy (labels = tau)")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    print(f"saved {path}")


def cmd_ood(args):
    """Robustness table: every judge x every condition. The temperature is fitted on
    validation instructions, a fraction `--val-augment` of them rewritten with the
    word pool `--val-pool` (default: the held-out calibration pool)."""
    import random as _r
    from .core import Episode
    from .language_ood import KINDS, STRESS, perturb
    judges = {"keyword": KeywordJudge()}
    if args.judge == "laya":
        judges["laya"] = CachedJudge(LayaJudge(args.repo, args.subfolder, args.device))
    else:
        judges["mock"] = CachedJudge(MockJudge())
    val = make_episodes(args.n_val, 2, augment=args.val_augment, pool=args.val_pool)
    test = make_episodes(args.n_test, 3)
    print(f"temperature fitted on val with augment={args.val_augment} from pool '{args.val_pool}'")
    temps = {"keyword": 1.0}
    for name, j in judges.items():
        if name != "keyword":
            temps[name] = fit_temperature(*collect(j, val))
    conds = {"in-distribution": test}
    for kind in KINDS + STRESS:
        rng = _r.Random(11)
        conds[kind] = [Episode(ep.scene, perturb(ep.utt, [o.id for o in ep.scene.objects], rng, kind))
                       for ep in test]
    taus = [float(t) for t in args.taus.split(",")]
    table = []
    print(f"temperatures {({k: round(v, 2) for k, v in temps.items()})}")
    for cname, eps in conds.items():
        print(f"\n== {cname} ==")
        for jname, j in judges.items():
            d, l = collect(j, eps)
            ece, bins = reliability([apply_temperature(p, temps[jname]) for p in d], l)
            print(f"  {jname:8s} ECE={ece:.3f}")
            print(f"  {'':8s} tau   commit  acc@commit  correct  wrong  saved_s(oracle)  false_commit_ambig")
            for r in evaluate(j, eps, temps[jname], taus):
                acc = r["accuracy_at_commit"]
                acc0 = 0.0 if acc != acc else acc          # nan -> 0 when nothing committed
                r.update(condition=cname, judge=jname, ece=ece, bins=bins,
                         correct_rate=r["commit_rate"] * acc0,
                         wrong_rate=r["commit_rate"] * (1 - acc0) if r["commit_rate"] else 0.0)
                table.append(r)
                print(f"  {'':8s} {r['tau']:.2f}  {r['commit_rate']:.2f}    {acc:.3f}      "
                      f"{r['correct_rate']:.2f}     {r['wrong_rate']:.2f}   "
                      f"{r['mean_time_saved_s']:.2f} ({r['oracle_time_saved_s']:.2f})      "
                      f"{r['false_commit_on_ambiguous']:.3f}")
    for jname, j in judges.items():
        if isinstance(j, CachedJudge):
            print(f"[{jname}] model calls: {j.misses}, served from cache: {j.hits}")
    out_name = args.out or f"ood_{args.judge}.json"
    OUT.mkdir(exist_ok=True)
    json.dump({"taus": taus, "temperatures": temps, "val_pool": args.val_pool,
               "val_augment": args.val_augment, "table": table},
              (OUT / out_name).open("w"), indent=2, default=float)
    print(f"saved {OUT / out_name}")
    for kind in ["synonym", "asr", "all", "asr-confusable"]:
        ep = next((e for e in conds[kind] if e.utt.words != test[conds[kind].index(e)].utt.words), conds[kind][0])
        print(f"example [{kind}]: {' '.join(ep.utt.words)}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("data")
    d.add_argument("--augment", type=float, default=0.0)
    d.add_argument("--out-dir", default="data")
    e = sub.add_parser("eval")
    e.add_argument("--judge", choices=["mock", "laya"], default="mock")
    e.add_argument("--repo", default="convaiinnovations/laya")
    e.add_argument("--subfolder", default=None)
    e.add_argument("--device", default=None)
    e.add_argument("--n-val", type=int, default=300)
    e.add_argument("--n-test", type=int, default=300)
    o = sub.add_parser("ood")
    o.add_argument("--judge", choices=["mock", "laya"], default="mock")
    o.add_argument("--repo", default="convaiinnovations/laya")
    o.add_argument("--subfolder", default=None)
    o.add_argument("--device", default=None)
    o.add_argument("--n-val", type=int, default=300)
    o.add_argument("--n-test", type=int, default=300)
    o.add_argument("--taus", default="0.6,0.8,0.9,0.95,0.99")
    o.add_argument("--val-augment", type=float, default=0.5)
    o.add_argument("--val-pool", choices=["train", "calib"], default="calib")
    o.add_argument("--out", default=None, help="output file name inside outputs/")
    args = ap.parse_args()
    {"data": cmd_data, "eval": cmd_eval, "ood": cmd_ood}[args.cmd](args)


if __name__ == "__main__":
    main()
