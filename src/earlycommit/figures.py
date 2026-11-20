"""Paper figures + table from the two final `ood` result files.

  python -m earlycommit.figures --noaug outputs/final_noaug.json --aug outputs/final_aug.json

Writes to outputs/figures/:
  fig_tradeoff.pdf/.png   correct vs wrong commits across tau, 3 judges x 3 unseen-language conditions
  fig_reliability.pdf/.png reliability diagrams (all combined condition)
  table_main.tex          main results table at one tau, with 95% Wilson intervals
  alt_text.txt            figure descriptions for the accessibility requirement
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

CONDS = [("synonym", "Unseen synonyms"), ("asr", "Unseen ASR slips"), ("all", "All combined")]
JUDGES = [("keyword", "Keyword", "s", "#7f7f7f"), ("noaug", "Laya", "o", "#1f77b4"),
          ("aug", "Laya + aug.", "^", "#d62728")]


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (float("nan"),) * 3
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def load(noaug: str, aug: str):
    a, b = json.load(open(noaug)), json.load(open(aug))
    rows = {}
    for r in a["table"]:
        key = "keyword" if r["judge"] == "keyword" else "noaug"
        rows[(key, r["condition"], round(r["tau"], 3))] = r
    for r in b["table"]:
        if r["judge"] != "keyword":
            rows[("aug", r["condition"], round(r["tau"], 3))] = r
    taus = sorted({k[2] for k in rows})
    return rows, taus, a, b


def counts(r):
    """(correct, wrong, n_clear, false_amb, n_amb) with fallbacks for older result files."""
    n = r.get("n_clear")
    if n is None:
        return None
    correct = r["n_correct"]
    wrong = r["n_committed"] - correct
    return correct, wrong, n, r["n_false_commit_ambiguous"], r["n_ambiguous"]


def fig_tradeoff(rows, taus, out: Path):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.5), sharey=True)
    for ax, (cond, title) in zip(axes, CONDS):
        for key, name, marker, color in JUDGES:
            pts = [rows[(key, cond, t)] for t in taus if (key, cond, t) in rows]
            if not pts:
                continue
            x = [100 * r["wrong_rate"] for r in pts]
            y = [100 * r["correct_rate"] for r in pts]
            kw = key == "keyword"
            ax.plot(x, y, marker=marker, color=color, ms=7 if kw else 4, lw=1.2, label=name,
                    linestyle="--" if kw else "-", zorder=5 if kw else 3)
            if key == "aug":
                for r, xi, yi in zip(pts, x, y):
                    if round(r["tau"], 2) in (0.6, 0.9):
                        ax.annotate(f"τ={r['tau']:.1f}", (xi, yi), textcoords="offset points",
                                    xytext=(4, -9), fontsize=6.5, color=color)
        ax.set_title(title, fontsize=8.5)
        ax.set_ylim(0, 100); ax.set_xlim(left=-0.5)
        ax.set_xlabel("wrong commits (%)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("correct early commits (%)", fontsize=8)
    axes[0].legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig_tradeoff.{ext}", dpi=300)
    plt.close(fig)


def fig_reliability(a, b, out: Path, cond="all"):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(4.6, 2.4), sharey=True)
    srcs = [("Laya", a), ("Laya + aug.", b)]
    for ax, (name, d) in zip(axes, srcs):
        r = next(x for x in d["table"] if x["judge"] != "keyword" and x["condition"] == cond)
        bins = [bb for bb in r["bins"] if bb[2] >= 10]          # skip bins with <10 prefixes
        conf, acc, n = zip(*bins) if bins else ([], [], [])
        ax.plot([0, 1], [0, 1], ":", color="k", lw=0.8)
        ax.plot(conf, acc, "o-", color="#d62728" if "aug" in name else "#1f77b4", ms=3.5, lw=1.2)
        ax.set_title(f"{name}\nECE = {r['ece']:.3f}", fontsize=8.5)
        ax.set_xlabel("confidence", fontsize=8); ax.tick_params(labelsize=7)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal"); ax.grid(alpha=0.3)
    axes[0].set_ylabel("empirical accuracy", fontsize=8)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig_reliability.{ext}", dpi=300)
    plt.close(fig)


def table_main(rows, tau, out: Path):
    conds = [("in-distribution", "In-distribution"), ("disfluency", "Disfluency")] + CONDS + \
            [("asr-confusable", "Confusable ASR$^\\dagger$")]
    lines = [r"\begin{tabular}{llccc}", r"\toprule",
             r"Condition & Judge & Correct (\%) & Wrong (\%) & False commit, ambig. (\%) \\", r"\midrule"]
    fmt = lambda p, lo, hi: f"{100*p:.1f} [{100*lo:.1f}, {100*hi:.1f}]"
    for cond, cname in conds:
        for i, (key, name, _, _) in enumerate(JUDGES):
            r = rows.get((key, cond, tau))
            if r is None:
                continue
            c = counts(r)
            if c:
                cor, wr, n, fa, na = c
                cells = [fmt(*wilson(cor, n)), fmt(*wilson(wr, n)), fmt(*wilson(fa, na))]
            else:
                cells = [f"{100*r['correct_rate']:.1f}", f"{100*r['wrong_rate']:.1f}",
                         f"{100*r['false_commit_on_ambiguous']:.1f}"]
            lines.append(f"{cname if i == 0 else ''} & {name} & " + " & ".join(cells) + r" \\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    (out / "table_main.tex").write_text("\n".join(lines) + "\n")
    return lines


def alt_text(rows, taus, tau, out: Path):
    def r(key, cond):
        return rows.get((key, cond, tau))
    parts = []
    for cond, cname in CONDS:
        k, n, a = r("keyword", cond), r("noaug", cond), r("aug", cond)
        if k and n and a:
            parts.append(f"{cname}: at tau {tau}, keyword matching commits correctly on "
                         f"{100*k['correct_rate']:.0f}% with {100*k['wrong_rate']:.0f}% wrong; Laya "
                         f"{100*n['correct_rate']:.0f}% correct, {100*n['wrong_rate']:.0f}% wrong; Laya with "
                         f"augmentation {100*a['correct_rate']:.0f}% correct, {100*a['wrong_rate']:.0f}% wrong.")
    txt = ("Figure (tradeoff): three line plots side by side, one per unseen-language condition "
           "(synonyms, speech-recognition slips, all combined). Each plots the percentage of "
           "instructions with a correct early commit (vertical axis) against the percentage with a "
           "wrong commit (horizontal axis) as the threshold tau varies, for three judges: keyword "
           "matching (grey squares, dashed; a single point, since it does not depend on tau), Laya "
           "(blue circles) and Laya with augmentation (red triangles). Up and to the left is better. "
           + " ".join(parts) + "\n\n"
           "Figure (reliability): two square plots of confidence (horizontal) against empirical "
           "accuracy (vertical) on the all-combined unseen-language condition, for Laya and Laya with "
           "augmentation, each with a dotted diagonal marking perfect calibration and its expected "
           "calibration error in the title. Points close to the diagonal mean the probabilities can "
           "be read as the chance of being right.\n")
    (out / "alt_text.txt").write_text(txt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--noaug", required=True)
    ap.add_argument("--aug", required=True)
    ap.add_argument("--tau", type=float, default=0.9, help="operating point for the table")
    ap.add_argument("--out-dir", default="outputs/figures")
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    rows, taus, a, b = load(args.noaug, args.aug)
    tau = min(taus, key=lambda t: abs(t - args.tau))
    fig_tradeoff(rows, taus, out)
    fig_reliability(a, b, out)
    lines = table_main(rows, tau, out)
    alt_text(rows, taus, tau, out)
    print("\n".join(lines))
    print(f"\nsaved to {out}/: fig_tradeoff.pdf/.png, fig_reliability.pdf/.png, table_main.tex, alt_text.txt")


if __name__ == "__main__":
    main()
