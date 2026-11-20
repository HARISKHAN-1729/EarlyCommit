"""Regenerate every table, figure and headline number in the paper from result files.

  python -m earlycommit.paper --results results --out paper_outputs

Needs the three result files produced by `run ood` (they are included in results/):
  final_noaug.json        Laya without augmentation, T fitted on the calibration pool (+ keyword baseline)
  final_aug.json          Laya with augmentation,    T fitted on the calibration pool
  final_aug_trainpool.json Laya with augmentation,   T fitted on the training pool

Writes table1.tex, table2.tex, fig_tradeoff.pdf, fig_reliability.pdf and
evaluate_barchart.pdf, prints every number quoted in the paper, and compares
them with the values in the paper (PASS / FAIL). No GPU needed.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

TAU = 0.9
UNSEEN = ["synonym", "asr", "all"]
CONDS = [("in-distribution", "Clean (in-distribution)"), ("disfluency", "Unseen disfluencies"),
         ("synonym", "Unseen synonyms"), ("asr", "Unseen ASR slips"), ("all", "All combined"),
         ("asr-confusable", "Confusable ASR slips")]


def load(path):
    d = json.load(open(path))
    return d, {(r["judge"], r["condition"], round(r["tau"], 2)): r for r in d["table"]}


def rates(r):
    n = r["n_clear"]
    return 100 * r["n_correct"] / n, 100 * (r["n_committed"] - r["n_correct"]) / n


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="paper_outputs")
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    res, out = Path(args.results), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    N, n = load(res / "final_noaug.json")
    A, a = load(res / "final_aug.json")
    P, p = load(res / "final_aug_trainpool.json")
    judges = [("Keyword", n, "keyword"), ("Laya", n, "laya"), ("Laya + aug.", a, "laya")]

    # ---------------- Table 1
    lines = [r"\begin{tabular}{@{}lccc@{}}", r"\toprule",
             r"Condition & Keyword & Laya & Laya + aug. \\", r"\midrule"]
    fmt = lambda x: f"{x:.0f}" if abs(x - round(x)) < 1e-9 and x in (0, 100) else f"{x:.1f}"
    for c, label in CONDS:
        cells = [" / ".join(fmt(round(v, 1)) for v in rates(t[(j, c, TAU)])) for _, t, j in judges]
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "table1.tex").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    # ---------------- Table 2
    tr, ca = p[("laya", "all", TAU)], a[("laya", "all", TAU)]
    t2 = [("Fitted temperature $T$", f"{P['temperatures']['laya']:.2f}", f"{A['temperatures']['laya']:.2f}"),
          ("Calibration error, clean", f"{p[('laya','in-distribution',TAU)]['ece']:.3f}", f"{a[('laya','in-distribution',TAU)]['ece']:.3f}"),
          ("Calibration error, unseen", f"{tr['ece']:.3f}", f"{ca['ece']:.3f}"),
          ("Correct commits", f"{rates(tr)[0]:.1f}\\%", f"{rates(ca)[0]:.1f}\\%"),
          ("Wrong commits", f"{rates(tr)[1]:.1f}\\%", f"{rates(ca)[1]:.1f}\\%"),
          ("Accuracy at commit", f"{tr['accuracy_at_commit']:.2f}", f"{ca['accuracy_at_commit']:.2f}"),
          ("False commits, ambiguous", f"{100*tr['false_commit_on_ambiguous']:.1f}\\%", f"{100*ca['false_commit_on_ambiguous']:.1f}\\%")]
    lines = [r"\begin{tabular}{@{}lcc@{}}", r"\toprule",
             r"Temperature fitted on & Training pool & Calibration pool \\", r"\midrule"]
    lines += [f"{l} & {x} & {y} \\\\" for l, x, y in t2] + [r"\bottomrule", r"\end{tabular}"]
    (out / "table2.tex").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    # ---------------- headline numbers and checks against the paper
    checks = []
    def check(name, got, want, tol):
        ok = abs(got - want) <= tol
        checks.append(ok)
        print(f"{'PASS' if ok else 'FAIL'}  {name}: {got:.3f} (paper {want})")
    ratios = [rates(a[("laya", c, TAU)])[0] / rates(n[("keyword", c, TAU)])[0] for c in UNSEEN]
    check("min ratio aug/keyword (paper 1.6)", min(ratios), 1.6, 0.05)
    check("max ratio aug/keyword (paper 2)", max(ratios), 2.0, 0.06)
    acc = [a[("laya", c, TAU)]["accuracy_at_commit"] for c in UNSEEN]
    check("min accuracy at commit, aug (paper 0.92)", min(acc), 0.92, 0.005)
    lows = [wilson(a[("laya", c, TAU)]["n_correct"], a[("laya", c, TAU)]["n_committed"])[0] for c in UNSEEN]
    check("lowest lower 95% bound of accuracy (paper 90.2)", min(lows), 90.2, 0.05)
    check("highest lower 95% bound of accuracy (paper 93.6)", max(lows), 93.6, 0.05)
    tp = p[("laya", "all", TAU)]
    check("train-pool upper 95% bound (paper 83.3)", wilson(tp["n_correct"], tp["n_committed"])[1], 83.3, 0.05)
    check("T calibration pool (paper 5.29)", A["temperatures"]["laya"], 5.29, 0.005)
    check("T training pool (paper 0.61)", P["temperatures"]["laya"], 0.61, 0.005)
    check("time saved, clean (paper 0.79 s)", a[("laya", "in-distribution", TAU)]["mean_time_saved_s"], 0.79, 0.005)
    hw = max((lambda lo_hi: (lo_hi[1] - lo_hi[0]) / 2)(wilson(t[(j, c, TAU)]["n_correct"], t[(j, c, TAU)]["n_clear"]))
             for c, _ in CONDS for _, t, j in judges)
    check("largest 95% half-width on correct commits (paper 2.8)", hw, 2.8, 0.05)
    same = all((a[("laya", c, 0.85)]["n_committed"], a[("laya", c, 0.85)]["n_correct"]) ==
               (a[("laya", c, 0.95)]["n_committed"], a[("laya", c, 0.95)]["n_correct"]) for c, _ in CONDS)
    checks.append(same)
    print(f"{'PASS' if same else 'FAIL'}  tau 0.85 and 0.95 give identical decisions")
    print(f"\n{sum(checks)}/{len(checks)} checks pass")

    # ---------------- Figure 3: tradeoff
    from .figures import fig_tradeoff, load as fload
    rows, taus, _, _ = fload(str(res / "final_noaug.json"), str(res / "final_aug.json"))
    fig_tradeoff(rows, taus, out)

    # ---------------- Figure 4: reliability (three calibrations)
    plt.rcParams.update({"font.family": "serif", "font.size": 7.5})
    fig, ax = plt.subplots(figsize=(3.3, 1.75))
    ax.plot([0, 1], [0, 1], ":", color="k", lw=0.8, label="perfect calibration")
    for d, lab, st, col in [(N, "Laya, calibration pool $T$", "o-", "#1f77b4"),
                            (A, "Laya + aug., calibration pool $T$", "^-", "#d62728"),
                            (P, "Laya + aug., training pool $T$", "s--", "#7f7f7f")]:
        r = next(x for x in d["table"] if x["judge"] == "laya" and x["condition"] == "all")
        b = [x for x in r["bins"] if x[2] >= 10]
        ax.plot([x[0] for x in b], [x[1] for x in b], st, color=col, ms=3.2, lw=1.1, label=f"{lab}, ECE {r['ece']:.3f}")
    ax.set_xlim(0, 1.01); ax.set_ylim(0, 1.01); ax.grid(alpha=0.3, lw=0.5)
    ax.set_xlabel("calibrated top probability"); ax.set_ylabel("empirical accuracy")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=2, fontsize=6.0, frameon=False)
    fig.savefig(out / "fig_reliability.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out / "fig_reliability.png", dpi=200, bbox_inches="tight", pad_inches=0.02); plt.close(fig)

    # ---------------- bar chart used inside Figure 2
    vals = [rates(t[(j, "all", TAU)])[0] for _, t, j in judges]
    fig, ax = plt.subplots(figsize=(2.4, 1.7))
    cols = ["#6B7078", "#3E7CA8", "#163A5F"]
    ax.bar(range(3), vals, color=cols, width=0.62)
    for i, v in enumerate(vals):
        ax.text(i, v + 2.5, f"{v:.0f}%", ha="center", fontsize=11, fontweight="bold", color=cols[i])
    ax.set_xticks(range(3)); ax.set_xticklabels(["Keyword", "Laya", "Laya\n+ aug."])
    ax.set_ylim(0, 78); ax.set_yticks([])
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.set_title("correct early commits\nunseen words, τ = 0.9", fontsize=8.5)
    fig.tight_layout(); fig.savefig(out / "evaluate_barchart.pdf"); fig.savefig(out / "evaluate_barchart.png", dpi=200); plt.close(fig)
    print("\nwrote", sorted(x.name for x in out.iterdir()))


if __name__ == "__main__":
    main()
