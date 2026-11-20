"""Demo video: EarlyCommit vs. a robot that waits for the full instruction.

Top: two copies of the same tabletop, each with a planar 2-link arm (top-down view).
     Left arm acts on EarlyCommit's decision; right arm waits until the sentence
     ends, then decides with the same judge (the usual pipeline).
Bottom-left: calibrated probability per object after every word, with tau.
Bottom: transcript streaming in at speaking rate.

  python -m earlycommit.demo --judge mock --out outputs/demo.mp4
  USE_TF=0 python -m earlycommit.demo --judge laya --repo ./checkpoints/laya-earlycommit-aug \
      --device cuda --temperature 5.5 --tau 0.9 --condition synonym --out outputs/demo_synonym.mp4
  python -m earlycommit.demo --show-scene --seed 3            # see objects + ids
  ... --seed 3 --target obj_2                                  # generated sentence for obj_2
  ... --seed 3 --text "grab the red cup on the left please"    # your own sentence
"""
from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .calibrate import apply_temperature
from .core import SECONDS_PER_WORD, Episode, make_episodes
from .judge import LayaJudge, MockJudge
from .language import Utterance, consistent, make_utterance
from .scenes import Scene, random_scene

COLOR_RGB = {"red": "#d62728", "blue": "#1f5fd6", "green": "#2ca02c", "yellow": "#f2c200",
             "white": "#f7f7f7", "black": "#222222"}
BASE = np.array([0.0, -0.36])          # robot base, front edge of the table
L1 = L2 = 0.34                          # link lengths (m)
SPEED = 0.30                            # end-effector speed (m/s)
FPS = 20


# ---------------------------------------------------------------- timeline ---
@dataclass
class Timeline:
    words: list[str]
    probs: list[dict[str, float]]       # calibrated, after word i
    judge_ms: float
    commit_idx: int | None              # EarlyCommit
    commit_target: str | None
    base_target: str | None             # baseline decision once the whole instruction is in
    avail: list[float]                  # time word i reaches the judge (end of word, or ASR release)
    shown: list[float]                  # time word i appears in the on-screen transcript
    t_words_end: float                  # end of speech
    audio: str | None = None            # wav to mux into the video (speech mode)

    def word_end(self, i: int) -> float:
        return self.avail[i]

    def t_commit(self) -> float | None:
        return None if self.commit_idx is None else self.avail[self.commit_idx] + self.judge_ms / 1000

    def t_base(self) -> float:
        return self.avail[-1] + self.judge_ms / 1000


def build_timeline(ep: Episode, judge, T: float, tau: float, min_words: int = 2,
                   avail: list[float] | None = None, shown: list[float] | None = None,
                   t_words_end: float | None = None, audio: str | None = None) -> Timeline:
    import time
    n = len(ep.utt.words)
    if avail is None:                   # template mode: fixed speaking rate
        avail = [(i + 1) * SECONDS_PER_WORD for i in range(n)]
        shown = [i * SECONDS_PER_WORD for i in range(n)]
        t_words_end = n * SECONDS_PER_WORD
    probs, lat = [], []
    for i in range(n):
        t0 = time.perf_counter()
        p = judge.predict(ep.scene, ep.utt.words[: i + 1])
        lat.append((time.perf_counter() - t0) * 1000)
        probs.append(apply_temperature(p, T))
    commit_idx = commit_target = None
    for i, p in enumerate(probs):
        if i + 1 < min_words:
            continue
        k = max(p, key=p.get)
        if p[k] >= tau:
            commit_idx, commit_target = i, k
            break
    last = probs[-1]
    k = max(last, key=last.get)
    base_target = k if last[k] >= tau else None
    return Timeline(ep.utt.words, probs, float(np.median(lat[1:] or lat)), commit_idx,
                    commit_target, base_target, list(avail), list(shown), float(t_words_end), audio)


# -------------------------------------------------------------- kinematics ---
# The gripper travels in a straight line (smooth start/stop) and the joints follow by
# IK on ONE fixed branch per reach, chosen before the arm moves. A fixed branch is
# continuous over the workspace, so the elbow never flips sides mid-motion.
HOME = BASE + np.array([0.0, 0.11])    # folded in front of the table edge, clear of all objects


def joints(p: np.ndarray, branch: int) -> np.ndarray:
    d = p - BASE
    r = min(np.linalg.norm(d), L1 + L2 - 1e-4)
    cos_e = (r ** 2 - L1 ** 2 - L2 ** 2) / (2 * L1 * L2)
    e = branch * math.acos(max(-1.0, min(1.0, cos_e)))
    a = math.atan2(d[1], d[0]) - math.atan2(L2 * math.sin(e), L1 + L2 * math.cos(e))
    return np.array([a, e])


def fk(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    elbow = BASE + L1 * np.array([math.cos(q[0]), math.sin(q[0])])
    ee = elbow + L2 * np.array([math.cos(q[0] + q[1]), math.sin(q[0] + q[1])])
    return elbow, ee


def _ee_path(target: np.ndarray, s: float) -> np.ndarray:
    return HOME + s * (target - HOME)


def choose_branch(target: np.ndarray | None) -> int:
    """Pick the IK branch with the least elbow travel over the whole reach (the most
    natural-looking motion), among branches that keep the elbow inside the view."""
    if target is None:
        return 1
    best, best_score = 1, -1e9
    for br in (1, -1):
        elbows = np.array([fk(joints(_ee_path(target, s), br))[0] for s in np.linspace(0, 1, 41)])
        travel = np.linalg.norm(np.diff(elbows, axis=0), axis=1).sum()
        score = -travel - (10.0 if elbows[:, 1].min() < -0.54 else 0.0)
        if score > best_score:
            best, best_score = br, score
    return best


def arm_pose(t: float, t_start: float | None, target: np.ndarray | None, branch: int):
    """(elbow, end-effector, arrived) at time t for a reach HOME -> target starting at t_start."""
    if t_start is None or target is None or t < t_start:
        e, ee = fk(joints(HOME, branch))
        return e, ee, False
    s = min(1.0, (t - t_start) / reach_duration(target))
    s = 0.5 - 0.5 * math.cos(math.pi * s)                    # smooth start and stop
    e, ee = fk(joints(_ee_path(target, s), branch))
    return e, ee, s >= 1.0


def reach_duration(target: np.ndarray) -> float:
    return max(np.linalg.norm(target - HOME) / SPEED, 0.4)


# ---------------------------------------------------------------- drawing ----
def draw_object(ax, o, highlight=False):
    from matplotlib import patches
    fc = COLOR_RGB[o.color]
    ec = "#ff00aa" if highlight else "#555555"
    lw = 3.0 if highlight else 1.0
    x, y = o.x, o.y
    if o.shape == "cup":
        ax.add_patch(patches.Circle((x, y), 0.028, fc=fc, ec=ec, lw=lw))
        ax.add_patch(patches.Circle((x, y), 0.016, fc="#00000033", ec="none"))
        ax.add_patch(patches.Rectangle((x + 0.024, y - 0.008), 0.014, 0.016, fc=fc, ec=ec, lw=lw))
    elif o.shape == "block":
        ax.add_patch(patches.Rectangle((x - 0.026, y - 0.026), 0.052, 0.052, fc=fc, ec=ec, lw=lw))
    elif o.shape == "ball":
        ax.add_patch(patches.Circle((x, y), 0.024, fc=fc, ec=ec, lw=lw))
        ax.add_patch(patches.Circle((x - 0.008, y + 0.008), 0.006, fc="#ffffff88", ec="none"))
    elif o.shape == "bowl":
        ax.add_patch(patches.Ellipse((x, y), 0.07, 0.045, fc=fc, ec=ec, lw=lw))
        ax.add_patch(patches.Ellipse((x, y), 0.045, 0.025, fc="#00000033", ec="none"))
    elif o.shape == "bottle":
        ax.add_patch(patches.Rectangle((x - 0.013, y - 0.035), 0.026, 0.06, fc=fc, ec=ec, lw=lw))
        ax.add_patch(patches.Rectangle((x - 0.006, y + 0.025), 0.012, 0.018, fc=fc, ec=ec, lw=lw))


def render(ep: Episode, tl: Timeline, tau: float, out: Path, title_note: str = ""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation, patches

    objs = {o.id: o for o in ep.scene.objects}
    pos = {o.id: np.array([o.x, o.y]) for o in ep.scene.objects}
    t_c, t_b = tl.t_commit(), tl.t_base()
    ec_target = pos.get(tl.commit_target) if tl.commit_target else None
    bl_target = pos.get(tl.base_target) if tl.base_target else None

    def arrival(t_start, target):
        if t_start is None or target is None:
            return None
        return t_start + reach_duration(target)
    a_ec, a_bl = arrival(t_c, ec_target), arrival(t_b, bl_target)
    t_end = max(x for x in [tl.t_words_end, tl.avail[-1], a_ec, a_bl, t_b] if x is not None) + 1.2
    n_frames = int(t_end * FPS)

    fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
    gs = fig.add_gridspec(3, 2, height_ratios=[3.2, 2.2, 0.8], hspace=0.35, wspace=0.12)
    ax_ec, ax_bl = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    ax_bar = fig.add_subplot(gs[1, :])
    ax_tx = fig.add_subplot(gs[2, :])
    fig.suptitle("EarlyCommit: calibrated word-by-word commitment" + title_note, fontsize=14, weight="bold")

    arm_artists = {}
    for ax, name in [(ax_ec, "EarlyCommit (acts when P ≥ τ)"), (ax_bl, "Baseline (waits for full instruction)")]:
        ax.set_xlim(-0.40, 0.40); ax.set_ylim(-0.56, 0.24); ax.set_aspect("equal"); ax.axis("off")
        ax.add_patch(patches.FancyBboxPatch((-0.32, -0.22), 0.64, 0.44, boxstyle="round,pad=0.01",
                                            fc="#e9dcc4", ec="#9c8660", lw=1.5))
        for o in ep.scene.objects:
            draw_object(ax, o)
        ax.add_patch(patches.Circle(BASE, 0.035, fc="#666666", ec="#333333", zorder=5))
        ax.set_title(name, fontsize=11)
        link, = ax.plot([], [], "-", color="#444444", lw=7, solid_capstyle="round", zorder=6)
        joint, = ax.plot([], [], "o", color="#888888", ms=9, zorder=7)
        grip, = ax.plot([], [], "s", color="#ff00aa", ms=11, zorder=8)
        status = ax.text(0, 0.215, "", ha="center", va="top", fontsize=10.5, weight="bold")
        clock = ax.text(-0.38, -0.55, "", ha="left", va="bottom", fontsize=9, color="#555555")
        arm_artists[ax] = (link, joint, grip, status, clock)

    ids = [o.id for o in ep.scene.objects]
    labels = [f"{objs[i].color} {objs[i].shape} ({objs[i].column} {objs[i].row})" for i in ids]
    bars = ax_bar.barh(range(len(ids)), [0] * len(ids), color=[COLOR_RGB[objs[i].color] for i in ids],
                       edgecolor="#555555")
    ax_bar.set_yticks(range(len(ids)), labels, fontsize=9); ax_bar.invert_yaxis()
    ax_bar.set_xlim(0, 1); ax_bar.axvline(tau, ls="--", color="#ff00aa", lw=1.5)
    ax_bar.text(tau, -0.9, f"τ = {tau}", color="#ff00aa", ha="center", fontsize=9)
    ax_bar.set_xlabel("calibrated probability that this is the object meant (updated after every word)",
                      fontsize=9)
    ax_tx.axis("off"); ax_tx.set_xlim(0, 1); ax_tx.set_ylim(0, 1)
    tx = ax_tx.text(0.5, 0.55, "", ha="center", va="center", fontsize=15, family="monospace")
    tx_note = ax_tx.text(0.5, 0.05, "", ha="center", va="bottom", fontsize=10, color="#ff00aa")

    branches = {ax_ec: choose_branch(ec_target), ax_bl: choose_branch(bl_target)}

    def set_arm(ax, t, t_start, target, decided_label, ask):
        link, joint, grip, status, clock = arm_artists[ax]
        elbow, ee, arrived = arm_pose(t, t_start, target, branches[ax])
        link.set_data([BASE[0], elbow[0], ee[0]], [BASE[1], elbow[1], ee[1]])
        joint.set_data([elbow[0]], [elbow[1]]); grip.set_data([ee[0]], [ee[1]])
        if ask and t_start is not None and t >= t_start:
            status.set_text("Which one do you mean?"); status.set_color("#b35900")
        elif t_start is None or t < t_start:
            status.set_text("listening…"); status.set_color("#777777")
        elif arrived:
            status.set_text(f"grasped: {decided_label}"); status.set_color("#1a7f37")
        else:
            status.set_text(f"moving to: {decided_label}"); status.set_color("#0b5cad")
        clock.set_text(f"t = {t:4.1f} s")

    def label(i):
        return None if i is None else f"{objs[i].color} {objs[i].shape}"

    def update(frame):
        t = frame / FPS
        n_heard = sum(1 for x in tl.shown if x <= t)
        n_judged = sum(1 for x in tl.avail if x + tl.judge_ms / 1000 <= t)
        p = tl.probs[n_judged - 1] if n_judged else {i: 1 / len(ids) for i in ids}
        for b, i in zip(bars, ids):
            b.set_width(p[i])
            committed = tl.commit_target == i and t_c is not None and t >= t_c
            b.set_linewidth(3 if committed else 0.8)
            b.set_edgecolor("#ff00aa" if committed else "#555555")
        ws = tl.words[:n_heard]
        text = " ".join(ws)
        tx.set_text(text if len(text) < 70 else "…" + text[-68:])
        if t_c is not None and t >= t_c:
            tx_note.set_text(f"EarlyCommit committed after “{tl.words[tl.commit_idx]}” "
                             f"(word {tl.commit_idx + 1} of {len(tl.words)})")
        set_arm(ax_ec, t, t_c if t_c is not None else (t_b if tl.base_target is None else None),
                ec_target, label(tl.commit_target), ask=tl.commit_target is None)
        set_arm(ax_bl, t, t_b, bl_target, label(tl.base_target), ask=tl.base_target is None)
        return []

    anim = animation.FuncAnimation(fig, update, frames=n_frames, interval=1000 / FPS, blit=False)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix == ".gif":
        anim.save(out, writer=animation.PillowWriter(fps=FPS))
    else:
        anim.save(out, writer=animation.FFMpegWriter(fps=FPS, bitrate=2400))
        if tl.audio:                                   # put the real voice on the video
            import subprocess
            tmp = out.with_name(out.stem + "_noaudio.mp4")
            out.replace(tmp)
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp), "-i", tl.audio,
                            "-c:v", "copy", "-c:a", "aac", "-b:a", "128k", str(out)], check=True)
            tmp.unlink()
    # A still for the paper: the frame just after EarlyCommit commits.
    t_still = (t_c + 0.4) if t_c is not None else tl.t_words_end
    update(int(t_still * FPS))
    fig.savefig(out.with_suffix(".png"), dpi=200)
    plt.close(fig)
    saved = (a_bl - a_ec) if (a_ec is not None and a_bl is not None) else None
    return saved


# ------------------------------------------------------------ episode pick ---
def custom_scene(seed: int) -> Scene:
    return random_scene(random.Random(seed), f"scene{seed}", n_min=4, n_max=7)


def show_scene(seed: int, out: Path):
    """Print and draw a scene with object ids, so you can choose what to ask for."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import patches
    sc = custom_scene(seed)
    fig, ax = plt.subplots(figsize=(6.4, 4.4), dpi=150)
    ax.set_xlim(-0.36, 0.36); ax.set_ylim(-0.27, 0.25); ax.set_aspect("equal"); ax.axis("off")
    ax.add_patch(patches.FancyBboxPatch((-0.32, -0.22), 0.64, 0.44, boxstyle="round,pad=0.01",
                                        fc="#e9dcc4", ec="#9c8660", lw=1.5))
    for o in sc.objects:
        draw_object(ax, o)
        ax.text(o.x, o.y - 0.045, o.id, ha="center", va="top", fontsize=8, weight="bold")
    ax.text(0, -0.25, "robot side (front)", ha="center", va="top", fontsize=8, color="#666666")
    ax.set_title(f"scene --seed {seed}", fontsize=10)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print(f"scene --seed {seed}:")
    for o in sc.objects:
        print(f"  {o.id}: {o.color} {o.shape} ({o.column} {o.row})")
    print("saved", out)


def pick_episode(args) -> Episode:
    if args.text or args.target:
        scene = custom_scene(args.seed)
        if args.target:
            if args.target not in {o.id for o in scene.objects}:
                raise SystemExit(f"{args.target} is not in scene --seed {args.seed}; run with --show-scene")
            utt = make_utterance(scene, random.Random(args.seed * 7 + 1), p_ambiguous=0.0,
                                 target_id=args.target)
            if args.condition != "in-distribution":
                from .language_ood import perturb
                utt = perturb(utt, [o.id for o in scene.objects], random.Random(11), args.condition)
            return Episode(scene, utt)
        words = args.text.lower().replace(",", " ").split()
        cands = [consistent(scene.objects, words[: i + 1]) for i in range(len(words))]
        target = cands[-1][0] if len(cands[-1]) == 1 else None
        commit = next((i for i, c in enumerate(cands) if len(c) == 1), None)
        return Episode(scene, Utterance(words, target, cands, commit, len(cands[-1]) != 1))
    from .language_ood import perturb
    eps = make_episodes(400, args.seed)
    rng = random.Random(11)
    for ep in eps:
        utt = ep.utt
        if args.condition != "in-distribution":
            utt = perturb(utt, [o.id for o in ep.scene.objects], rng, args.condition)
            if utt.words == ep.utt.words:
                continue                                    # want a visibly perturbed sentence
        if args.ambiguous != utt.ambiguous or len(ep.scene.objects) < 4:
            continue
        if not utt.ambiguous:
            saved = (len(utt.words) - 1 - utt.commit_index) * SECONDS_PER_WORD
            if saved < args.min_saved:
                continue
        return Episode(ep.scene, utt)
    raise SystemExit("no matching episode; try another --seed or lower --min-saved")


def speech_timeline(args, judge):
    """Audio -> streaming ASR -> judge after every released word."""
    from .speech import FakeStreamer, SR, WhisperStreamer, load_audio, normalize, synthesize
    scene = custom_scene(args.seed)
    out = Path(args.out)
    if args.speak:
        wav = synthesize(args.speak, out.with_suffix(".wav"), engine=args.tts)
        intended = args.speak
    else:
        wav = out.with_suffix(".wav")
        from .speech import save_wav
        save_wav(load_audio(args.audio), wav)
        intended = args.intended
    audio = load_audio(wav)
    dur = len(audio) / SR
    print(f"audio    : {wav} ({dur:.2f} s)")
    if args.asr == "fake":
        streamer = FakeStreamer(normalize(intended or ""))
    else:
        dev = args.asr_device or args.device or "cuda"
        streamer = WhisperStreamer(args.asr_model, device=dev)
        streamer.decode(audio[: SR])                     # warm-up, not timed in the stream
        streamer.decode_ms.clear()
    print(f"streaming ASR ({args.asr}, hop {args.hop}s):")
    released = streamer.stream(audio, hop=args.hop)
    if not released:
        raise SystemExit("ASR released no words; check the audio")
    words = [w.word for w in released]
    avail = [w.t_avail for w in released]
    if streamer.decode_ms:
        print(f"ASR decode: median {np.median(streamer.decode_ms):.0f} ms per re-decode")
    # Ground truth only from the intended sentence; the judge only ever sees `words`.
    target = None
    if intended:
        c = consistent(scene.objects, normalize(intended))
        target = c[0] if len(c) == 1 else None
        heard_as = " ".join(words)
        if heard_as != " ".join(normalize(intended)):
            print(f"ASR heard: {heard_as!r}\nintended : {' '.join(normalize(intended))!r}")
    cands = [consistent(scene.objects, words[: i + 1]) for i in range(len(words))]
    utt = Utterance(words, target, cands, None, target is None)
    ep = Episode(scene, utt)
    tl = build_timeline(ep, judge, args.temperature, args.tau, avail=avail, shown=avail,
                        t_words_end=dur, audio=str(wav))
    return ep, tl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", choices=["mock", "laya"], default="mock")
    ap.add_argument("--repo", default="convaiinnovations/laya")
    ap.add_argument("--device", default=None)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--tau", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--condition", default="in-distribution",
                    choices=["in-distribution", "synonym", "asr", "disfluency", "all"])
    ap.add_argument("--ambiguous", action="store_true", help="pick an ambiguous instruction")
    ap.add_argument("--min-saved", type=float, default=0.9, help="min oracle time saved (s)")
    ap.add_argument("--text", default=None, help="your own instruction, in scene --seed")
    ap.add_argument("--target", default=None, help="object id to ask for (e.g. obj_2), in scene --seed")
    ap.add_argument("--show-scene", action="store_true", help="print/draw scene --seed with ids and exit")
    sp = ap.add_argument_group("real speech (scene --seed)")
    sp.add_argument("--speak", default=None, help="sentence to synthesise with TTS and stream through Whisper")
    sp.add_argument("--audio", default=None, help="your own recording (wav/mp3/m4a) to stream through Whisper")
    sp.add_argument("--intended", default=None, help="with --audio: what you meant to say (for checking only)")
    sp.add_argument("--tts", choices=["gtts", "espeak"], default="gtts")
    sp.add_argument("--asr", choices=["whisper", "fake"], default="whisper")
    sp.add_argument("--asr-model", default="small.en")
    sp.add_argument("--hop", type=float, default=0.3, help="ASR re-decode interval (s)")
    sp.add_argument("--asr-device", default=None, help="cuda or cpu for Whisper (default: --device)")
    ap.add_argument("--out", default="outputs/demo.mp4")
    args = ap.parse_args()

    if args.show_scene:
        show_scene(args.seed, Path(args.out).with_name(f"scene_{args.seed}.png"))
        return
    judge = MockJudge() if args.judge == "mock" else LayaJudge(args.repo, None, args.device)
    if args.speak or args.audio:
        ep, tl = speech_timeline(args, judge)
    else:
        ep = pick_episode(args)
        tl = build_timeline(ep, judge, args.temperature, args.tau)
    print("sentence :", " ".join(ep.utt.words))
    print("scene    :", ", ".join(f"{o.id}={o.color} {o.shape} ({o.column} {o.row})" for o in ep.scene.objects))
    print(f"judge    : median {tl.judge_ms:.1f} ms per word")
    for i, (w, p) in enumerate(zip(tl.words, tl.probs)):
        k = max(p, key=p.get)
        print(f"  {i+1:2d} {w:10s} top={k} {p[k]:.2f}" + ("   <- COMMIT" if i == tl.commit_idx else ""))
    print("truth    :", ep.utt.target, "| committed:", tl.commit_target, "| baseline:", tl.base_target)
    note = "" if args.condition == "in-distribution" else f"  [{args.condition}: unseen words]"
    if args.speak or args.audio:
        note = "  [real audio + streaming Whisper]"
    saved = render(ep, tl, args.tau, Path(args.out), note)
    if saved is not None:
        print(f"EarlyCommit finished its reach {saved:.2f} s before the baseline")
    print("saved", args.out, "and", Path(args.out).with_suffix(".png"))


if __name__ == "__main__":
    main()
