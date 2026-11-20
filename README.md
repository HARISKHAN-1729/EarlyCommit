
# In progress

# EarlyCommit

**Calibrated word-by-word commitment for robots acting before you finish speaking.**

EarlyCommit lets a robot pick the object a person means *while the instruction is still being spoken*. After every recognised word, a fine-tuned text classifier (Laya) scores every object in the scene. The probabilities are temperature-calibrated, and the robot **commits** to the best object as soon as its probability reaches a threshold τ. Otherwise it **waits** for the next word, or **asks** which object is meant if the sentence ends without a commit.

This repository contains the code, the result files behind every number in the paper, the two fine-tuned checkpoints, and one Colab notebook (`demo_finetuning_evaluation.ipynb`) that fine-tunes, evaluates and demos the tool.

---

## Contents

1. [Repository layout](#1-repository-layout)
2. [Quick start in Google Colab](#2-quick-start-in-google-colab)
3. [Checkpoints: get, unzip, connect](#3-checkpoints-get-unzip-connect)
4. [Reproduce the paper without a GPU](#4-reproduce-the-paper-without-a-gpu)
5. [Full evaluation on a GPU](#5-full-evaluation-on-a-gpu)
6. [Training the checkpoints](#6-training-the-checkpoints)
7. [Demo videos](#7-demo-videos)
8. [Running outside Colab](#8-running-outside-colab)
9. [Command reference](#9-command-reference)
10. [Where each paper number comes from](#10-where-each-paper-number-comes-from)
11. [Troubleshooting](#11-troubleshooting)
12. [Licence, maintenance and responsible use](#12-licence-maintenance-and-responsible-use)

---

## 1. Repository layout

```
EarlyCommit/
├── README.md                this file
├── LICENSE                  Apache 2.0
├── requirements.txt
├── demo_finetuning_evaluation.ipynb   one notebook with three sections:
│                                        Fine-tuning Laya  (GPU)
│                                        System Evaluation (parts A–B on CPU, C–F on GPU)
│                                        Demo              (parts 1–2 on CPU, 3–5 on GPU)
├── results/                 the three result files used in the paper
│   ├── final_noaug.json         Laya, temperature fitted on the calibration pool (+ keyword baseline)
│   ├── final_aug.json           Laya + augmentation, temperature fitted on the calibration pool
│   └── final_aug_trainpool.json Laya + augmentation, temperature fitted on the training pool (Table 2)
├── checkpoints/             the two fine-tuned checkpoints (see section 3)
└── src/earlycommit/         the Python package
    ├── scenes.py            random tabletop scenes (3–7 objects, 3×2 grid)
    ├── language.py          instruction generator + incremental oracle (exact labels after every word)
    ├── language_ood.py      synonyms, recognition slips, fillers; disjoint train / calibration / test pools
    ├── core.py              episodes, commit rule, fine-tuning data writer
    ├── judge.py             Laya judge, keyword baseline, mock judge, prediction cache
    ├── calibrate.py         temperature scaling, temperature fitting, calibration error
    ├── train.py             fine-tuning (Laya's RLCD recipe)
    ├── run.py               data generation and evaluation (`data`, `eval`, `ood`)
    ├── paper.py             regenerates every table, figure and headline number, with checks
    ├── figures.py           trade-off figure helpers
    ├── bench.py             latency benchmark
    ├── checkpoints.py       unpacks and checks checkpoint zips
    ├── speech.py            text-to-speech, streaming Whisper with LocalAgreement-2
    └── demo.py              demo videos (kinematic arm replay, not physics)
```

The notebook never copies code: every cell calls a module in `src/earlycommit/`, as `python -m src.earlycommit.<module>` (run from the `EarlyCommit/` folder) or `python -m earlycommit.<module>` (with `PYTHONPATH=src`, which Setup cell 1 sets). Both forms run the same code.

---

## 2. Quick start in Google Colab

1. **Open Colab** at colab.research.google.com and upload the notebook `demo_finetuning_evaluation.ipynb` (File → Upload notebook).
2. **Choose a GPU:** Runtime → Change runtime type → GPU. An **A100** is best for fine-tuning; a **T4** works for evaluation and the demo but is slower. Evaluation parts A–B and Demo parts 1–2 also run without a GPU.
3. **Upload `EarlyCommit.zip`** (this whole folder, zipped) to `/content` with the file panel on the left (folder icon → upload). Wait until the upload has finished.
4. **Run the two setup cells at the top of the notebook.** *Setup cell 1* unzips the project to `/content/EarlyCommit`, moves there and sets the paths. *Setup cell 2* installs Laya.
5. **Jump to any section**, in any order:

| Goal | Notebook section | GPU | Checkpoints |
|---|---|---|---|
| Check every paper number quickly | **System Evaluation**, parts A–B | no | no |
| Re-run the full evaluation | **System Evaluation**, parts C–F | yes | yes |
| Train the two checkpoints yourself | **EarlyCommit: fine-tuning Laya** | yes (A100) | produces them |
| Watch the robot commit mid-sentence | **Demo**, parts 3–5 (parts 1–2 need no GPU) | yes | Laya + aug. |

The Demo section's speech parts (4–5) also need `faster-whisper` and `gTTS`. If they are not installed yet, run `!pip install -q faster-whisper gTTS` once before the Demo section.

---

## 3. Checkpoints: get, unzip, connect

The evaluation and the demo use two fine-tuned Laya checkpoints (about 400 MB each):

| File | Model | Used by |
|---|---|---|
| `laya-earlycommit.zip` | Laya fine-tuned on template instructions | evaluation (run 1), latency benchmark |
| `laya-earlycommit-aug.zip` | Laya fine-tuned with half the instructions rewritten with train-pool words | evaluation (runs 2 and 3), demo |

**Getting them.** The two checkpoints are included in `EarlyCommit/checkpoints/`, and the System Evaluation and Demo sections use whatever checkpoints are in that folder. If you fine-tune the models yourself (section 6), the new checkpoints are written to the same folder and are used from then on. If you instead have the checkpoints as zip files, follow the steps below.

**Step 1: upload.** Large files are safest through Google Drive:
1. Upload both zips to your Google Drive (drive.google.com) and wait until Drive shows them as fully uploaded.
2. In the notebook, mount Drive and copy them into `checkpoints/`:
   ```python
   from google.colab import drive; drive.mount('/content/drive')
   !cp /content/drive/MyDrive/laya-earlycommit.zip /content/drive/MyDrive/laya-earlycommit-aug.zip checkpoints/
   ```
   (The notebook contains this cell in System Evaluation part C, commented out. Remove the `#` to use it.)

Uploading through the Colab file panel also works, but wait for the progress indicator to disappear. An interrupted upload gives a damaged zip (see section 11).

**Step 2: unzip and check.** System Evaluation part C does this, and skips any checkpoint that is already in place. By hand:
```bash
python -m earlycommit.checkpoints --zip checkpoints/laya-earlycommit.zip     --name laya-earlycommit
python -m earlycommit.checkpoints --zip checkpoints/laya-earlycommit-aug.zip --name laya-earlycommit-aug
```
This tests the zip, finds the model folder at any depth inside it, moves it to `checkpoints/<name>/` and checks the files. Expected output:
```
ready: checkpoints/laya-earlycommit -> ['encoder', 'model.safetensors', 'rl_agent_config.json', 'tokenizer']
ready: checkpoints/laya-earlycommit-aug -> ['encoder', 'model.safetensors', 'rl_agent_config.json', 'tokenizer']
```

**Step 3: connect.** Every command takes the folder with `--repo`, for example `--repo checkpoints/laya-earlycommit-aug`. `judge.LayaJudge` loads a local folder with `laya.Agent(folder)`. Nothing else needs configuring.

**Quick load test (GPU):**
```bash
python -m earlycommit.bench --device cuda --repo checkpoints/laya-earlycommit --n 20
```
It should print `model on: cuda:0` and a latency line.

**Expected latency per word** (fine-tuned judge, `--n 100`): 31.6 ms median and 38.3 ms 95th percentile on a laptop RTX 3060 (the paper's numbers), about 150 ms and 180 ms on a free Colab T4. Both are below the ~300 ms it takes to speak a word. Close other GPU jobs before benchmarking.

---

## 4. Reproduce the paper without a GPU

```bash
python -m earlycommit.paper --results results --out paper_outputs
```
Takes seconds on a CPU. It reads the three result files in `results/` and writes:

| File | Paper item |
|---|---|
| `paper_outputs/table1.tex` | Table 1 |
| `paper_outputs/table2.tex` | Table 2 |
| `paper_outputs/fig_tradeoff.pdf` | Figure 3 |
| `paper_outputs/fig_reliability.pdf` | Figure 4 |
| `paper_outputs/evaluate_barchart.pdf` | bar chart inside Figure 2 |

It also prints every headline number of the paper and checks it. Expected last line: `11/11 checks pass`.

**Smoke test of the whole pipeline** (CPU, about a minute, no checkpoint):
```bash
python -m earlycommit.run ood --judge mock --n-val 60 --n-test 100 --out mock_smoke_test.json
```
The mock judge is the oracle plus noise; its numbers are not paper numbers.

---

## 5. Full evaluation on a GPU

This re-creates the three result files. In the notebook it is **System Evaluation**, part E (part F then rebuilds the tables and figures); the commands are:

```bash
export PYTHONPATH=src USE_TF=0
TAUS=0.5,0.6,0.7,0.8,0.85,0.9,0.95
COMMON="--judge laya --device cuda --n-val 500 --n-test 1500 --taus $TAUS"
mkdir -p logs
python -m earlycommit.run ood $COMMON --repo checkpoints/laya-earlycommit     --val-pool calib --out final_noaug.json         > logs/final_noaug.txt 2>&1 &
python -m earlycommit.run ood $COMMON --repo checkpoints/laya-earlycommit-aug --val-pool calib --out final_aug.json           > logs/final_aug.txt 2>&1 &
python -m earlycommit.run ood $COMMON --repo checkpoints/laya-earlycommit-aug --val-pool train --out final_aug_trainpool.json > logs/final_aug_trainpool.txt 2>&1 &
wait
grep -H -E "temperatures|saved" logs/*.txt
python -m earlycommit.paper --results outputs --out paper_outputs_new
```

What one `ood` run does:
1. **Fit the temperature T** on 500 validation instructions, half of them rewritten with the word pool given by `--val-pool` (`calib` = held-out calibration pool, used in the paper; `train` = the training pool, the Table 2 ablation). Minimises the soft-label negative log-likelihood.
2. **Evaluate 1,500 test instructions** (1,250 with a unique target, 250 ambiguous) under six conditions: clean, unseen disfluencies, unseen synonyms, unseen recognition slips, all combined, and confusable slips. The unseen conditions use test-pool words only.
3. **Compare** the keyword baseline and Laya at every τ in `--taus`.
4. **Write** `outputs/<out>.json`, with rates, raw counts, calibration bins and temperatures.

The evaluation feeds the generated words straight to the judge, without speech recognition. Speech is only used in the demo.

**Time.** Each run makes about 53,000 judge calls (a cache removes repeats). With all three in parallel: about 20–30 minutes on an A100, longer on a T4. GPU memory is not an issue; each process uses under 2 GB.

**Expected temperatures:** T ≈ 5.29 for both calibration-pool runs, T ≈ 0.61 for the training-pool run.

---

## 6. Training the checkpoints

In the notebook this is the section **EarlyCommit: fine-tuning Laya** (parts 1–3), or:
```bash
export PYTHONPATH=src USE_TF=0
python -m earlycommit.train --out checkpoints/laya-earlycommit
python -m earlycommit.train --out checkpoints/laya-earlycommit-aug --augment 0.5
```

Each command:
1. **Generates the data.** 4,000 training and 500 validation instructions (seeds 1 and 2; the test set uses seed 3). Every prefix of two or more words becomes one example with a soft label: uniform over the objects the oracle still considers possible. That gives 30,972 examples, or 33,007 with `--augment 0.5`, which rewrites half of the training and validation instructions with **train-pool** words only. Written to `<out>_data/`.
2. **Tokenises** with Laya's own sequence builder, keeping the checkpoint's 512-token budget (no example is truncated; the longest has 348 tokens).
3. **Fine-tunes the whole model** with Laya's RLCD recipe: the logits are perturbed 4 times with Gaussian noise (σ from 0.4 to 0.1), each perturbed distribution is rewarded with a proper scoring rule (log + spherical), and the policy-gradient term is added to soft cross-entropy. 2 epochs; AdamW with learning rate 2.5e-5 for the encoder and 1e-4 for the head; weight decay 0.01; cosine schedule; batch 64 (32 × 2 accumulation); gradient clipping at 1; bfloat16 (float16 with a gradient scaler on GPUs without bfloat16).
4. **Fits Laya's internal temperature** on held-out validation prefixes and stores it in the checkpoint.
5. **Saves** `<out>/` with `model.safetensors`, `encoder/`, `tokenizer/` and `rl_agent_config.json`.

It prints the validation accuracy on determined prefixes before training (the paper reports 44% for the base model) and after each epoch.

**On a T4,** add `--micro-batch 16 --grad-accum 4` if memory runs out; the effective batch stays 64. GPU training is not bit-for-bit deterministic, so retrained checkpoints can differ slightly from the paper's.

---

## 7. Demo videos

In the notebook this is the **Demo** section (parts 1–5), or run the commands below. Each video shows two kinematic two-link arms side by side: EarlyCommit (commits word by word) and the baseline (same judge and τ, decides after the last word). Bars show the calibrated probability of each object after every word. **This is a visualisation of the judge's decisions, not a physics simulation.**

```bash
export PYTHONPATH=src USE_TF=0
R=checkpoints/laya-earlycommit-aug

# look at a scene and its object ids
python -m earlycommit.demo --show-scene --seed 3

# no GPU, no checkpoint: mock judge
python -m earlycommit.demo --judge mock --temperature 1.3 --seed 3 --target obj_1 --out outputs/mock.mp4

# fine-tuned judge, generated instruction for one object
python -m earlycommit.demo --judge laya --repo $R --device cuda --temperature 5.29 --tau 0.9 --seed 3 --target obj_1 --out outputs/grab_obj_1.mp4

# real speech: text-to-speech streamed through Whisper (Figure 1)
python -m earlycommit.demo --judge laya --repo $R --device cuda --temperature 5.29 --tau 0.9 --seed 3 \
    --speak "grab the red cup in the middle please" --out outputs/speech_fig1.mp4

# your own recording (wav, mp3, m4a)
python -m earlycommit.demo --judge laya --repo $R --device cuda --temperature 5.29 --tau 0.9 --seed 3 \
    --audio my_voice.m4a --intended "can you get the white cup on the left" --out outputs/speech_me.mp4
```

Each run prints every word with the top object and its probability, marks the commit, and prints `truth | committed | baseline`. It saves the video and a still frame (`.png`). With speech, it also prints what Whisper heard every 0.3 s and which words it released. A word is released once two consecutive transcripts agree on it.

---

## 8. Running outside Colab

Tested on Ubuntu with Python 3.10 (laptop RTX 3060) and on Google Colab (A100).
```bash
cd EarlyCommit
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # ffmpeg separately: sudo apt install ffmpeg
export PYTHONPATH=src USE_TF=0
python -m earlycommit.paper --results results --out paper_outputs   # check: 11/11 checks pass
```
Then use any command in sections 3–7. The notebook also runs in local Jupyter if started from the `EarlyCommit/` folder.

---

## 9. Command reference

All commands are run from `EarlyCommit/` with `PYTHONPATH=src` (and `USE_TF=0` for anything that loads Laya).

| Command | What it does | Main options |
|---|---|---|
| `python -m earlycommit.paper` | tables, figures, headline numbers and checks from result files | `--results DIR --out DIR` |
| `python -m earlycommit.run data` | writes train/val/test prefix data (JSONL) to `outputs/<out-dir>/` | `--augment 0.5 --out-dir data` |
| `python -m earlycommit.run ood` | full robustness evaluation | `--judge laya/mock --repo DIR --device cuda/cpu --n-val --n-test --taus --val-pool calib/train --val-augment 0.5 --out FILE` |
| `python -m earlycommit.run eval` | in-distribution calibration and commit curves only | `--judge --repo --device --n-val --n-test` |
| `python -m earlycommit.train` | fine-tunes Laya | `--out DIR --augment 0.5 --epochs 2 --micro-batch 32 --grad-accum 2` |
| `python -m earlycommit.checkpoints` | unpacks and checks a checkpoint zip | `--zip FILE --name NAME` |
| `python -m earlycommit.bench` | judge latency per word | `--repo DIR --device cuda/cpu --n 100` |
| `python -m earlycommit.demo` | demo video | `--show-scene`, `--seed`, `--target obj_k`, `--text "..."`, `--speak "..."`, `--audio FILE --intended "..."`, `--judge`, `--repo`, `--temperature`, `--tau`, `--asr-device`, `--tts gtts/espeak`, `--out` |

All randomness is seeded (scenes and instructions: seed per split; augmentation: seed + 1000; evaluation perturbations: 11; benchmark prefixes: 7), so data is identical across machines.

---

## 10. Where each paper number comes from

| Paper item | Source |
|---|---|
| Table 1, Figure 3, bar chart in Figure 2 | `results/final_noaug.json` (keyword, Laya) and `results/final_aug.json` (Laya + aug.), τ = 0.9 |
| Table 2 | `results/final_aug_trainpool.json` vs `results/final_aug.json`, condition "all combined" |
| Figure 4 | calibration bins in all three result files, condition "all combined" |
| T = 5.29 and T = 0.61 | the `temperatures` field of the result files |
| Confidence intervals, 1.6–2× ratio, lower bounds, time saved | computed by `python -m earlycommit.paper` from raw counts |
| Latency 31.6 / 38.3 ms | `python -m earlycommit.bench` on a laptop RTX 3060 |
| 44% before fine-tuning | printed by `python -m earlycommit.train` before training |

---

## 11. Troubleshooting

| Problem | Fix |
|---|---|
| `End-of-central-directory signature not found` when unzipping a checkpoint | The upload was incomplete. Upload again and wait until it finishes, or use Google Drive (section 3). `python -m earlycommit.checkpoints` tests zips before unpacking. |
| `No module named earlycommit` | Run from the `EarlyCommit/` folder and set `PYTHONPATH=src` (Setup cell 1 does this). |
| Laya loading hangs | Set `USE_TF=0` (Setup cell 1 does). |
| `numpy.ufunc ... no __dict__` or transformers import errors in Colab | NumPy was upgraded inside a running session. Runtime → Restart session, then continue. Don't `pip install -U numpy`. |
| CUDA out of memory in training | `--micro-batch 16 --grad-accum 4` |
| Whisper error mentioning `libcudnn` or `ctranslate2` | Add `--asr-device cpu` to the demo command (Whisper on CPU, Laya on GPU). |
| gTTS fails (no internet) | `!apt-get -qq install espeak-ng` and add `--tts espeak`. |
| A parallel evaluation run is missing its `saved` line | `tail -30 logs/<run>.txt` shows the error. |
| Demo fails with `No module named faster_whisper` or `gtts` | Run `!pip install -q faster-whisper gTTS` once, then the Demo cell again. |
| Setup cell 1 says it cannot find the zip | The archive must be named exactly `EarlyCommit.zip` and uploaded to `/content`. Names are case-sensitive in Colab. |

---

## 12. Licence, maintenance and responsible use

**Licence:** Apache 2.0 (see `LICENSE`), matching Laya.

**Maintenance:** upon acceptance, the repository will be public on GitHub with versioned Zenodo releases, and the checkpoints will be on Hugging Face. Issues and pull requests are welcome.

**Responsible use:**
- EarlyCommit lets a robot move while a person is still speaking.
- τ sets the accepted risk (1 − τ) of choosing the wrong object. It is a calibrated heuristic, not a safety certification, and not a finite-sample guarantee.
- Keep the robot's normal physical safety layers.
- Refit the temperature when moving to new speakers or vocabularies, on language variation that was not used in training.

**Scope:** the evaluation uses templated tabletop instructions with simulated recognition slips; there is no user study and no physical robot.

