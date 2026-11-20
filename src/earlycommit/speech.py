"""Real speech in, streaming words out.

Audio comes from text-to-speech (--speak) or from your own recording (--audio).
It is fed to Whisper the way a live system would receive it: every `hop` seconds
the ASR re-decodes the audio heard SO FAR (never the future). A word is released
to the judge only when two consecutive decodes agree on it (LocalAgreement-2, the
policy used by whisper_streaming), so unstable guesses are not acted on.

Each released word carries the wall-clock time at which a live system would have
had it: the hop time at which it became stable + the measured decode latency.

  pip install faster-whisper gTTS          # gTTS needs internet (Colab has it)
"""
from __future__ import annotations

import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SR = 16000


# ------------------------------------------------------------------ audio ---
def load_audio(path: str | Path) -> np.ndarray:
    """Any audio file -> 16 kHz mono float32 (needs ffmpeg)."""
    out = subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-i", str(path),
                          "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(out, dtype=np.float32).copy()


def save_wav(audio: np.ndarray, path: str | Path):
    import wave
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes(pcm.tobytes())


def synthesize(text: str, out_wav: str | Path, engine: str = "gtts") -> Path:
    """Text-to-speech -> 16 kHz wav, with 0.3 s of leading silence."""
    out_wav = Path(out_wav)
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    raw = out_wav.with_suffix(".raw_tts")
    if engine == "gtts":
        from gtts import gTTS
        gTTS(text, lang="en").save(str(raw.with_suffix(".mp3")))
        raw = raw.with_suffix(".mp3")
    elif engine == "espeak":
        exe = shutil.which("espeak-ng") or shutil.which("espeak")
        if not exe:
            raise SystemExit("espeak not found: !apt-get -qq install espeak-ng")
        subprocess.run([exe, "-s", "160", "-w", str(raw.with_suffix(".wav")), text], check=True)
        raw = raw.with_suffix(".wav")
    else:
        raise ValueError(engine)
    audio = load_audio(raw)
    raw.unlink(missing_ok=True)
    audio = np.concatenate([np.zeros(int(0.3 * SR), np.float32), audio])
    save_wav(audio, out_wav)
    return out_wav


# --------------------------------------------------------------- streaming ---
def normalize(text: str) -> list[str]:
    text = text.lower().replace("-", " ")
    text = re.sub(r"[^a-z0-9' ]+", " ", text)
    return [w for w in text.split() if w]


@dataclass
class StreamedWord:
    word: str
    t_avail: float        # when a live system would have this word (s from audio start)


class WhisperStreamer:
    def __init__(self, model: str = "small.en", device: str = "cuda", compute_type: str | None = None):
        from faster_whisper import WhisperModel
        ct = compute_type or ("float16" if device == "cuda" else "int8")
        self.model = WhisperModel(model, device=device, compute_type=ct)
        self.decode_ms: list[float] = []

    def decode(self, audio: np.ndarray) -> list[str]:
        t0 = time.perf_counter()
        segments, _ = self.model.transcribe(audio, language="en", beam_size=1, temperature=0.0,
                                            condition_on_previous_text=False, without_timestamps=True,
                                            vad_filter=False)
        text = " ".join(s.text for s in segments)          # consuming the generator runs decoding
        self.decode_ms.append((time.perf_counter() - t0) * 1000)
        return normalize(text)

    def stream(self, audio: np.ndarray, hop: float = 0.3, min_audio: float = 0.5,
               log: bool = True) -> list[StreamedWord]:
        dur = len(audio) / SR
        committed: list[StreamedWord] = []
        prev: list[str] | None = None
        t = min_audio
        while True:
            last = t >= dur
            t = min(t, dur)
            hyp = self.decode(audio[: int(t * SR)])
            ready = t + self.decode_ms[-1] / 1000
            n = len(committed)
            if last:
                new = hyp[n:]                                 # end of audio: release the rest
            else:
                agree = 0                                     # LocalAgreement-2
                if prev is not None:
                    while agree < min(len(prev), len(hyp)) and prev[agree] == hyp[agree]:
                        agree += 1
                new = hyp[n:agree] if agree > n else []
                # a committed word is never taken back; if hypotheses diverge before
                # the committed prefix we simply wait for them to re-agree
            for w in new:
                committed.append(StreamedWord(w, ready))
            if log and new:
                print(f"  t={t:4.2f}s (+{self.decode_ms[-1]:.0f} ms)  heard: {' '.join(hyp)!r:55s}"
                      f" -> released: {' '.join(w for w in new)}")
            prev = hyp
            if last:
                break
            t += hop
        return committed


class FakeStreamer:
    """Test double: releases the intended words evenly over the audio. No ASR."""
    def __init__(self, words: list[str]):
        self.words, self.decode_ms = words, [0.0]

    def stream(self, audio: np.ndarray, hop: float = 0.3, min_audio: float = 0.5, log: bool = True):
        dur = len(audio) / SR
        step = (dur - 0.3) / max(1, len(self.words))
        return [StreamedWord(w, 0.3 + (i + 1) * step) for i, w in enumerate(self.words)]
