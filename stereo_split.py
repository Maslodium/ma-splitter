"""
Split a STEREO stem into two mono parts by stereo position.

Use case: two guitars panned hard left/right (classic J-Rock) land in one
Demucs 'guitar' stem. Demucs can't split guitar-from-guitar, but the pan
information is still in the stereo field — we just collapsed it to mono before.
This recovers it.

Methods:
  lr       : raw left vs right channel (fast, crude)
  midside  : Mid=(L+R)/2 vs Side=(L-R)  (center vs edges)
  pan      : per-bin panorama mask — for each STFT bin decide left/right by the
             L/R magnitude ratio, then iSTFT each side. Best for two panned
             sources; keeps a bin's full spectrum on the side it leans to.

Output: two mono WAVs (a "left/A" and "right/B" part). Naming/role labelling is
decided by the caller, not here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

SR = 44100


def _load_stereo(wav: Path):
    import soundfile as sf

    y, sr = sf.read(str(wav), always_2d=True)
    if y.shape[1] == 1:
        return None, sr  # mono, nothing to split
    return y.astype(np.float32), sr


def split_stereo(wav: Path, out_a: Path, out_b: Path, method: str = "pan") -> dict:
    """Split a stereo stem into two mono files. Returns info dict.

    out_a = left / mid / left-leaning ; out_b = right / side / right-leaning.
    Falls back to duplicating mono if the stem isn't real stereo.
    """
    import soundfile as sf

    y, sr = _load_stereo(Path(wav))
    info = {"method": method, "stereo": y is not None}
    out_a = Path(out_a); out_b = Path(out_b)
    out_a.parent.mkdir(parents=True, exist_ok=True)

    if y is None:
        # mono: can't split — write the same signal to both, flag it
        mono, sr = sf.read(str(wav))
        sf.write(str(out_a), mono, sr)
        sf.write(str(out_b), mono, sr)
        info["note"] = "mono stem; both halves identical"
        return info

    L, R = y[:, 0], y[:, 1]

    if method == "lr":
        a, b = L, R
    elif method == "midside":
        a = (L + R) / 2.0
        b = (L - R) / 2.0
    else:  # pan mask
        a, b = _pan_split(L, R, sr)

    # energy balance for reporting
    info["rms_a"] = float(np.sqrt(np.mean(a ** 2)))
    info["rms_b"] = float(np.sqrt(np.mean(b ** 2)))
    sf.write(str(out_a), a.astype(np.float32), sr)
    sf.write(str(out_b), b.astype(np.float32), sr)
    return info


def _pan_split(L: np.ndarray, R: np.ndarray, sr: int):
    """Per-bin panorama separation.

    For each STFT bin, pan = (|R|-|L|)/(|R|+|L|) in [-1,+1]. We build soft masks
    that send left-leaning bins to part A and right-leaning bins to part B, with
    a smooth crossover around center so shared/center content splits evenly
    rather than chattering between sides.
    """
    import scipy.signal as sps

    nper = 2048
    f, t, ZL = sps.stft(L, sr, nperseg=nper)
    _, _, ZR = sps.stft(R, sr, nperseg=nper)
    magL, magR = np.abs(ZL), np.abs(ZR)
    denom = magL + magR + 1e-9
    pan = (magR - magL) / denom  # -1 hard left .. +1 hard right

    # Soft masks: left part gets more of negative-pan bins, right part positive.
    # tanh gives a smooth crossover; center content (pan~0) splits ~50/50.
    k = 4.0
    mask_b = 0.5 * (1.0 + np.tanh(k * pan))   # ->1 on the right
    mask_a = 1.0 - mask_b                       # ->1 on the left

    # Apply to the SUM spectrum (L+R) so each side keeps full timbre.
    Zsum = ZL + ZR
    _, a = sps.istft(Zsum * mask_a, sr, nperseg=nper)
    _, b = sps.istft(Zsum * mask_b, sr, nperseg=nper)
    return a, b


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="split a stereo stem by pan position")
    ap.add_argument("wav", type=Path)
    ap.add_argument("out_a", type=Path)
    ap.add_argument("out_b", type=Path)
    ap.add_argument("--method", default="pan", choices=["pan", "lr", "midside"])
    a = ap.parse_args()
    res = split_stereo(a.wav, a.out_a, a.out_b, a.method)
    print(res)
