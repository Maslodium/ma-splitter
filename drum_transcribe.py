"""
Drum transcription via band-wise onset detection (not pitch transcription).

Why not basic-pitch: cymbals/snares are broadband noise with no pitch, so a
pitch tracker can't see them. Onset detection looks for sharp energy bursts
instead, which is exactly what a drum hit is — broadband-ness HELPS here.

Approach:
  1. Split the drum stem into frequency bands:
       kick   : 20-150 Hz   (low thump)
       snare  : 150-2500 Hz  (body + crack)
       hat/cym: 6000+ Hz     (bright noise, incl. open cymbals ~ white noise)
  2. Detect onsets independently per band (so simultaneous hits register).
  3. Classify each onset by which band dominates AND spectral flatness
     (flatness high = noisy = cymbal; lower + low-freq body = snare/kick).
  4. Optionally snap to the beat grid (cymbals "every quarter" become robust).
  5. Emit a GM-percussion MIDI (channel 10):
       36 kick, 38 snare, 42 closed hat, 46 open hat, 49 crash.

This is offline and per-stem, so it slots into the existing pipeline.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# General MIDI percussion note numbers.
GM = {"kick": 36, "snare": 38, "rimshot": 37, "hat": 42, "open_hat": 46,
      "crash": 49, "tom_low": 45, "tom_mid": 47, "tom_high": 50}

SR = 44100
HOP = 512


def _band(y: np.ndarray, sr: int, lo: float, hi: float | None) -> np.ndarray:
    """Zero-phase-ish bandpass via FFT masking (cheap, good enough for onsets)."""
    import scipy.signal as sps

    ny = sr / 2
    if hi is None or hi >= ny:
        sos = sps.butter(4, lo / ny, btype="highpass", output="sos")
    elif lo <= 0:
        sos = sps.butter(4, hi / ny, btype="lowpass", output="sos")
    else:
        sos = sps.butter(4, [lo / ny, hi / ny], btype="bandpass", output="sos")
    return sps.sosfiltfilt(sos, y)


def _onsets(y_band: np.ndarray, sr: int, delta: float) -> np.ndarray:
    """Energy/flux onset times (sec) for one band — good for kick/snare."""
    import librosa

    env = librosa.onset.onset_strength(y=y_band, sr=sr, hop_length=HOP)
    frames = librosa.onset.onset_detect(
        onset_envelope=env, sr=sr, hop_length=HOP,
        backtrack=False, delta=delta, wait=int(0.03 * sr / HOP),
    )
    return librosa.frames_to_time(frames, sr=sr, hop_length=HOP)


def _superflux_onsets(y: np.ndarray, sr: int, fmin: float, delta: float) -> np.ndarray:
    """Superflux onsets (Boeck) on a mel band from fmin up. Designed for
    sustained/ringing sounds (open cymbals) where compression removes the sharp
    amplitude transient but the spectral shape still shifts on each hit."""
    import librosa

    mel = librosa.feature.melspectrogram(y=y, sr=sr, hop_length=HOP, n_mels=128,
                                         fmin=fmin, fmax=sr / 2)
    env = librosa.onset.onset_strength(S=librosa.power_to_db(mel, ref=np.max),
                                       sr=sr, hop_length=HOP, lag=2, max_size=5)
    frames = librosa.onset.onset_detect(
        onset_envelope=env, sr=sr, hop_length=HOP,
        backtrack=False, delta=delta, wait=int(0.03 * sr / HOP),
    )
    return librosa.frames_to_time(frames, sr=sr, hop_length=HOP)


def _estimate_beats(y: np.ndarray, sr: int, manual_bpm: float | None,
                    offset_ms: float = 0.0):
    """Return (bpm, beat_times).

    beat_times are the ACTUAL detected beat positions from librosa — they carry
    the PHASE (where beat 1 sits), which a bare BPM does not. Building the grid
    from these positions (instead of from t=0) means the grid aligns to the
    music even when the track has an intro/anacrusis/leading silence — the issue
    that forces manual grid nudging in rhythm games.

    manual_bpm: if given, fixes the tempo but we still take the detected phase
    (first beat) so a known BPM aligns correctly.
    offset_ms: manual phase nudge applied on top, for the rare case auto-phase
    is wrong (positive = shift grid later).
    """
    import librosa

    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=HOP)
    bpm = float(tempo) if np.ndim(tempo) == 0 else float(tempo[0])
    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=HOP) \
        if len(beat_frames) else np.array([])

    if manual_bpm:
        bpm = float(manual_bpm)
    elif bpm and bpm < 110:   # likely half-tempo of a fast song
        bpm *= 2

    if offset_ms:
        beat_times = beat_times + offset_ms / 1000.0
    return bpm, beat_times


def _build_grid(beat_times: np.ndarray, bpm: float, dur: float,
                subdivision: float = 1.0) -> np.ndarray:
    """Build a beat grid that respects PHASE.

    Anchor on the first detected beat and step by the quarter (or its
    subdivision). If no beats were detected, fall back to a from-zero grid.
    subdivision: 1.0 = quarters, 0.5 = eighths.
    """
    q = 60.0 / bpm
    step = q * subdivision
    if beat_times.size:
        start = float(beat_times[0]) % step  # phase of the first beat
    else:
        start = 0.0
    return np.arange(start, dur, step)


def _hf_flatness_at(y_high: np.ndarray, sr: int, t: float) -> float:
    """Flatness measured INSIDE the high band around time t. Computed on the
    HF-only signal so low-frequency kick energy doesn't crush the metric
    (the bug that made every cymbal look tonal). High here = noisy = cymbal."""
    import librosa

    i = int(t * sr)
    w = y_high[max(0, i - 1024): i + 1024]
    if w.size < 256:
        return 0.0
    sf = librosa.feature.spectral_flatness(y=w)
    return float(np.mean(sf))


def _pitched_tom(y: np.ndarray, sr: int, t: float) -> tuple[bool, float]:
    """Decide if a mid/low hit is a TOM rather than snare/kick.

    A tom has a clear fundamental pitch and a longer decay; a snare is broadband
    noise; a kick is a very short low thump. Returns (is_tom, fundamental_hz).
    Heuristic: take the onset window, find the dominant low/mid spectral peak,
    and check decay length. Approximate by nature.
    """
    import librosa

    i = int(t * sr)
    w = y[max(0, i): i + int(0.18 * sr)]
    if w.size < 512:
        return False, 0.0
    # dominant frequency in 60-450 Hz (tom fundamental range)
    spec = np.abs(np.fft.rfft(w * np.hanning(w.size)))
    freqs = np.fft.rfftfreq(w.size, 1 / sr)
    band = (freqs >= 60) & (freqs <= 450)
    if not band.any():
        return False, 0.0
    pk = freqs[band][int(np.argmax(spec[band]))]
    peak_mag = spec[band].max()
    # tonal-ness: peak vs mean in band (tom = peaky, snare = flat)
    tonal = peak_mag / (np.mean(spec[band]) + 1e-9)
    # decay: energy at +120ms vs onset (tom rings, snare/kick die fast)
    head = np.sqrt(np.mean(w[: int(0.02 * sr)] ** 2)) + 1e-9
    tail_seg = y[i + int(0.10 * sr): i + int(0.16 * sr)]
    tail = np.sqrt(np.mean(tail_seg ** 2)) if tail_seg.size else 0.0
    decay = tail / head
    is_tom = tonal > 6.0 and decay > 0.25
    return is_tom, float(pk)


def transcribe_drums(wav: Path, midi_path: Path, sensitivity: float = 0.6,
                     bpm: float | None = None, grid_fill: bool = False,
                     cymbal_gate: float = 1.05, detect_toms: bool = False,
                     grid_offset_ms: float = 0.0) -> dict:
    """Detect drum hits in a drum stem and write a GM-percussion MIDI.

    Engine:
      kick  -> low-band energy onsets
      snare -> mid-band energy onsets
      cymbals/hats -> SUPERFLUX on the HF mel band (handles compressed, ringing
                      cymbals where amplitude transients are flattened)
    sensitivity: 0..1, higher = detect more (lower onset delta).
    bpm: known tempo; None = auto-detect (+half-tempo correction). The beat
         grid's PHASE is always taken from librosa's detected beat positions, so
         it aligns to the music regardless of intro/anacrusis.
    grid_offset_ms: manual phase nudge for the rare case auto-phase is off.
    grid_fill: in sections where cymbals are regular (~1 hit/beat), add the
               missing grid cymbals so a steady ride/hat isn't full of gaps.
               Only fills near existing cymbal activity (won't invent hits in
               silent bridges).
    Returns a dict of counts per instrument (+ _tempo, _total).
    """
    import librosa
    import pretty_midi

    wav = Path(wav)
    midi_path = Path(midi_path)
    y, sr = librosa.load(str(wav), sr=SR, mono=True)
    if y.size == 0:
        pretty_midi.PrettyMIDI().write(str(midi_path))
        return {"_tempo": 0, "_total": 0}

    dur = y.size / sr
    delta = float(np.interp(sensitivity, [0, 1], [0.18, 0.03]))
    use_bpm, beat_times = _estimate_beats(y, sr, bpm, grid_offset_ms)

    low = _band(y, sr, 20, 150)
    mid = _band(y, sr, 150, 2500)
    high = _band(y, sr, 6000, None)
    rms = {"low": float(np.sqrt(np.mean(low ** 2)) + 1e-9),
           "mid": float(np.sqrt(np.mean(mid ** 2)) + 1e-9),
           "high": float(np.sqrt(np.mean(high ** 2)) + 1e-9)}

    def energy_at(sig: np.ndarray, t: float) -> float:
        i = int(t * sr)
        w = sig[max(0, i - 256): i + 512]
        return float(np.sqrt(np.mean(w ** 2)) if w.size else 0.0)

    # Local-adaptive HF background: instead of one global RMS, track a rolling
    # median of HF energy so a hit is judged against its OWN neighbourhood. This
    # kills cymbal hallucinations in already-loud sections (a small bump on top
    # of a roaring section is NOT a new hit) and recovers quiet hits in calm
    # sections (where the global RMS would hide them).
    frame = int(0.02 * sr)
    hf_env = np.array([np.sqrt(np.mean(high[i:i + frame] ** 2))
                       for i in range(0, max(1, len(high) - frame), frame)])
    env_t = np.arange(len(hf_env)) * 0.02
    win = max(5, int(0.5 / 0.02))  # ~0.5 s rolling window
    local_bg = np.array([
        np.median(hf_env[max(0, k - win): k + win + 1]) + 1e-9
        for k in range(len(hf_env))
    ])

    def hf_local_ratio(t: float) -> float:
        k = min(len(hf_env) - 1, int(t / 0.02))
        return float(hf_env[k] / local_bg[k])

    # velocity from energy ratio: maps ~[1..6]x over background to MIDI 50..120.
    def vel(ratio: float) -> int:
        return int(max(40, min(127, 50 + (ratio - 1.0) * 16)))

    events: list[tuple[float, str, int]] = []  # (time, inst, velocity)

    # --- kick / snare: band energy onsets ---
    for t in _onsets(low, sr, delta):
        r = energy_at(low, t) / rms["low"]
        if r > 1.3:
            events.append((float(t), "kick", vel(r)))
    for t in _onsets(mid, sr, delta):
        r_mid = energy_at(mid, t) / rms["mid"]
        r_low = energy_at(low, t) / rms["low"]
        if r_mid > 1.3 and r_mid >= r_low:   # don't mislabel kick mid-bleed
            inst = "snare"
            if detect_toms:
                is_tom, hz = _pitched_tom(y, sr, t)
                if is_tom:
                    inst = ("tom_low" if hz < 130 else
                            "tom_mid" if hz < 250 else "tom_high")
            events.append((float(t), inst, vel(r_mid)))

    # --- cymbals/hats: superflux on HF mel, gated by LOCAL background ---
    # NOTE: open cymbals ring continuously, so local background ~= cymbal level
    # and a tight gate would erase real hits. Keep the gate gentle: it only
    # culls clear hallucinations (energy at/below its own neighbourhood) while
    # the velocity field below carries loudness so weak hits stay distinguishable.
    cym_times = _superflux_onsets(y, sr, fmin=6000, delta=delta)
    CYM_GATE = cymbal_gate  # gentle default 1.05; higher = fewer hits, less noise
    kept_cym = []
    for t in cym_times:
        lr = hf_local_ratio(t)
        if lr < CYM_GATE:
            continue  # not above its own neighbourhood -> likely a hallucination
        kept_cym.append(t)
        tail = energy_at(high, t + 0.12) / rms["high"]
        head = energy_at(high, t) / rms["high"]
        flat_hi = _hf_flatness_at(high, sr, t)
        if head > 2.2 and flat_hi > 0.25:
            inst = "crash"
        elif tail > 0.45 * head:
            inst = "open_hat"
        else:
            inst = "hat"
        events.append((float(t), inst, vel(lr)))
    cym_times = np.array(sorted(kept_cym))

    # --- optional grid fill for cymbals in active sections ---
    # Grid is PHASE-aligned to the detected beats, not anchored at t=0.
    if grid_fill and cym_times.size > 4 and use_bpm:
        q = 60.0 / use_bpm
        cym = cym_times
        grid = _build_grid(beat_times, use_bpm, dur, subdivision=1.0)
        for g in grid:
            d = np.min(np.abs(cym - g))
            near_active = np.any(np.abs(cym - g) < 2 * q)
            # fill only where cymbals are active nearby AND the HF energy at the
            # grid node actually rises above background (don't invent hits).
            if near_active and d > 0.5 * q and hf_local_ratio(g) >= CYM_GATE:
                events.append((float(g), "hat", 70))

    # --- snap cymbal/hat times to the PHASE-aligned 8th-note grid ---
    if use_bpm:
        q = 60.0 / use_bpm
        eighth = q / 2
        grid8 = _build_grid(beat_times, use_bpm, dur, subdivision=0.5)
        snap_tol = 0.10 * q
        snapped = []
        for t, inst, v in events:
            if inst in ("hat", "open_hat", "crash") and grid8.size:
                j = int(np.argmin(np.abs(grid8 - t)))
                if abs(grid8[j] - t) <= snap_tol:
                    t = float(grid8[j])
            snapped.append((t, inst, v))
        events = snapped

    # dedup identical (time, inst) within 40ms, keep the louder one
    events.sort()
    deduped: list[tuple[float, str, int]] = []
    for t, inst, v in events:
        hit = next((i for i, (s, e, _) in enumerate(deduped)
                    if abs(t - s) < 0.04 and e == inst), None)
        if hit is not None:
            if v > deduped[hit][2]:
                deduped[hit] = (deduped[hit][0], inst, v)
            continue
        deduped.append((t, inst, v))
    events = deduped

    # Write GM percussion MIDI (channel 10 == is_drum).
    pm = pretty_midi.PrettyMIDI(initial_tempo=use_bpm)
    drum = pretty_midi.Instrument(program=0, is_drum=True, name="drums")
    counts: dict[str, int] = {}
    for t, inst, v in sorted(events):
        drum.notes.append(pretty_midi.Note(velocity=v, pitch=GM[inst],
                                           start=t, end=t + 0.05))
        counts[inst] = counts.get(inst, 0) + 1
    pm.instruments.append(drum)
    midi_path.parent.mkdir(parents=True, exist_ok=True)
    pm.write(str(midi_path))

    counts["_tempo"] = round(use_bpm, 1)
    counts["_total"] = len(events)
    return counts


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="drum stem -> GM percussion MIDI")
    ap.add_argument("wav", type=Path)
    ap.add_argument("out", type=Path, nargs="?")
    ap.add_argument("--sensitivity", type=float, default=0.6)
    ap.add_argument("--bpm", type=float, default=None, help="known tempo; omit for auto")
    ap.add_argument("--grid-fill", action="store_true", help="fill missing cymbals on the beat grid")
    ap.add_argument("--cymbal-gate", type=float, default=1.05,
                    help="cymbal vs local-background gate (higher = fewer, cleaner)")
    ap.add_argument("--detect-toms", action="store_true",
                    help="try to split tom-toms (and rimshot) out of the snare class")
    ap.add_argument("--grid-offset-ms", type=float, default=0.0,
                    help="manual phase nudge for the beat grid (ms)")
    a = ap.parse_args()

    out = a.out or a.wav.with_suffix(".drums.mid")
    res = transcribe_drums(a.wav, out, sensitivity=a.sensitivity, bpm=a.bpm,
                           grid_fill=a.grid_fill, cymbal_gate=a.cymbal_gate,
                           detect_toms=a.detect_toms, grid_offset_ms=a.grid_offset_ms)
    print(f"tempo ~{res.pop('_tempo', '?')} BPM, total hits {res.pop('_total', 0)}")
    for k, v in sorted(res.items()):
        print(f"  {k:9s} {v}")
    print(f"wrote {out}")
