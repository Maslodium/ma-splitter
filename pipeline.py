"""
Robo-troupe offline pipeline: audio -> instrument stems -> MIDI per part.

Stage 1: Demucs separates the track into stems (drums/bass/other/vocals[/guitar/piano]).
Stage 2: Basic Pitch transcribes each melodic stem to a MIDI file.

Drums are skipped by default (pitch transcription is meaningless for percussion);
they get a dedicated onset-based path later. Use --include-drums to force it.
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

# basic-pitch supports several backends; force ONNX (no TensorFlow needed on Py3.12).
os.environ.setdefault("BASIC_PITCH_BACKEND", "onnx")

# basic-pitch prints emoji status lines; the Windows console codec (cp1251) can't
# encode them and raises UnicodeEncodeError *after* the MIDI is already saved.
# Force UTF-8 output so those prints don't crash the run.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Stems that Demucs htdemucs_6s can output. Drums handled separately.
MELODIC_STEMS = {"bass", "other", "vocals", "guitar", "piano"}


def find_ffmpeg() -> str | None:
    """Return a path to an ffmpeg binary, preferring the pip-bundled one."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def ensure_ffmpeg_on_path() -> None:
    """Demucs/librosa look for ffmpeg on PATH; inject the bundled one if needed."""
    if shutil.which("ffmpeg"):
        return
    exe = find_ffmpeg()
    if exe:
        os.environ["PATH"] = str(Path(exe).parent) + os.pathsep + os.environ["PATH"]


def pick_device(requested: str) -> str:
    import torch

    if requested == "cpu":
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    print("[warn] CUDA not available, falling back to CPU.")
    return "cpu"


def _run_demucs_once(audio: Path, out_root: Path, model: str, device: str,
                     segment: int | None, two_stems: str | None = None) -> tuple[int, str]:
    """Run demucs once, streaming output live while also capturing it.

    Returns (returncode, captured_output). Capturing lets us detect *why* a run
    failed (OOM vs. corrupt file) instead of masking it behind a generic error.

    two_stems: if set (e.g. "vocals"), demucs emits only <stem>.wav and
    no_<stem>.wav — used for the cascade's clean vocal-vs-instrumental split.
    """
    cmd = [
        sys.executable, "-u", "-m", "demucs",
        "-n", model,
        "-d", device,
        "-o", str(out_root),
        str(audio),
    ]
    if segment:
        cmd += ["--segment", str(segment)]
    if two_stems:
        cmd += ["--two-stems", two_stems]

    extra = f", two-stems={two_stems}" if two_stems else ""
    print(f"[demucs] separating ({model}, {device}, segment={segment}{extra}) ...")
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", bufsize=1,
    )
    captured: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        captured.append(line)
    code = proc.wait()
    return code, "".join(captured)


def _looks_like_oom(text: str) -> bool:
    t = text.lower()
    return ("out of memory" in t or "cuda error" in t or "cublas" in t
            or "alloc" in t and "fail" in t)


def run_demucs(audio: Path, out_root: Path, model: str, device: str, segment: int | None,
               two_stems: str | None = None) -> Path:
    """Run Demucs with a smart fallback chain, return the stem folder.

    On CUDA OOM: shrink --segment and retry on GPU before giving up; only then
    fall back to CPU. On any other error: surface the real demucs output.
    """
    if device == "cuda":
        # Progressively smaller windows fit the 4GB card; keep the GPU if we can.
        seg_chain = [s for s in (segment, 5, 3, 2) if s and s <= (segment or 99)]
        seen = set()
        for seg in seg_chain:
            if seg in seen:
                continue
            seen.add(seg)
            code, out = _run_demucs_once(audio, out_root, model, "cuda", seg, two_stems)
            if code == 0:
                return _stem_dir(out_root, model, audio)
            if _looks_like_oom(out):
                print(f"[demucs] CUDA out of memory at segment={seg} -> trying smaller")
                continue
            # Non-OOM failure on GPU: show it and try CPU as a last resort.
            print("[demucs] CUDA run failed (not OOM). Last output:")
            print(_tail(out))
            break
        print("[demucs] falling back to CPU ...")
        device = "cpu"

    code, out = _run_demucs_once(audio, out_root, model, device, segment, two_stems)
    if code != 0:
        raise RuntimeError(
            "demucs failed on CPU. Last output below — most often a corrupt/empty "
            "audio file or a missing ffmpeg codec:\n" + _tail(out))
    return _stem_dir(out_root, model, audio)


# Models for each cascade stage. Stage 1 = best vocal isolation; stage 2 = 6-way.
CASCADE_VOCAL_MODEL = "htdemucs_ft"
CASCADE_ARRANGE_MODEL = "htdemucs_6s"
ARRANGEMENT_STEMS = ("drums", "bass", "guitar", "piano", "other")


def separate_cascade(audio: Path, stems_root: Path, device: str, segment: int | None) -> Path:
    """Two-stage 'one at a time, precisely' separation.

    Stage 1: htdemucs_ft, --two-stems=vocals -> clean ALL-vocals + instrumental.
             (ft is the strongest vocal model; pulling every voice out here means
             it can't leak into guitar/other downstream.)
    Stage 2: htdemucs_6s on the instrumental only -> drums/bass/guitar/piano/other,
             cleaner because the vocal energy is already gone.

    Returns a folder holding the assembled final stems (vocals + arrangement),
    so the transcription stage downstream needs no changes.
    """
    work = stems_root / "cascade" / audio.stem
    stage1_root = work / "1_vocals"
    stage2_root = work / "2_arrangement"
    final = work / "final"
    final.mkdir(parents=True, exist_ok=True)

    # --- Stage 1: clean vocals vs instrumental ---
    print("\n=== cascade stage 1: isolating ALL vocals (htdemucs_ft) ===")
    voc_dir = run_demucs(audio, stage1_root, CASCADE_VOCAL_MODEL, device, segment,
                         two_stems="vocals")
    vocals = voc_dir / "vocals.wav"
    no_vocals = voc_dir / "no_vocals.wav"
    if not vocals.is_file() or not no_vocals.is_file():
        raise RuntimeError(f"stage 1 did not produce vocals/no_vocals in {voc_dir}")
    shutil.copy2(vocals, final / "vocals.wav")

    # --- Stage 2: decompose the instrumental ---
    print("\n=== cascade stage 2: decomposing arrangement (htdemucs_6s) ===")
    inst_dir = run_demucs(no_vocals, stage2_root, CASCADE_ARRANGE_MODEL, device, segment)
    for stem in ARRANGEMENT_STEMS:
        src = inst_dir / f"{stem}.wav"
        if src.is_file():
            shutil.copy2(src, final / f"{stem}.wav")
        else:
            print(f"[cascade] note: stage 2 had no '{stem}' stem (skipped)")

    print(f"\n[cascade] assembled stems -> {final}")
    return final


# Which stems each demucs model can output (guitar/piano only exist in 6s).
MODEL_STEMS = {
    "htdemucs":     {"drums", "bass", "other", "vocals"},
    "htdemucs_ft":  {"drums", "bass", "other", "vocals"},
    "htdemucs_6s":  {"drums", "bass", "other", "vocals", "guitar", "piano"},
    "mdx_extra":    {"drums", "bass", "other", "vocals"},
    "mdx_extra_q":  {"drums", "bass", "other", "vocals"},
}

# parallel: each stem from the untouched original. All instruments from one 6s
# pass (already taken from the full mix), vocals from a dedicated ft pass.
PARALLEL_PLAN = {
    "vocals": "htdemucs_ft",
    "bass":   "htdemucs_6s",
    "drums":  "htdemucs_6s",
    "guitar": "htdemucs_6s",
    "piano":  "htdemucs_6s",
    "other":  "htdemucs_6s",
}

# parallel-deep: a DIFFERENT model per instrument, each run over the original.
# bass/drums use the 4-stem htdemucs (often a tighter low end / transients);
# guitar/piano need 6s (only model with those); vocals use ft. More passes =
# more time, but each instrument gets a model better suited to it.
DEEP_PARALLEL_PLAN = {
    "vocals": "htdemucs_ft",
    "bass":   "htdemucs",
    "drums":  "htdemucs",
    "guitar": "htdemucs_6s",
    "piano":  "htdemucs_6s",
    "other":  "htdemucs_6s",
}


def separate_parallel(audio: Path, stems_root: Path, device: str, segment: int | None,
                      parts: set[str] | None = None, plan: dict | None = None,
                      tag: str = "parallel") -> Path:
    """Independent 'whole plank for every cut' separation.

    Each requested stem is extracted by a demucs pass over the ORIGINAL mix
    (never a residual). Parts that share a model are harvested from one pass of
    that model — that pass already derives each stem from the full original, so
    the 'from the untouched original' guarantee holds without redundant runs.

    plan: {part -> model}. Vocals always use --two-stems for the cleanest split.
    parts: which final stems are needed (lets us skip whole models).
    """
    plan = plan or PARALLEL_PLAN
    parts = parts or set(plan)
    work = stems_root / tag / audio.stem
    final = work / "final"
    final.mkdir(parents=True, exist_ok=True)

    # Group requested parts by the model that produces them, validating that the
    # model can actually output that stem (guitar/piano => 6s only).
    by_model: dict[str, list[str]] = {}
    for part in parts:
        model = plan.get(part)
        if not model:
            continue
        if part not in MODEL_STEMS.get(model, set()):
            print(f"[{tag}] '{model}' can't output '{part}' -> falling back to htdemucs_6s")
            model = "htdemucs_6s"
        by_model.setdefault(model, []).append(part)

    for model, want in sorted(by_model.items()):
        # Vocals-only pass uses two-stems for the cleanest vocal isolation.
        if want == ["vocals"]:
            print(f"\n=== {tag}: vocals from original ({model}) ===")
            d = run_demucs(audio, work / model, model, device, segment, two_stems="vocals")
        else:
            print(f"\n=== {tag}: {', '.join(sorted(want))} from original ({model}) ===")
            d = run_demucs(audio, work / model, model, device, segment)
        for part in want:
            src = d / f"{part}.wav"
            if src.is_file():
                shutil.copy2(src, final / f"{part}.wav")
            else:
                print(f"[{tag}] note: {model} pass had no '{part}' stem (skipped)")

    print(f"\n[{tag}] assembled stems -> {final}")
    return final


# best-of: take each stem from the SOURCE that measured cleanest on dense music.
# Sources:
#   "ft"       - htdemucs_ft two-stems over the original (best vocals)
#   "6s"       - htdemucs_6s over the original (only model with guitar/piano)
#   "htdemucs" - 4-stem htdemucs over the original (tighter bass/drums transients)
#   "cascade"  - htdemucs_6s over the vocal-removed residual (best piano: no vocal
#                overtones to confuse it)
BESTOF_PLAN = {
    "vocals": "ft",
    "bass":   "htdemucs",
    "drums":  "htdemucs",
    "guitar": "6s",
    "other":  "6s",
    "piano":  "cascade",
}
BESTOF_SOURCES = ("ft", "6s", "htdemucs", "cascade")
_SOURCE_MODEL = {"ft": "htdemucs_ft", "6s": "htdemucs_6s",
                 "htdemucs": "htdemucs", "cascade": "htdemucs_6s"}


def separate_bestof(audio: Path, stems_root: Path, device: str, segment: int | None,
                    parts: set[str] | None = None, plan: dict | None = None) -> Path:
    """Assemble each stem from its best-measured source, minimising demucs passes.

    A given source/model is run at most once and all stems needing it are
    harvested from that single pass.
    """
    plan = {**BESTOF_PLAN, **(plan or {})}
    parts = parts or set(plan)
    tag = "bestof"
    work = stems_root / tag / audio.stem
    final = work / "final"
    final.mkdir(parents=True, exist_ok=True)

    # Validate sources; guitar/piano only exist via a 6s-based source.
    resolved: dict[str, str] = {}
    for part in parts:
        src = plan.get(part, BESTOF_PLAN.get(part, "6s"))
        model = _SOURCE_MODEL.get(src, "htdemucs_6s")
        if part in ("guitar", "piano") and src not in ("6s", "cascade"):
            print(f"[bestof] '{src}' can't yield '{part}' -> using 6s")
            src = "6s"
        elif part not in MODEL_STEMS.get(model, set()) and src != "cascade":
            print(f"[bestof] '{model}' can't output '{part}' -> using 6s")
            src = "6s"
        resolved[part] = src

    sources_needed = set(resolved.values())
    no_vocals: Path | None = None

    # 1) ft pass over original -> vocals (+ no_vocals, reused by 'cascade').
    if "ft" in sources_needed or "cascade" in sources_needed:
        print("\n=== bestof: ft pass over original (vocals) ===")
        d = run_demucs(audio, work / "ft", "htdemucs_ft", device, segment, two_stems="vocals")
        if "ft" in sources_needed and (d / "vocals.wav").is_file():
            shutil.copy2(d / "vocals.wav", final / "vocals.wav")
        if (d / "no_vocals.wav").is_file():
            no_vocals = d / "no_vocals.wav"

    # 2) 6s over original.
    if "6s" in sources_needed:
        want = [p for p, s in resolved.items() if s == "6s"]
        print(f"\n=== bestof: 6s over original ({', '.join(sorted(want))}) ===")
        d = run_demucs(audio, work / "6s", "htdemucs_6s", device, segment)
        for p in want:
            if (d / f"{p}.wav").is_file():
                shutil.copy2(d / f"{p}.wav", final / f"{p}.wav")

    # 3) htdemucs over original.
    if "htdemucs" in sources_needed:
        want = [p for p, s in resolved.items() if s == "htdemucs"]
        print(f"\n=== bestof: htdemucs over original ({', '.join(sorted(want))}) ===")
        d = run_demucs(audio, work / "htdemucs", "htdemucs", device, segment)
        for p in want:
            if (d / f"{p}.wav").is_file():
                shutil.copy2(d / f"{p}.wav", final / f"{p}.wav")

    # 4) 6s over the vocal-removed residual (cascade-style).
    if "cascade" in sources_needed:
        want = [p for p, s in resolved.items() if s == "cascade"]
        if no_vocals and no_vocals.is_file():
            print(f"\n=== bestof: 6s over no-vocals residual ({', '.join(sorted(want))}) ===")
            d = run_demucs(no_vocals, work / "cascade", "htdemucs_6s", device, segment)
            for p in want:
                if (d / f"{p}.wav").is_file():
                    shutil.copy2(d / f"{p}.wav", final / f"{p}.wav")
        else:
            print("[bestof] no residual available for cascade sources; skipped")

    chosen = ", ".join(f"{p}:{resolved[p]}" for p in sorted(resolved))
    print(f"\n[bestof] sources -> {chosen}")
    print(f"[bestof] assembled stems -> {final}")
    return final


# Which sources can yield each stem (for the adaptive search).
_STEM_SOURCES = {
    "vocals": ["ft", "6s", "htdemucs"],
    "bass":   ["6s", "htdemucs"],
    "drums":  ["6s", "htdemucs"],
    "other":  ["6s", "htdemucs", "cascade"],
    "guitar": ["6s", "cascade"],
    "piano":  ["6s", "cascade"],
}


def separate_bestof_auto(audio: Path, stems_root: Path, device: str, segment: int | None,
                         parts: set[str] | None = None) -> Path:
    """Adaptive best-of: run every source, transcribe each candidate version of
    each stem, and keep the version with the LOWEST out-of-key fraction
    (auto-detected key). Most expensive mode — runs all demucs passes — but picks
    the cleanest source per stem from measured evidence, not a fixed map.
    """
    parts = parts or set(_STEM_SOURCES)
    work = stems_root / "bestof-auto" / audio.stem
    final = work / "final"
    final.mkdir(parents=True, exist_ok=True)

    # 1) run each source once, harvest all stems it can give us.
    src_dirs: dict[str, Path] = {}
    print("\n=== bestof-auto: ft over original ===")
    d_ft = run_demucs(audio, work / "ft", "htdemucs_ft", device, segment, two_stems="vocals")
    src_dirs["ft"] = d_ft
    no_vocals = d_ft / "no_vocals.wav" if (d_ft / "no_vocals.wav").is_file() else None
    print("\n=== bestof-auto: 6s over original ===")
    src_dirs["6s"] = run_demucs(audio, work / "6s", "htdemucs_6s", device, segment)
    print("\n=== bestof-auto: htdemucs over original ===")
    src_dirs["htdemucs"] = run_demucs(audio, work / "htdemucs", "htdemucs", device, segment)
    if no_vocals:
        print("\n=== bestof-auto: 6s over no-vocals residual (cascade) ===")
        src_dirs["cascade"] = run_demucs(no_vocals, work / "cascade", "htdemucs_6s",
                                         device, segment)

    # 2) for each wanted stem, score every candidate source by out-of-key%.
    print("\n[bestof-auto] scoring candidates per stem (lower out-of-key% = better)")
    chosen: dict[str, str] = {}
    scratch = work / "_score"
    scratch.mkdir(parents=True, exist_ok=True)
    for part in sorted(parts):
        candidates = []
        for src in _STEM_SOURCES.get(part, []):
            d = src_dirs.get(src)
            wav = d / f"{part}.wav" if d else None
            if not wav or not wav.is_file():
                continue
            if src == "ft" and part == "vocals":
                wav = d / "vocals.wav"
            mid = scratch / f"{part}_{src}.mid"
            try:
                profile = dict(PART_PROFILES.get(part, DEFAULT_PROFILE))
                transcribe_stem(wav, mid, profile)
                pct = _out_of_key_pct(mid)
                candidates.append((pct, src, wav))
                print(f"   {part:7s} {src:9s} out-of-key {pct:5.1f}%")
            except Exception as e:  # noqa: BLE001
                print(f"   {part:7s} {src:9s} failed ({e!r})")
        if not candidates:
            continue
        candidates.sort()
        best_pct, best_src, best_wav = candidates[0]
        chosen[part] = best_src
        shutil.copy2(best_wav, final / f"{part}.wav")
        print(f"   -> {part}: chose {best_src} ({best_pct:.1f}%)")

    print(f"\n[bestof-auto] winners -> "
          + ", ".join(f"{p}:{chosen[p]}" for p in sorted(chosen)))
    print(f"[bestof-auto] assembled stems -> {final}")
    return final


def _out_of_key_pct(midi_path: Path) -> float:
    """Out-of-scale note fraction vs the MIDI's own auto-detected key."""
    est = estimate_key([midi_path])
    if not est:
        return 100.0
    _, tonic, mode, _ = est
    import pretty_midi

    scale = _MAJOR_SCALE if mode == "maj" else _MINOR_SCALE
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    notes = [n for inst in pm.instruments for n in inst.notes]
    if not notes:
        return 100.0
    oo = sum(1 for n in notes if ((n.pitch - tonic) % 12) not in scale)
    return 100.0 * oo / len(notes)


def _role_of(midi_path: Path) -> tuple[float, float, float]:
    """Return (polyphony, mean_pitch, density) features for solo/rhythm guess."""
    import pretty_midi

    pm = pretty_midi.PrettyMIDI(str(midi_path))
    notes = [n for inst in pm.instruments for n in inst.notes]
    if not notes:
        return 0.0, 0.0, 0.0
    # polyphony: average simultaneous notes (notes overlapping in time)
    starts = sorted(n.start for n in notes)
    overlaps = 0
    for n in notes:
        overlaps += sum(1 for m in notes if m is not n and m.start < n.end and m.end > n.start)
    poly = overlaps / len(notes)
    mean_pitch = sum(n.pitch for n in notes) / len(notes)
    dur = pm.get_end_time() or 1.0
    density = len(notes) / dur
    return poly, mean_pitch, density


def classify_lead_rhythm(midi_a: Path, midi_b: Path) -> tuple[str, str]:
    """Combined heuristic: the LEAD part has lower polyphony, higher mean pitch,
    and (usually) lower note density than the chord-strumming RHYTHM part.
    Returns (label_a, label_b) each in {'lead','rhythm'}."""
    pa, ma, da = _role_of(midi_a)
    pb, mb, db = _role_of(midi_b)
    # score: higher => more 'lead-like'. low polyphony + high pitch + low density.
    def lead_score(poly, mp, dens):
        return (-poly) + (mp / 12.0) + (-dens)
    sa, sb = lead_score(pa, ma, da), lead_score(pb, mb, db)
    if sa >= sb:
        return "lead", "rhythm"
    return "rhythm", "lead"


def _tail(text: str, n: int = 15) -> str:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    return "\n".join(lines[-n:])


def _stem_dir(out_root: Path, model: str, audio: Path) -> Path:
    stem_dir = out_root / model / audio.stem
    if not stem_dir.is_dir():
        raise RuntimeError(f"expected stems at {stem_dir}, not found")
    return stem_dir


_BP_MODEL = None


def _get_bp_model():
    """Build the ONNX Model once and reuse it across stems."""
    global _BP_MODEL
    if _BP_MODEL is None:
        from basic_pitch.inference import Model
        from basic_pitch import ICASSP_2022_MODEL_PATH

        _BP_MODEL = Model(ICASSP_2022_MODEL_PATH)
    return _BP_MODEL


# Per-part transcription profiles, derived from the WKWK analysis:
#   min/max Hz  -> physical band of the instrument (kills overtone/bleed notes
#                  outside it, e.g. the 29 Hz sub-bass leaking into 'vocals')
#   onset/frame -> basic-pitch sensitivity. Higher = fewer, more confident notes
#                  (good for dense polyphony like the over-full guitar).
#   min_len_ms  -> drop very short blips (transcription noise).
# These are DEFAULTS; the GUI can override all of them globally.
PART_PROFILES = {
    # bass/piano get longer min_len + slightly higher onset to stop the note
    # fragmentation we saw in v2 (bass 1722->2064 was over-split).
    "bass":   {"min_freq": 30,  "max_freq": 400,  "onset": 0.6, "frame": 0.3, "min_len_ms": 130},
    "guitar": {"min_freq": 80,  "max_freq": 1320, "onset": 0.6, "frame": 0.4, "min_len_ms": 90},
    "piano":  {"min_freq": 50,  "max_freq": 2100, "onset": 0.55, "frame": 0.35, "min_len_ms": 120},
    "vocals": {"min_freq": 80,  "max_freq": 1100, "onset": 0.5, "frame": 0.3, "min_len_ms": 100},
    "other":  {"min_freq": 60,  "max_freq": 2100, "onset": 0.55, "frame": 0.35, "min_len_ms": 110},
    "drums":  {"min_freq": 30,  "max_freq": 3000, "onset": 0.5, "frame": 0.3, "min_len_ms": 40},
}

DEFAULT_PROFILE = {"min_freq": None, "max_freq": None, "onset": 0.5, "frame": 0.3, "min_len_ms": 80}


def analyze_stem(wav: Path) -> dict:
    """Measure a stem before transcription so thresholds can follow the audio."""
    import librosa
    import numpy as np

    y, sr = librosa.load(str(wav), sr=22050, mono=True)
    if y.size == 0:
        return {"duration": 0.0, "rms": 0.0, "peak": 0.0, "flatness": 1.0,
                "onset_density": 0.0, "harmonic_ratio": 0.0, "centroid": 0.0}

    rms = float(np.sqrt(np.mean(y ** 2)))
    peak = float(np.max(np.abs(y)))
    flatness = float(np.mean(librosa.feature.spectral_flatness(y=y)))
    centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=512)
    onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, hop_length=512)
    duration = float(y.size / sr)
    harmonic, percussive = librosa.effects.hpss(y)
    h_rms = float(np.sqrt(np.mean(harmonic ** 2)))
    p_rms = float(np.sqrt(np.mean(percussive ** 2)))
    return {
        "duration": duration,
        "rms": rms,
        "peak": peak,
        "flatness": flatness,
        "onset_density": float(len(onsets) / max(duration, 0.001)),
        "harmonic_ratio": float(h_rms / (h_rms + p_rms + 1e-9)),
        "centroid": centroid,
    }


def choose_adaptive_profile(part: str, base: dict, metrics: dict) -> dict:
    """Tune Basic Pitch/pYIN thresholds from stem measurements."""
    p = dict(base)
    dense = metrics.get("onset_density", 0.0) > 5.0
    noisy = metrics.get("flatness", 0.0) > 0.12
    percussive = metrics.get("harmonic_ratio", 1.0) < 0.45
    quiet = metrics.get("rms", 0.0) < 0.015

    if dense or noisy:
        p["onset"] = min(0.82, float(p["onset"]) + 0.08)
        p["frame"] = min(0.78, float(p["frame"]) + 0.06)
        p["min_len_ms"] = max(float(p["min_len_ms"]), 110)
    if percussive and part in {"guitar", "other", "piano"}:
        p["onset"] = min(0.88, float(p["onset"]) + 0.06)
        p["min_len_ms"] = max(float(p["min_len_ms"]), 130)
    if quiet:
        p["onset"] = max(0.35, float(p["onset"]) - 0.06)
        p["frame"] = max(0.22, float(p["frame"]) - 0.04)
    if part == "bass":
        p["max_freq"] = min(float(p.get("max_freq") or 420), 420)
        p["min_len_ms"] = max(float(p["min_len_ms"]), 140)
    if part == "vocals" and noisy:
        p["max_freq"] = min(float(p.get("max_freq") or 1200), 1200)
    return p


def choose_adaptive_engine(part: str, user_mono: bool, piano_engine: str,
                           metrics: dict) -> tuple[object, str]:
    """Pick the least-wrong local transcriber for this stem."""
    if user_mono:
        return transcribe_stem_mono, "pYIN-mono"
    if part == "piano" and piano_engine == "onsets-frames":
        return transcribe_stem_piano_of, "onsets-frames"

    monoline = (
        part in {"bass", "vocals"}
        and metrics.get("harmonic_ratio", 0.0) > 0.58
        and metrics.get("flatness", 1.0) < 0.08
        and metrics.get("onset_density", 99.0) < 4.0
    )
    if monoline:
        return transcribe_stem_mono, "pYIN-auto"
    return transcribe_stem, "basic-pitch"


def preprocess_stem_for_midi(wav: Path, out_wav: Path, part: str,
                             profile: dict, metrics: dict) -> Path:
    """Make a cleaner analysis signal for transcription without replacing stems.

    The saved audition stem remains the Demucs output. This creates a private
    mono helper WAV for the MIDI engine: instrument bandpass, gentle harmonic
    emphasis and an adaptive noise gate.
    """
    import librosa
    import numpy as np
    import scipy.signal as sps
    import soundfile as sf

    y, sr = librosa.load(str(wav), sr=44100, mono=True)
    if y.size == 0:
        return wav

    lo = profile.get("min_freq")
    hi = profile.get("max_freq")
    ny = sr / 2
    try:
        if lo and hi and hi < ny:
            sos = sps.butter(4, [float(lo) / ny, float(hi) / ny], btype="bandpass", output="sos")
            y = sps.sosfiltfilt(sos, y)
        elif lo:
            sos = sps.butter(4, float(lo) / ny, btype="highpass", output="sos")
            y = sps.sosfiltfilt(sos, y)
        elif hi and hi < ny:
            sos = sps.butter(4, float(hi) / ny, btype="lowpass", output="sos")
            y = sps.sosfiltfilt(sos, y)
    except ValueError:
        pass

    if part != "bass" and metrics.get("harmonic_ratio", 0.5) < 0.75:
        harmonic, percussive = librosa.effects.hpss(y)
        y = harmonic + 0.18 * percussive

    frame = 2048
    hop = 512
    rms = librosa.feature.rms(y=y, frame_length=frame, hop_length=hop, center=True)[0]
    if rms.size:
        floor = float(np.percentile(rms, 25))
        gate = max(floor * 1.8, 1e-5)
        gain_frames = np.clip((rms - gate) / (gate * 1.5 + 1e-9), 0.18, 1.0)
        idx = np.minimum(np.arange(y.size) // hop, gain_frames.size - 1)
        y = y * gain_frames[idx]

    peak = float(np.max(np.abs(y)))
    if peak > 0:
        y = 0.92 * y / peak

    out_wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_wav), y.astype(np.float32), sr)
    return out_wav


def transcribe_stem(wav: Path, midi_path: Path, profile: dict | None = None) -> int:
    """Transcribe one stem to a single .mid file, return the note count.

    We call basic-pitch's low-level predict() and write the MIDI ourselves,
    instead of predict_and_save(). predict_and_save() prints emoji status lines
    that crash on the Windows cp1251 console *after* writing the file; doing the
    write ourselves sidesteps that entirely and gives a clean output path.

    profile: dict with min_freq/max_freq/onset/frame/min_len_ms. Frequency bounds
    and thresholds are passed straight to basic-pitch so notes outside an
    instrument's physical band (overtones, bleed) are never created.
    """
    from basic_pitch.inference import predict

    p = {**DEFAULT_PROFILE, **(profile or {})}
    midi_path.parent.mkdir(parents=True, exist_ok=True)
    _model_output, midi_data, _note_events = predict(
        str(wav),
        _get_bp_model(),
        onset_threshold=p["onset"],
        frame_threshold=p["frame"],
        minimum_note_length=p["min_len_ms"],
        minimum_frequency=p["min_freq"],
        maximum_frequency=p["max_freq"],
    )
    midi_data.write(str(midi_path))
    return sum(len(inst.notes) for inst in midi_data.instruments)


def transcribe_stem_mono(wav: Path, midi_path: Path, profile: dict | None = None) -> int:
    """Monophonic transcription via pYIN — for parts that are ONE line at a time
    (bass, a lead vocal). pYIN tracks a single pitch contour, so it can't
    hallucinate the chord/overtone clutter a polyphonic detector adds. We
    segment the contour into notes by pitch stability + voicing.

    Uses only librosa (no extra deps). Honours the profile's frequency band and
    minimum note length.
    """
    import librosa
    import numpy as np
    import pretty_midi

    p = {**DEFAULT_PROFILE, **(profile or {})}
    fmin = p["min_freq"] or 40.0
    fmax = p["max_freq"] or 1000.0
    min_len = (p["min_len_ms"] or 80) / 1000.0

    y, sr = librosa.load(str(wav), sr=22050, mono=True)
    hop = 256
    f0, voiced, vprob = librosa.pyin(
        y, sr=sr, fmin=float(fmin), fmax=float(fmax),
        frame_length=2048, hop_length=hop)
    times = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=hop)

    midi_path.parent.mkdir(parents=True, exist_ok=True)
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0)

    # Segment the f0 contour into notes: a note continues while voiced and the
    # rounded MIDI pitch stays the same; a pitch change or unvoiced gap ends it.
    cur_pitch = None
    start_t = 0.0
    last_t = 0.0

    def emit(pitch, t0, t1):
        if pitch is None or (t1 - t0) < min_len:
            return
        inst.notes.append(pretty_midi.Note(velocity=90, pitch=int(pitch),
                                           start=float(t0), end=float(t1)))

    for i, (f, v) in enumerate(zip(f0, voiced)):
        t = times[i]
        if v and f and not np.isnan(f):
            m = int(round(librosa.hz_to_midi(f)))
            if cur_pitch is None:
                cur_pitch, start_t = m, t
            elif m != cur_pitch:
                emit(cur_pitch, start_t, t)
                cur_pitch, start_t = m, t
        else:
            if cur_pitch is not None:
                emit(cur_pitch, start_t, t)
                cur_pitch = None
        last_t = t
    if cur_pitch is not None:
        emit(cur_pitch, start_t, last_t)

    pm.instruments.append(inst)
    pm.write(str(midi_path))
    return len(inst.notes)


_OF_MODEL = None


def transcribe_stem_piano_of(wav: Path, midi_path: Path, profile: dict | None = None) -> int:
    """Polyphonic piano transcription via Onsets & Frames (Kong, PyTorch).

    A model trained specifically on solo piano (MAESTRO) — more accurate than the
    instrument-agnostic basic-pitch on dense piano polyphony. Requires the
    piano_transcription_inference package and its ~165 MB checkpoint (the
    installer fetches the checkpoint; see install.py). Falls back to basic-pitch
    if the package/checkpoint is unavailable.
    """
    global _OF_MODEL
    try:
        import librosa
        from piano_transcription_inference import PianoTranscription, sample_rate
    except Exception as e:  # noqa: BLE001
        print(f"[piano-of] package unavailable ({e!r}); falling back to basic-pitch")
        return transcribe_stem(wav, midi_path, profile)

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if _OF_MODEL is None:
        _OF_MODEL = PianoTranscription(device=device)
    audio, _ = librosa.load(str(wav), sr=sample_rate, mono=True)
    midi_path.parent.mkdir(parents=True, exist_ok=True)
    _OF_MODEL.transcribe(audio, str(midi_path))

    import pretty_midi

    pm = pretty_midi.PrettyMIDI(str(midi_path))
    return sum(len(i.notes) for i in pm.instruments)


# ---- Key detection + in-key filtering (Krumhansl-Schmuckler) ----------------

_KS_MAJOR = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
_KS_MINOR = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
_NAME_TO_PC = {"C": 0, "C#": 1, "DB": 1, "D": 2, "D#": 3, "EB": 3, "E": 4, "F": 5,
               "F#": 6, "GB": 6, "G": 7, "G#": 8, "AB": 8, "A": 9, "A#": 10, "BB": 10, "B": 11}
_MAJOR_SCALE = {0, 2, 4, 5, 7, 9, 11}
_MINOR_SCALE = {0, 2, 3, 5, 7, 8, 10}  # natural minor

# A part losing more than this fraction to the key filter is left untouched:
# it usually means the key guess is wrong, or the part is legitimately chromatic.
KEY_FILTER_SAFETY = 0.35


def _pc_histogram(midi_paths) -> list[float]:
    import pretty_midi

    hist = [0.0] * 12
    for f in midi_paths:
        if not f.is_file():
            continue
        pm = pretty_midi.PrettyMIDI(str(f))
        for inst in pm.instruments:
            for n in inst.notes:
                hist[n.pitch % 12] += (n.end - n.start)
    return hist


def estimate_key(midi_paths):
    """Return (label like 'G:min', tonic int, mode 'maj'/'min', confidence) or None."""
    import numpy as np

    hist = _pc_histogram(midi_paths)
    if sum(hist) == 0:
        return None
    h = np.array(hist) / sum(hist)
    best = None
    for tonic in range(12):
        for mode, prof in (("maj", _KS_MAJOR), ("min", _KS_MINOR)):
            pr = np.roll(np.array(prof), tonic)
            pr = pr / pr.sum()
            corr = float(np.corrcoef(h, pr)[0, 1])
            if best is None or corr > best[3]:
                best = (f"{_NOTE_NAMES[tonic]}:{mode}", tonic, mode, corr)
    return best


def parse_key(label: str):
    """'G:min' / 'Gm' / 'Bb:maj' -> (tonic_pc, mode) or None."""
    if not label:
        return None
    s = label.strip()
    mode = "min" if (":min" in s.lower() or s.lower().endswith("m")) else "maj"
    root = s.split(":")[0].rstrip("mM").strip().upper()
    pc = _NAME_TO_PC.get(root)
    if pc is None:
        return None
    return pc, mode


def filter_midi_in_key(midi_path: Path, tonic: int, mode: str) -> tuple[int, int]:
    """Remove notes whose pitch-class is outside the key's diatonic scale.

    Returns (kept, removed). Honours KEY_FILTER_SAFETY: if removal would exceed
    that fraction, the file is left unchanged (likely wrong key / chromatic part).
    """
    import pretty_midi

    scale = _MAJOR_SCALE if mode == "maj" else _MINOR_SCALE
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    total = sum(len(i.notes) for i in pm.instruments)
    if total == 0:
        return 0, 0
    would_remove = sum(1 for i in pm.instruments for n in i.notes
                       if ((n.pitch - tonic) % 12) not in scale)
    if would_remove / total > KEY_FILTER_SAFETY:
        return total, -would_remove  # negative flags "skipped for safety"

    removed = 0
    for inst in pm.instruments:
        keep = [n for n in inst.notes if ((n.pitch - tonic) % 12) in scale]
        removed += len(inst.notes) - len(keep)
        inst.notes = keep
    pm.write(str(midi_path))
    return total - removed, removed


def clean_midi(midi_path: Path, drop_octave_ghosts: bool = True,
               max_polyphony: int = 0) -> tuple[int, int]:
    """Post-process a transcribed MIDI to remove transcription clutter.

    drop_octave_ghosts: when a note and a note an exact octave (or two) above it
        start within 50 ms and the upper one does not clearly outlast the lower,
        the upper is treated as a harmonic the transcriber mistook for a real
        note, and removed. (Measured as the dominant artifact on guitar.)
    max_polyphony: if > 0, at any moment keep at most this many simultaneous
        notes, dropping the shortest extras (curbs polyphonic smear). 0 = off.

    Returns (kept, removed).
    """
    import pretty_midi

    pm = pretty_midi.PrettyMIDI(str(midi_path))
    notes = [n for inst in pm.instruments for n in inst.notes]
    total = len(notes)
    if total == 0:
        return 0, 0
    notes.sort(key=lambda n: (n.start, n.pitch))
    kill: set[int] = set()

    if drop_octave_ghosts:
        for i, n in enumerate(notes):
            for j in range(i + 1, len(notes)):
                m = notes[j]
                if m.start - n.start > 0.05:
                    break
                if id(m) in kill or id(n) in kill:
                    continue
                d = m.pitch - n.pitch
                if d > 0 and d % 12 == 0:
                    if (m.end - m.start) <= (n.end - n.start) * 1.1:
                        kill.add(id(m))

    if max_polyphony and max_polyphony > 0:
        # Sample the timeline, keeping an incremental active-note window. The
        # old version rebuilt active notes by scanning the whole file at every
        # grid tick, which became very slow on dense transcription output.
        import numpy as np

        end = pm.get_end_time()
        grid = np.arange(0, end, 0.05)
        by_start = sorted(notes, key=lambda n: n.start)
        active: list = []
        start_i = 0
        for t in grid:
            while start_i < len(by_start) and by_start[start_i].start <= t:
                active.append(by_start[start_i])
                start_i += 1
            active = [n for n in active if id(n) not in kill and t < n.end]
            if len(active) > max_polyphony:
                ranked = sorted(active, key=lambda n: (n.end - n.start))  # shortest first
                for n in ranked[:len(active) - max_polyphony]:
                    kill.add(id(n))

    for inst in pm.instruments:
        inst.notes = [n for n in inst.notes if id(n) not in kill]
    pm.write(str(midi_path))
    return total - len(kill), len(kill)


def clean_midi_smart(midi_path: Path, part: str, profile: dict,
                     metrics: dict | None = None) -> dict:
    """Second-pass musical cleanup after the generic clutter remover.

    This pass is deliberately conservative: merge tiny same-pitch gaps, remove
    very short quiet notes, and suppress isolated short blips. It avoids key or
    scale assumptions so it works on metal riffs, chromatic runs and modal parts.
    """
    import pretty_midi

    pm = pretty_midi.PrettyMIDI(str(midi_path))
    total_before = sum(len(inst.notes) for inst in pm.instruments)
    if total_before == 0:
        return {"before": 0, "after": 0, "removed": 0, "merged": 0}

    min_len = float(profile.get("min_len_ms") or 80) / 1000.0
    gap = 0.045 if part in {"bass", "vocals"} else 0.03
    quiet_velocity = 34 if part in {"bass", "piano"} else 28
    isolated_gap = 0.16
    merged_count = 0
    removed_count = 0

    for inst in pm.instruments:
        if inst.is_drum:
            continue
        notes = sorted(inst.notes, key=lambda n: (n.pitch, n.start, n.end))
        merged = []
        for note in notes:
            if merged and note.pitch == merged[-1].pitch and 0 <= note.start - merged[-1].end <= gap:
                merged[-1].end = max(merged[-1].end, note.end)
                merged[-1].velocity = max(merged[-1].velocity, note.velocity)
                merged_count += 1
            else:
                merged.append(note)

        timeline = sorted(merged, key=lambda n: (n.start, n.pitch))
        keep = []
        for i, note in enumerate(timeline):
            dur = note.end - note.start
            prev_gap = note.start - timeline[i - 1].end if i > 0 else 999.0
            next_gap = timeline[i + 1].start - note.end if i + 1 < len(timeline) else 999.0
            is_quiet_speck = dur < min_len * 0.75 and note.velocity <= quiet_velocity
            is_isolated_speck = dur < min_len * 0.55 and prev_gap > isolated_gap and next_gap > isolated_gap
            if is_quiet_speck or is_isolated_speck:
                removed_count += 1
                continue
            keep.append(note)
        inst.notes = sorted(keep, key=lambda n: (n.start, n.pitch))

    pm.write(str(midi_path))
    total_after = sum(len(inst.notes) for inst in pm.instruments)
    return {
        "before": total_before,
        "after": total_after,
        "removed": removed_count + max(0, total_before - total_after - removed_count),
        "merged": merged_count,
    }


def stabilize_note_midi(midi_path: Path, part: str) -> dict:
    """Make MIDI easier to edit: clear notes with dependable durations.

    This pass intentionally avoids adding new controls. It keeps the output close
    to "note + duration": merge tiny same-pitch gaps, remove invalid fragments,
    and enforce one active note for stems that should normally be monophonic.
    """
    import pretty_midi

    pm = pretty_midi.PrettyMIDI(str(midi_path))
    before = sum(len(inst.notes) for inst in pm.instruments)
    if before == 0:
        return {"before": 0, "after": 0, "removed": 0, "merged": 0, "trimmed": 0, "weak": 0}

    base = part.split("_", 1)[0]
    mono = base in {"bass", "vocals"}
    min_len = {
        "bass": 0.11,
        "vocals": 0.09,
        "guitar": 0.075,
        "piano": 0.085,
        "other": 0.08,
    }.get(base, 0.08)
    merge_gap = 0.055 if mono else 0.035
    same_start_window = 0.035
    merged_count = 0
    trimmed_count = 0
    weak_count = 0

    for inst in pm.instruments:
        if inst.is_drum:
            continue
        clean = []
        for note in sorted(inst.notes, key=lambda n: (n.start, n.pitch, n.end)):
            if note.end <= note.start:
                continue
            if (note.end - note.start) < min_len * 0.6:
                continue
            clean.append(note)

        by_pitch = []
        for note in sorted(clean, key=lambda n: (n.pitch, n.start, n.end)):
            if by_pitch and note.pitch == by_pitch[-1].pitch and note.start - by_pitch[-1].end <= merge_gap:
                by_pitch[-1].end = max(by_pitch[-1].end, note.end)
                by_pitch[-1].velocity = max(by_pitch[-1].velocity, note.velocity)
                merged_count += 1
            else:
                by_pitch.append(note)

        timeline = sorted(by_pitch, key=lambda n: (n.start, n.end, n.pitch))
        strong_velocity = max((n.velocity for n in timeline), default=0)
        weak_velocity = max(24, int(strong_velocity * 0.58))
        filtered = []
        for i, note in enumerate(timeline):
            dur = note.end - note.start
            prev_gap = note.start - timeline[i - 1].end if i > 0 else 999.0
            next_gap = timeline[i + 1].start - note.end if i + 1 < len(timeline) else 999.0
            near_context = prev_gap < 0.12 or next_gap < 0.12
            if dur < min_len * 1.25 and note.velocity <= weak_velocity and near_context:
                weak_count += 1
                continue
            filtered.append(note)
        timeline = filtered

        if mono:
            stable = []
            for note in timeline:
                if not stable:
                    stable.append(note)
                    continue
                prev = stable[-1]
                if note.start < prev.end:
                    prev_score = (prev.end - prev.start, prev.velocity)
                    note_score = (note.end - note.start, note.velocity)
                    if abs(note.start - prev.start) <= same_start_window:
                        if note_score > prev_score:
                            stable[-1] = note
                        continue
                    if note_score > prev_score and (note.start - prev.start) < min_len:
                        stable[-1] = note
                        continue
                    if note.start - prev.start >= min_len:
                        prev.end = min(prev.end, note.start)
                        trimmed_count += 1
                        stable.append(note)
                    elif note.end > prev.end:
                        prev.end = note.end
                        prev.velocity = max(prev.velocity, note.velocity)
                        merged_count += 1
                else:
                    stable.append(note)
            timeline = stable
        else:
            for i, note in enumerate(timeline[:-1]):
                nxt = timeline[i + 1]
                if note.pitch == nxt.pitch and nxt.start < note.end:
                    note.end = max(note.start + min_len, min(note.end, nxt.start))
                    trimmed_count += 1

        inst.notes = [
            n for n in sorted(timeline, key=lambda n: (n.start, n.pitch))
            if (n.end - n.start) >= min_len
        ]

    pm.write(str(midi_path))
    after = sum(len(inst.notes) for inst in pm.instruments)
    return {
        "before": before,
        "after": after,
        "removed": max(0, before - after),
        "merged": merged_count,
        "trimmed": trimmed_count,
        "weak": weak_count,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="audio -> stems -> MIDI per instrument part")
    ap.add_argument("audio", type=Path, help="input audio file (wav/mp3/flac/...)")
    ap.add_argument("-o", "--out", type=Path, default=Path("output"), help="output dir for MIDI")
    ap.add_argument("--stems-dir", type=Path, default=Path("stems"), help="where Demucs writes stems")
    ap.add_argument("-n", "--model", default="htdemucs_6s",
                    help="demucs model (used only in 'single' separation mode)")
    ap.add_argument("--separation", default="cascade",
                    choices=["cascade", "parallel", "parallel-deep", "bestof",
                             "bestof-auto", "single"],
                    help="cascade = 2-stage (ft vocals -> 6s arrangement); "
                         "parallel = each stem from the untouched original "
                         "(ft vocals + 6s instruments); "
                         "parallel-deep = like parallel but a different model "
                         "per instrument (more passes, more time); "
                         "bestof = assemble each stem from its best-measured source; "
                         "single = one demucs pass with --model")
    ap.add_argument("--bestof-plan", default=None,
                    help="override bestof sources, e.g. 'piano:6s,bass:htdemucs' "
                         "(sources: ft, 6s, htdemucs, cascade)")
    ap.add_argument("-d", "--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--segment", type=int, default=7, help="demucs segment length (lower = less VRAM)")
    ap.add_argument("--include-drums", action="store_true",
                    help="transcribe the drum stem via onset detection (GM percussion)")
    ap.add_argument("--drum-sensitivity", type=float, default=0.6,
                    help="drum onset sensitivity 0..1 (higher = more hits)")
    ap.add_argument("--bpm", type=float, default=None,
                    help="known tempo for drum grid; omit for auto-detect")
    ap.add_argument("--drum-grid-fill", action="store_true",
                    help="fill missing cymbals on the beat grid in active sections")
    ap.add_argument("--cymbal-gate", type=float, default=1.05,
                    help="cymbal vs local-background gate (higher = fewer, cleaner)")
    ap.add_argument("--detect-toms", action="store_true",
                    help="try to split tom-toms out of the snare class (approximate)")
    ap.add_argument("--grid-offset-ms", type=float, default=0.0,
                    help="manual phase nudge for the drum beat grid (ms)")
    ap.add_argument("--save-audio", dest="save_audio", action="store_true", default=True,
                    help="copy the separated stem WAVs next to the MIDI (default on)")
    ap.add_argument("--no-save-audio", dest="save_audio", action="store_false",
                    help="do not copy stem WAVs into the output folder")
    # Global transcription overrides. When set, they replace the per-part
    # profile value for that knob across ALL stems (frequency bounds stay
    # per-part unless explicitly overridden).
    ap.add_argument("--onset", type=float, default=None,
                    help="override onset threshold for all parts (0..1, higher=fewer notes)")
    ap.add_argument("--frame", type=float, default=None,
                    help="override frame threshold for all parts (0..1, higher=fewer notes)")
    ap.add_argument("--min-note-ms", type=float, default=None,
                    help="override minimum note length in ms for all parts")
    ap.add_argument("--adaptive-midi", dest="adaptive_midi", action="store_true", default=True,
                    help="analyze each stem and adapt MIDI thresholds/engine choice (default on)")
    ap.add_argument("--no-adaptive-midi", dest="adaptive_midi", action="store_false",
                    help="use static transcription profiles")
    ap.add_argument("--midi-preprocess", dest="midi_preprocess", action="store_true", default=True,
                    help="write a cleaned helper WAV before melodic transcription (default on)")
    ap.add_argument("--no-midi-preprocess", dest="midi_preprocess", action="store_false",
                    help="send raw Demucs stems directly to the MIDI engine")
    ap.add_argument("--smart-clean", dest="smart_clean", action="store_true", default=True,
                    help="merge gaps and remove quiet/isolated MIDI specks after transcription")
    ap.add_argument("--no-smart-clean", dest="smart_clean", action="store_false",
                    help="disable the second-pass smart MIDI cleanup")
    ap.add_argument("--freq-bounds", default="per-part", choices=["per-part", "off"],
                    help="per-part = use instrument frequency bands (default); "
                         "off = no frequency filtering")
    # Key-aware filtering.
    ap.add_argument("--key-filter", default="off", choices=["off", "auto", "manual"],
                    help="off = no key filtering; auto = detect key and drop "
                         "out-of-scale notes; manual = use --key")
    ap.add_argument("--key", default=None,
                    help="key for manual mode, e.g. 'G:min', 'Bb:maj', 'Am'")
    ap.add_argument("--clean-octaves", dest="clean_octaves", action="store_true", default=True,
                    help="remove octave-ghost notes from melodic MIDI (default on)")
    ap.add_argument("--no-clean-octaves", dest="clean_octaves", action="store_false",
                    help="keep octave doublings")
    ap.add_argument("--max-polyphony", type=int, default=0,
                    help="cap simultaneous notes per melodic part (0 = no cap)")
    # Stereo splitting (e.g. two panned guitars -> two parts).
    ap.add_argument("--split-stem", default="",
                    help="comma-separated stems to split by stereo, e.g. 'guitar'")
    ap.add_argument("--split-method", default="pan", choices=["pan", "lr", "midside"],
                    help="stereo split method")
    ap.add_argument("--split-naming", default="LR", choices=["LR", "solo-rhythm", "hybrid"],
                    help="LR = _L/_R; solo-rhythm = _lead/_rhythm by heuristic; "
                         "hybrid = _L/_R plus a guessed role suffix")
    ap.add_argument("--mono-stems", default="",
                    help="comma-separated stems to transcribe MONOPHONICALLY via "
                         "pYIN. Best for CLEAN single-line parts (a clean lead, a "
                         "synth line). On distorted/dense material basic-pitch wins, "
                         "so this is off by default — enable per-track when it fits.")
    ap.add_argument("--piano-engine", default="basic-pitch",
                    choices=["basic-pitch", "onsets-frames"],
                    help="transcription engine for the piano stem. onsets-frames "
                         "is a piano-specific model (more accurate on dense piano).")
    args = ap.parse_args()

    if not args.audio.is_file():
        print(f"[error] no such file: {args.audio}")
        return 1

    ensure_ffmpeg_on_path()
    if not find_ffmpeg():
        print("[warn] ffmpeg not found; mp3/m4a decoding may fail (wav/flac still OK)")

    wanted = set(MELODIC_STEMS)
    if args.include_drums:
        wanted.add("drums")
    split_stems = {s.strip() for s in args.split_stem.split(",") if s.strip()}
    mono_stems = {s.strip() for s in args.mono_stems.split(",") if s.strip()}

    device = pick_device(args.device)
    if args.separation == "cascade":
        stem_dir = separate_cascade(args.audio, args.stems_dir, device, args.segment)
    elif args.separation == "parallel":
        stem_dir = separate_parallel(args.audio, args.stems_dir, device, args.segment, wanted)
    elif args.separation == "parallel-deep":
        stem_dir = separate_parallel(args.audio, args.stems_dir, device, args.segment, wanted,
                                     plan=DEEP_PARALLEL_PLAN, tag="parallel-deep")
    elif args.separation == "bestof":
        override = {}
        if args.bestof_plan:
            for pair in args.bestof_plan.split(","):
                if ":" in pair:
                    p, s = pair.split(":", 1)
                    p, s = p.strip(), s.strip()
                    if s in BESTOF_SOURCES:
                        override[p] = s
        stem_dir = separate_bestof(args.audio, args.stems_dir, device, args.segment,
                                   wanted, plan=override)
    elif args.separation == "bestof-auto":
        stem_dir = separate_bestof_auto(args.audio, args.stems_dir, device, args.segment, wanted)
    else:
        stem_dir = run_demucs(args.audio, args.stems_dir, args.model, device, args.segment)

    midi_root = args.out / args.audio.stem
    audio_root = midi_root / "audio"
    work_root = midi_root / "_midi_work"
    if args.save_audio:
        audio_root.mkdir(parents=True, exist_ok=True)

    produced = []
    for wav in sorted(stem_dir.glob("*.wav")):
        name = wav.stem
        # Copy the intermediate audio next to the MIDI so it's easy to audition
        # exactly what the transcriber heard for each part.
        if args.save_audio:
            shutil.copy2(wav, audio_root / wav.name)
        if name not in wanted:
            print(f"[skip] {name}")
            continue

        # Drums get onset-based percussion transcription, not pitch transcription.
        if name == "drums":
            from drum_transcribe import transcribe_drums

            print("[drums] onset detection (superflux cymbals + band kick/snare) ...")
            midi_path = midi_root / "drums.mid"
            res = transcribe_drums(wav, midi_path, sensitivity=args.drum_sensitivity,
                                   bpm=args.bpm, grid_fill=args.drum_grid_fill,
                                   cymbal_gate=args.cymbal_gate, detect_toms=args.detect_toms,
                                   grid_offset_ms=args.grid_offset_ms)
            summary = ", ".join(f"{k}={v}" for k, v in res.items()
                                if not k.startswith("_"))
            print(f"           -> tempo ~{res.get('_tempo','?')} BPM, "
                  f"{res.get('_total',0)} hits ({summary})")
            produced.append(name)
            continue

        # Build this part's profile: per-part defaults, adaptive tuning, then
        # explicit user overrides. User knobs always win.
        metrics = analyze_stem(wav) if args.adaptive_midi else {}
        profile = dict(PART_PROFILES.get(name, DEFAULT_PROFILE))
        if args.adaptive_midi:
            profile = choose_adaptive_profile(name, profile, metrics)
        if args.freq_bounds == "off":
            profile["min_freq"] = None
            profile["max_freq"] = None
        if args.onset is not None:
            profile["onset"] = args.onset
        if args.frame is not None:
            profile["frame"] = args.frame
        if args.min_note_ms is not None:
            profile["min_len_ms"] = args.min_note_ms

        fb = "off" if profile["min_freq"] is None else f"{profile['min_freq']}-{profile['max_freq']}Hz"
        if args.adaptive_midi:
            print(f"[analyze] {name:7s} rms={metrics.get('rms', 0):.4f}, "
                  f"flat={metrics.get('flatness', 0):.3f}, "
                  f"onsets/s={metrics.get('onset_density', 0):.2f}, "
                  f"harm={metrics.get('harmonic_ratio', 0):.2f}")

        # Pick the transcription engine for this stem:
        #   mono (--mono-stems) -> pYIN single-line
        #   piano + onsets-frames -> piano-specific O&F model
        #   else -> polyphonic basic-pitch
        use_mono = name in mono_stems
        if args.adaptive_midi:
            engine, eng_name = choose_adaptive_engine(name, use_mono, args.piano_engine, metrics)
        elif use_mono:
            engine, eng_name = transcribe_stem_mono, "pYIN-mono"
        elif name == "piano" and args.piano_engine == "onsets-frames":
            engine, eng_name = transcribe_stem_piano_of, "onsets-frames"
        else:
            engine, eng_name = transcribe_stem, "basic-pitch"

        tx_wav = wav
        if args.midi_preprocess and args.adaptive_midi:
            tx_wav = preprocess_stem_for_midi(wav, work_root / f"{name}.prep.wav",
                                              name, profile, metrics)

        # Stereo split: this stem becomes two parts (e.g. left/right guitar).
        if name in split_stems:
            from stereo_split import split_stereo

            a_wav = audio_root / f"{name}_A.wav" if args.save_audio else midi_root / f"_{name}_A.wav"
            b_wav = audio_root / f"{name}_B.wav" if args.save_audio else midi_root / f"_{name}_B.wav"
            a_wav.parent.mkdir(parents=True, exist_ok=True)
            sinfo = split_stereo(wav, a_wav, b_wav, method=args.split_method)
            if not sinfo.get("stereo"):
                print(f"[split] {name} is mono — cannot split, transcribing whole")
                n_notes = engine(tx_wav, midi_root / f"{name}.mid", profile)
                print(f"           -> {n_notes} notes")
                produced.append(name)
                continue

            print(f"[split] {name} -> two parts via {args.split_method} ({eng_name})")
            tmp_a = midi_root / f"{name}_A.mid"
            tmp_b = midi_root / f"{name}_B.mid"
            a_profile = profile
            b_profile = profile
            a_tx = a_wav
            b_tx = b_wav
            if args.midi_preprocess and args.adaptive_midi:
                a_metrics = analyze_stem(a_wav)
                b_metrics = analyze_stem(b_wav)
                a_profile = choose_adaptive_profile(name, profile, a_metrics)
                b_profile = choose_adaptive_profile(name, profile, b_metrics)
                a_tx = preprocess_stem_for_midi(a_wav, work_root / f"{name}_A.prep.wav",
                                                name, a_profile, a_metrics)
                b_tx = preprocess_stem_for_midi(b_wav, work_root / f"{name}_B.prep.wav",
                                                name, b_profile, b_metrics)
            na = engine(a_tx, tmp_a, a_profile)
            nb = engine(b_tx, tmp_b, b_profile)

            # Decide names by the chosen naming scheme.
            if args.split_naming == "LR":
                name_a, name_b = f"{name}_L", f"{name}_R"
            elif args.split_naming == "solo-rhythm":
                ra, rb = classify_lead_rhythm(tmp_a, tmp_b)
                name_a, name_b = f"{name}_{ra}", f"{name}_{rb}"
            else:  # hybrid: L/R plus guessed role
                ra, rb = classify_lead_rhythm(tmp_a, tmp_b)
                name_a, name_b = f"{name}_L_{ra}", f"{name}_R_{rb}"

            (midi_root / f"{name_a}.mid").unlink(missing_ok=True)
            (midi_root / f"{name_b}.mid").unlink(missing_ok=True)
            tmp_a.rename(midi_root / f"{name_a}.mid")
            tmp_b.rename(midi_root / f"{name_b}.mid")
            print(f"           -> {name_a}: {na} notes, {name_b}: {nb} notes")
            produced.extend([name_a, name_b])
            continue

        print(f"[{eng_name}] transcribing {name} "
              f"(onset={profile['onset']}, frame={profile['frame']}, "
              f"min_len={profile['min_len_ms']}ms, band={fb}) ...")
        midi_path = midi_root / f"{name}.mid"
        n_notes = engine(tx_wav, midi_path, profile)
        print(f"           -> {n_notes} notes")
        produced.append(name)

    # ---- MIDI cleanup (octave ghosts / polyphony cap / smart speck filter) ----
    if (args.clean_octaves or args.max_polyphony or args.smart_clean) and produced:
        print("\n[clean] removing transcription clutter from melodic parts")
        for p in produced:
            if p == "drums":
                continue
            mp = midi_root / f"{p}.mid"
            if not mp.is_file():
                continue
            if args.clean_octaves or args.max_polyphony:
                kept, removed = clean_midi(mp, drop_octave_ghosts=args.clean_octaves,
                                           max_polyphony=args.max_polyphony)
                if removed:
                    print(f"  {p:8s} removed {removed}, kept {kept}")
            if args.smart_clean:
                base_part = p.split("_", 1)[0]
                stats = clean_midi_smart(
                    mp,
                    base_part,
                    PART_PROFILES.get(base_part, DEFAULT_PROFILE),
                )
                if stats["removed"] or stats["merged"]:
                    print(f"  {p:8s} smart removed {stats['removed']}, "
                          f"merged {stats['merged']}, kept {stats['after']}")
                stable = stabilize_note_midi(mp, base_part)
                if stable["removed"] or stable["merged"] or stable["trimmed"] or stable["weak"]:
                    print(f"  {p:8s} stable removed {stable['removed']}, "
                          f"merged {stable['merged']}, weak {stable['weak']}, "
                          f"trimmed {stable['trimmed']}, "
                          f"kept {stable['after']}")


    # ---- Key-aware filtering (post-process MIDI) ----
    if args.key_filter != "off" and produced:
        melodic_midis = [midi_root / f"{p}.mid" for p in produced if p != "drums"]
        key_info = None
        if args.key_filter == "manual":
            parsed = parse_key(args.key or "")
            if parsed:
                key_info = (f"{_NOTE_NAMES[parsed[0]]}:{parsed[1]}", parsed[0], parsed[1], None)
            else:
                print(f"[key] could not parse --key '{args.key}', skipping key filter")
        else:  # auto
            key_info = estimate_key(melodic_midis)

        if key_info:
            label, tonic, mode, conf = key_info
            conf_s = f" (confidence {conf:.2f})" if conf is not None else ""
            print(f"\n[key] filtering to {label}{conf_s}")
            for p in produced:
                if p == "drums":
                    continue  # percussion is not tonal
                kept, removed = filter_midi_in_key(midi_root / f"{p}.mid", tonic, mode)
                if removed < 0:
                    print(f"  {p:8s} kept all {kept} (would lose {-removed}, "
                          f">{int(KEY_FILTER_SAFETY*100)}% -> skipped for safety)")
                else:
                    print(f"  {p:8s} kept {kept}, removed {removed}")

    print("\n=== done ===")
    print(f"stems: {stem_dir}")
    print(f"midi:  {midi_root}")
    if args.save_audio:
        print(f"audio: {audio_root}")
    print(f"parts: {', '.join(produced) if produced else '(none)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
