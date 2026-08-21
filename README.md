# M-A Splitter

Offline audio splitter and MIDI transcription tool for Windows.

The current pipeline is:

```text
audio file -> instrument stems -> per-part MIDI
```

It combines stem separation with MIDI transcription paths for melodic parts, piano and drums.

## Status

This is an early work-in-progress build. Stem separation is usable, but MIDI recognition is still experimental: timing, note choice, polyphony and drum detection may need manual cleanup after export. Contributions, experiments and test tracks are welcome.

## Quick Start

The repository includes the current Windows installer:

```text
Install M-A Splitter.exe
```

After installation, run `M-A Splitter.bat` or start the GUI from the installed project folder.

## Run From Source

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\pythonw.exe gui.py
```

Torch/torchaudio may need a hardware-specific install command for CUDA or CPU builds. The installer handles that automatically; source setup may need manual adjustment.

## Project Layout

- `gui.py` - Tkinter desktop interface.
- `pipeline.py` - audio separation and MIDI export pipeline.
- `drum_transcribe.py` - onset-based drum transcription.
- `stereo_split.py` - stereo panorama split helper.
- `requirements-lock.txt` - pinned Python dependency set.
- `Install M-A Splitter.exe` - current Windows installer build.

## Notes

- Settings are saved in `gui_settings.json` when the app runs.
- Generated `input/`, `output/`, model caches and virtual environments are intentionally ignored.
- MIDI output should be treated as a starting point for editing, not as a finished score.

