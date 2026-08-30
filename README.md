# M-A Splitter

Author: Maslodium

MIDI-AUDIO Splitter is a local desktop tool for offline audio stem separation
and draft MIDI extraction.

```text
audio file -> instrument stems -> adaptive MIDI transcription
```

The interface was rebuilt as a dark cyber-metal rack: custom title bar,
brushed-metal section rails, non-system control colors, dark work panels,
Oxanium display font and cyan/magenta accent lines.

## Features

- Separates audio into stems with Demucs.
- Exports per-part draft MIDI for bass, vocals, guitar, piano and other stems.
- Optional WAV stem export beside the MIDI files.
- Adaptive MIDI path:
  - analyzes loudness, spectral flatness, onset density and harmonic/percussive
    balance;
  - prepares private helper WAVs with bandpass, harmonic emphasis and adaptive
    noise gate;
  - can auto-pick pYIN, Basic Pitch or piano transcription paths;
  - applies smart MIDI cleanup for tiny gaps, octave ghosts and isolated specks.
- Drum detector can export GM percussion MIDI.
- Stereo field split helper can separate hard-panned material before MIDI
  extraction.

## Run From Source

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe install.py
.\.venv\Scripts\pythonw.exe gui.py
```

`install.py` installs the pinned runtime stack and then installs Basic Pitch
without its legacy TensorFlow dependency chain. Torch and torchaudio are pinned
as a matching `2.6.0` pair in `requirements-lock.txt`.

## MIDI Notes

MIDI extraction is intentionally treated as a draft assistant, not a finished
score. The adaptive pass improves thresholds and cleanup, but timing,
polyphony, note choice and drums may still need manual editing in a DAW.

For A/B comparisons:

```powershell
python pipeline.py song.wav --no-adaptive-midi
python pipeline.py song.wav --no-midi-preprocess
python pipeline.py song.wav --no-smart-clean
```

## Roadmap

- Better note grouping for guitar and vocal bends.
- Confidence-based MIDI event coloring/export metadata.
- Per-stem noise reduction before transcription.
- Local model experiments for MIDI transcription where frequency heuristics and
  Basic Pitch disagree.
- Batch processing and project presets.

## Licenses

- Demucs is used for source separation.
- Oxanium is bundled under the SIL Open Font License 1.1.

Check upstream model/code licenses before redistributing pretrained weights or
commercial bundles.

## Русское Описание

Автор: Maslodium

**M-A Splitter** расшифровывается как **MIDI-AUDIO Splitter**. Это локальная
программа для разделения аудио на стемы и чернового извлечения MIDI.

Интерфейс переработан в стиле тёмного cyber-metal rack: собственная верхняя
панель окна, металлические полосы разделов, не системные цвета кнопок и рамок,
тёмные рабочие области, шрифт Oxanium и киберпанк-акценты.

## Возможности

- Разделение аудио на стемы через Demucs.
- Черновой MIDI-экспорт по партиям: бас, вокал, гитара, пианино и другие
  дорожки.
- Сохранение WAV-стемов рядом с MIDI.
- Адаптивный MIDI-режим:
  - анализирует громкость, шумность, плотность атак и harmonic/percussive
    баланс;
  - готовит helper-WAV с фильтрацией, harmonic emphasis и noise gate;
  - выбирает pYIN, Basic Pitch или piano path по характеру стема;
  - чистит MIDI от мелкого мусора, октавных призраков и коротких дыр.
- Детектор барабанов может экспортировать GM percussion MIDI.
- Stereo field split помогает разделять сильно разведённые по панораме партии.

## Что Добавить Дальше

- Более умное склеивание нот для гитары и вокальных глайдов.
- Экспорт уверенности распознавания нот для ручной правки в DAW.
- Предварительный denoise отдельных стемов перед MIDI.
- Локальные модели MIDI-транскрипции для спорных мест, где частотный анализ и
  Basic Pitch расходятся.
- Пакетная обработка и пресеты проектов.
