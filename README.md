# M-A Splitter

Desktop audio stem splitter and draft MIDI extraction tool.

M-A means MIDI-AUDIO. The tool separates an audio file into instrument stems and
exports draft MIDI parts for further editing in a DAW.

Maintained by Maslodium.

## Features

- Stem separation through Demucs.
- Draft MIDI export for bass, vocals, guitar, piano and other stems.
- Optional WAV stem export beside the MIDI files.
- Adaptive MIDI mode with stem analysis, helper-WAV preprocessing and smart
  cleanup.
- Drum detector for GM percussion MIDI.
- Stereo field split helper for hard-panned material.
- Key-aware MIDI filtering and octave-ghost cleanup.

## Requirements

- Windows 10/11.
- macOS/Linux builds are experimental.
- Python 3.12 recommended.
- NVIDIA GPU is optional, but strongly recommended for Demucs.

## Run From Source

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe install.py
.\.venv\Scripts\pythonw.exe gui.py
```

`install.py` installs the pinned runtime stack and then installs Basic Pitch
without its legacy TensorFlow dependency chain.

## CLI

```powershell
python pipeline.py song.wav -o output
python pipeline.py song.wav --no-adaptive-midi
python pipeline.py song.wav --no-midi-preprocess
python pipeline.py song.wav --no-smart-clean
```

## Possible Improvements

- Better note grouping for guitar bends and vocal slides.
- Confidence metadata for MIDI events.
- Per-stem denoise before MIDI transcription.
- Local MIDI model experiments for cases where frequency analysis and Basic
  Pitch disagree.
- Batch processing for folders.

## Notes

MIDI extraction is a draft assistant, not a finished score. Timing, polyphony,
note choice and drums may still need manual editing after export.

Check upstream model/code licenses before redistributing pretrained weights or
commercial bundles.

---

# M-A Splitter

Настольная утилита для разделения аудио на стемы и чернового извлечения MIDI.

M-A означает MIDI-AUDIO. Программа разделяет аудиофайл на инструментальные
стемы и экспортирует черновые MIDI-партии для дальнейшей правки в DAW.

Поддерживает Maslodium.

## Возможности

- Разделение стемов через Demucs.
- Черновой MIDI-экспорт для баса, вокала, гитары, пианино и других стемов.
- Опциональное сохранение WAV-стемов рядом с MIDI.
- Adaptive MIDI mode: анализ стема, helper-WAV preprocessing и smart cleanup.
- Детектор барабанов для GM percussion MIDI.
- Stereo field split для сильно разведённого по панораме материала.
- Фильтрация MIDI по тональности и очистка октавных призраков.

## Требования

- Windows 10/11.
- Рекомендуется Python 3.12.
- NVIDIA GPU не обязателен, но сильно ускоряет Demucs.

## Запуск из исходников

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe install.py
.\.venv\Scripts\pythonw.exe gui.py
```

## CLI

```powershell
python pipeline.py song.wav -o output
python pipeline.py song.wav --no-adaptive-midi
python pipeline.py song.wav --no-midi-preprocess
python pipeline.py song.wav --no-smart-clean
```

## Возможные Доработки

- Более точное склеивание нот для гитарных бендов и вокальных глайдов.
- Confidence metadata для MIDI-событий.
- Denoise отдельных стемов перед MIDI-транскрипцией.
- Эксперименты с локальной MIDI-моделью для случаев, где частотный анализ и
  Basic Pitch расходятся.
- Пакетная обработка папок.

## Примечания

MIDI-экспорт нужно воспринимать как черновой помощник, а не как готовую
партитуру. Тайминг, полифония, выбор нот и барабаны могут требовать ручной
правки после экспорта.

Перед распространением весов моделей или коммерческой сборкой нужно отдельно
проверить лицензии upstream-проектов.
