# M-A Splitter

Desktop audio stem splitter and draft MIDI extraction tool.

M-A means MIDI-AUDIO. The tool separates an audio file into instrument stems and
exports draft MIDI parts for further editing in a DAW.

The interface was rebuilt as a dark cyber-metal rack: custom title bar,
brushed-metal section rails, dark work panels, non-system control colors,
Oxanium display font and cyan/magenta accents.

Maintained by Maslodium.

## Features

- Stem separation through Demucs.
- Draft MIDI export for bass, vocals, guitar, piano and other stems.
- Optional WAV stem export beside the MIDI files.
- Adaptive MIDI mode with stem analysis, helper-WAV preprocessing and smart
  cleanup.
- Drum detector for GM percussion MIDI.
- Stereo field split helper for hard-panned material.

## Requirements

- Windows 10/11.
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

## Notes

MIDI extraction is a draft assistant, not a finished score. Timing, polyphony,
note choice and drums may still need manual editing after export.

Useful next improvements are better guitar/vocal bends, confidence metadata,
per-stem denoise before MIDI and batch processing.

Check upstream model/code licenses before redistributing pretrained weights or
commercial bundles.

---

# M-A Splitter

Настольная утилита для разделения аудио на стемы и чернового извлечения MIDI.

M-A означает MIDI-AUDIO. Программа разделяет аудиофайл на инструментальные
стемы и экспортирует черновые MIDI-партии для дальнейшей правки в DAW.

Интерфейс переработан в тёмный cyber-metal rack: собственная верхняя панель
окна, металлические полосы разделов, тёмные рабочие панели, не системные цвета
контролов, шрифт Oxanium и cyan/magenta акценты.

Поддерживает Maslodium.

## Возможности

- Разделение стемов через Demucs.
- Черновой MIDI-экспорт для баса, вокала, гитары, пианино и других стемов.
- Опциональное сохранение WAV-стемов рядом с MIDI.
- Adaptive MIDI mode: анализ стема, helper-WAV preprocessing и smart cleanup.
- Детектор барабанов для GM percussion MIDI.
- Stereo field split для сильно разведённого по панораме материала.

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

## Примечания

MIDI-экспорт нужно воспринимать как черновой помощник, а не как готовую
партитуру. Тайминг, полифония, выбор нот и барабаны могут требовать ручной
правки после экспорта.

Полезные следующие улучшения: гитарные/вокальные бенды, confidence metadata,
denoise стемов перед MIDI и пакетная обработка.
