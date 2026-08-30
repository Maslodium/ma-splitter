# M-A Splitter

Offline audio splitter and MIDI transcription tool for Windows.

The current pipeline is:

```text
audio file -> instrument stems -> per-part MIDI
```

It combines stem separation with MIDI transcription paths for melodic parts, piano and drums.

Maintained by Maslodium.

## Status

This is an early work-in-progress build. Stem separation is usable, but MIDI recognition is still experimental: timing, note choice, polyphony and drum detection may need manual cleanup after export.

The project is published as a practical starting point for musicians, developers and audio experimenters who want to test it, improve the recognition logic, or adapt the pipeline to their own material.

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

---

# M-A Splitter

Оффлайн-инструмент для разделения аудио и черновой MIDI-транскрипции под Windows.

Текущий пайплайн:

```text
аудиофайл -> инструментальные дорожки -> MIDI по партиям
```

Он объединяет разделение на stems и отдельные пути MIDI-транскрипции для мелодических партий, пианино и ударных.

Поддерживает Maslodium.

## Статус

Это ранняя рабочая версия. Разделение дорожек уже можно использовать, но распознавание MIDI пока экспериментальное: тайминг, выбор нот, полифония и определение ударных могут требовать ручной чистки после экспорта.

Проект опубликован как практическая стартовая точка для музыкантов, разработчиков и аудио-экспериментаторов, которые хотят протестировать инструмент, улучшить распознавание или адаптировать пайплайн под свой материал.

## Быстрый старт

В репозитории есть текущий Windows-инсталлятор:

```text
Install M-A Splitter.exe
```

После установки запустите `M-A Splitter.bat` или GUI из установленной папки проекта.

## Запуск из исходников

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\pythonw.exe gui.py
```

Для Torch/torchaudio может понадобиться команда установки под конкретное железо: CUDA или CPU. Инсталлятор делает это автоматически, а при запуске из исходников иногда нужна ручная настройка.

## Структура проекта

- `gui.py` - desktop-интерфейс на Tkinter.
- `pipeline.py` - пайплайн разделения аудио и экспорта MIDI.
- `drum_transcribe.py` - транскрипция ударных по onset-событиям.
- `stereo_split.py` - вспомогательный split по стереопанораме.
- `requirements-lock.txt` - зафиксированный набор Python-зависимостей.
- `Install M-A Splitter.exe` - текущая сборка Windows-инсталлятора.

## Заметки

- Настройки сохраняются в `gui_settings.json` при запуске приложения.
- Сгенерированные `input/`, `output/`, кэши моделей и виртуальные окружения специально игнорируются.
- MIDI-результат стоит воспринимать как черновик для дальнейшего редактирования, а не как готовую партитуру.
