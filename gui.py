"""
M-A Splitter GUI: pick an audio file, pick an output folder, run the
audio -> stems -> MIDI pipeline, watch the log live, then audition the
produced stems and MIDI right from the window.

Pure Tkinter (no extra deps). Runs pipeline.py as a subprocess using the same
venv Python, so the UI stays responsive and output streams into the window.

Layout: progressive disclosure — option blocks appear only when their mode /
checkbox is active, keeping the window compact. Log + results live in one panel
on the right.
"""

import json
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

HERE = Path(__file__).resolve().parent
PIPELINE = HERE / "pipeline.py"
SETTINGS = HERE / "gui_settings.json"

AUDIO_TYPES = [
    ("Audio", "*.wav *.mp3 *.flac *.m4a *.ogg *.aac *.wma *.aiff"),
    ("WAV", "*.wav"),
    ("MP3", "*.mp3"),
    ("FLAC", "*.flac"),
    ("All files", "*.*"),
]

DEMUCS_MODELS = ["htdemucs_6s", "htdemucs", "htdemucs_ft", "mdx_extra", "mdx_extra_q"]


class Tooltip:
    """Lightweight hover tooltip for any Tk widget."""

    def __init__(self, widget: tk.Widget, text: str, delay: int = 450) -> None:
        self.widget = widget
        self.text = text
        self.delay = delay
        self._after: str | None = None
        self._tip: tk.Toplevel | None = None
        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")

    def _schedule(self, _e: object = None) -> None:
        self._cancel()
        self._after = self.widget.after(self.delay, self._show)

    def _cancel(self) -> None:
        if self._after:
            self.widget.after_cancel(self._after)
            self._after = None

    def _show(self) -> None:
        if self._tip or not self.text:
            return
        x = self.widget.winfo_rootx() + 16
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 6
        self._tip = tk.Toplevel(self.widget)
        self._tip.wm_overrideredirect(True)
        self._tip.wm_geometry(f"+{x}+{y}")
        tk.Label(self._tip, text=self.text, justify="left", bg="#fdf6d8", fg="#222",
                 relief="solid", borderwidth=1, font=("Segoe UI", 9),
                 wraplength=340, padx=7, pady=4).pack()

    def _hide(self, _e: object = None) -> None:
        self._cancel()
        if self._tip:
            self._tip.destroy()
            self._tip = None


def tip(widget: tk.Widget, text: str) -> tk.Widget:
    Tooltip(widget, text)
    return widget


class App:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        root.title("M-A Splitter v0.2 — аудио → MIDI")
        root.geometry("960x600")
        root.minsize(880, 560)

        self.proc: subprocess.Popen | None = None
        self.q: "queue.Queue[str]" = queue.Queue()
        self.last_result_dir: Path | None = None

        cfg = self._load_settings()
        self.in_path = tk.StringVar(value=cfg.get("in_path", ""))
        self.out_path = tk.StringVar(value=cfg.get("out_path", str(HERE / "output")))
        self.device = tk.StringVar(value=cfg.get("device", "auto"))
        self.separation = tk.StringVar(value=cfg.get("separation", "cascade"))
        self.model = tk.StringVar(value=cfg.get("model", "htdemucs_6s"))
        self.segment = tk.IntVar(value=cfg.get("segment", 7))
        self.include_drums = tk.BooleanVar(value=cfg.get("include_drums", False))
        self.save_audio = tk.BooleanVar(value=cfg.get("save_audio", True))
        self.tweak_thresholds = tk.BooleanVar(value=cfg.get("tweak_thresholds", False))
        self.onset = tk.DoubleVar(value=cfg.get("onset", 0.5))
        self.frame = tk.DoubleVar(value=cfg.get("frame", 0.3))
        self.min_note_ms = tk.IntVar(value=cfg.get("min_note_ms", 80))
        self.freq_bounds = tk.StringVar(value=cfg.get("freq_bounds", "per-part"))
        self.key_filter = tk.StringVar(value=cfg.get("key_filter", "off"))
        self.key = tk.StringVar(value=cfg.get("key", "G:min"))
        self.piano_engine = tk.StringVar(value=cfg.get("piano_engine", "basic-pitch"))
        self.clean_octaves = tk.BooleanVar(value=cfg.get("clean_octaves", True))
        self.max_polyphony = tk.IntVar(value=cfg.get("max_polyphony", 0))
        saved_mono = set(cfg.get("mono_stems", []))
        self.mono_stems = {s: tk.BooleanVar(value=s in saved_mono)
                           for s in ("bass", "vocals", "other", "guitar", "piano")}
        self.drum_sensitivity = tk.DoubleVar(value=cfg.get("drum_sensitivity", 0.6))
        self.bpm = tk.StringVar(value=cfg.get("bpm", ""))
        self.drum_grid_fill = tk.BooleanVar(value=cfg.get("drum_grid_fill", False))
        self.cymbal_gate = tk.DoubleVar(value=cfg.get("cymbal_gate", 1.05))
        self.detect_toms = tk.BooleanVar(value=cfg.get("detect_toms", False))
        self.grid_offset_ms = tk.IntVar(value=cfg.get("grid_offset_ms", 0))
        default_bestof = {"vocals": "ft", "bass": "htdemucs", "drums": "htdemucs",
                          "guitar": "6s", "other": "6s", "piano": "cascade"}
        saved = cfg.get("bestof_plan", {})
        self.bestof_plan = {k: tk.StringVar(value=saved.get(k, v))
                            for k, v in default_bestof.items()}
        saved_split = set(cfg.get("split_stems", []))
        self.split_stems = {s: tk.BooleanVar(value=s in saved_split)
                            for s in ("guitar", "other", "vocals", "piano")}
        self.split_method = tk.StringVar(value=cfg.get("split_method", "pan"))
        self.split_naming = tk.StringVar(value=cfg.get("split_naming", "LR"))
        self.split_enable = tk.BooleanVar(value=bool(saved_split))

        self._build()
        self.root.after(100, self._drain_log)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ---------- settings ----------
    def _load_settings(self) -> dict:
        try:
            return json.loads(SETTINGS.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _save_settings(self) -> None:
        data = {
            "in_path": self.in_path.get(), "out_path": self.out_path.get(),
            "device": self.device.get(), "separation": self.separation.get(),
            "model": self.model.get(), "segment": int(self.segment.get()),
            "include_drums": bool(self.include_drums.get()),
            "save_audio": bool(self.save_audio.get()),
            "tweak_thresholds": bool(self.tweak_thresholds.get()),
            "onset": round(float(self.onset.get()), 2),
            "frame": round(float(self.frame.get()), 2),
            "min_note_ms": int(self.min_note_ms.get()),
            "freq_bounds": self.freq_bounds.get(),
            "key_filter": self.key_filter.get(), "key": self.key.get(),
            "piano_engine": self.piano_engine.get(),
            "clean_octaves": bool(self.clean_octaves.get()),
            "max_polyphony": int(self.max_polyphony.get()),
            "mono_stems": [s for s, v in self.mono_stems.items() if v.get()],
            "drum_sensitivity": round(float(self.drum_sensitivity.get()), 2),
            "bpm": self.bpm.get(),
            "drum_grid_fill": bool(self.drum_grid_fill.get()),
            "cymbal_gate": round(float(self.cymbal_gate.get()), 2),
            "detect_toms": bool(self.detect_toms.get()),
            "grid_offset_ms": int(self.grid_offset_ms.get()),
            "bestof_plan": {k: v.get() for k, v in self.bestof_plan.items()},
            "split_stems": [s for s, v in self.split_stems.items() if v.get()],
            "split_method": self.split_method.get(),
            "split_naming": self.split_naming.get(),
        }
        try:
            SETTINGS.parent.mkdir(parents=True, exist_ok=True)
            SETTINGS.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

    # ---------- UI ----------
    def _build(self) -> None:
        pad = {"padx": 8, "pady": 4}

        # ===== Left column: controls in a scroll-free compact stack =====
        left = ttk.Frame(self.root)
        left.pack(side="left", fill="both", expand=False)

        # File row
        frm = ttk.Frame(left)
        frm.pack(fill="x", **pad)
        frm.columnconfigure(1, weight=1)
        ttk.Label(frm, text="Аудиофайл:").grid(row=0, column=0, sticky="w")
        e_in = ttk.Entry(frm, textvariable=self.in_path, width=44)
        e_in.grid(row=0, column=1, sticky="ew", padx=4)
        tip(e_in, "Исходная фонограмма для разбора (wav, mp3, flac, m4a и др.).")
        tip(ttk.Button(frm, text="Обзор…", command=self._pick_input),
            "Выбрать аудиофайл.").grid(row=0, column=2)
        ttk.Label(frm, text="Папка вывода:").grid(row=1, column=0, sticky="w")
        e_out = ttk.Entry(frm, textvariable=self.out_path, width=44)
        e_out.grid(row=1, column=1, sticky="ew", padx=4)
        tip(e_out, "Каталог для сохранения MIDI-партий и (опционально) аудио-стемов.")
        tip(ttk.Button(frm, text="Обзор…", command=self._pick_output),
            "Выбрать папку для результатов.").grid(row=1, column=2)

        # ===== Separation =====
        sep = ttk.LabelFrame(left, text="Разделение")
        sep.pack(fill="x", **pad)
        ttk.Label(sep, text="Вычислитель:").grid(row=0, column=0, sticky="e", padx=6, pady=4)
        tip(ttk.Combobox(sep, textvariable=self.device, values=["auto", "cuda", "cpu"],
                         width=8, state="readonly"),
            "Устройство вычислений. auto — GPU при наличии, иначе CPU. "
            "cuda — видеокарта NVIDIA. cpu — без ускорителя (медленно).") \
            .grid(row=0, column=1, sticky="w", padx=4)
        ttk.Label(sep, text="Окно (с):").grid(row=0, column=2, sticky="e", padx=6)
        tip(ttk.Spinbox(sep, from_=1, to=60, textvariable=self.segment, width=5),
            "Длительность окна разделения, с. Меньше — ниже расход видеопамяти. "
            "Для 4 ГБ рекомендуется 5–7.").grid(row=0, column=3, sticky="w")

        ttk.Label(sep, text="Режим:").grid(row=1, column=0, sticky="e", padx=6, pady=4)
        cb_sep = ttk.Combobox(sep, textvariable=self.separation,
                              values=["cascade", "parallel", "parallel-deep", "bestof",
                                      "bestof-auto", "single"], width=12, state="readonly")
        cb_sep.grid(row=1, column=1, sticky="w", padx=4)
        tip(cb_sep, "Алгоритм разделения на стемы. cascade — вокал (ft), затем "
                    "инструменты из остатка (6s). parallel — каждый стем из исходной "
                    "фонограммы. parallel-deep — отдельная модель на инструмент. "
                    "bestof — заданный оптимальный источник на стем. bestof-auto — "
                    "автоподбор источника по метрике (дольше всего). single — один проход.")

        # Conditional: single-mode model picker
        self.model_row = ttk.Frame(sep)
        self.model_row.grid(row=2, column=0, columnspan=4, sticky="w", padx=2)
        ttk.Label(self.model_row, text="Модель:").grid(row=0, column=0, sticky="e", padx=6)
        tip(ttk.Combobox(self.model_row, textvariable=self.model, values=DEMUCS_MODELS,
                         width=14, state="readonly"),
            "Модель Demucs для режима single.").grid(row=0, column=1, sticky="w")

        # Conditional: best-of editor button
        self.bestof_row = ttk.Frame(sep)
        self.bestof_row.grid(row=3, column=0, columnspan=4, sticky="w", padx=2)
        tip(ttk.Button(self.bestof_row, text="Источники стемов…", command=self._edit_bestof),
            "Назначить источник для каждого стема в режиме bestof.").pack(side="left", padx=6)

        tip(ttk.Checkbutton(sep, text="Сохранять стемы для прослушивания",
                            variable=self.save_audio),
            "Сохранять разделённые аудиодорожки рядом с MIDI для контроля "
            "входа транскрипции.").grid(row=4, column=0, columnspan=4, sticky="w", padx=6, pady=2)

        cb_sep.bind("<<ComboboxSelected>>", lambda _e: self._sync_separation())

        # ===== Transcription =====
        tr = ttk.LabelFrame(left, text="Транскрипция")
        tr.pack(fill="x", **pad)
        ttk.Label(tr, text="Диапазон:").grid(row=0, column=0, sticky="e", padx=6, pady=4)
        tip(ttk.Combobox(tr, textvariable=self.freq_bounds, values=["per-part", "off"],
                         width=9, state="readonly"),
            "Ограничение нот частотным диапазоном инструмента (per-part) отсекает "
            "ложные ноты от обертонов и протечек. off — без фильтра.") \
            .grid(row=0, column=1, sticky="w", padx=4)
        ttk.Label(tr, text="Фортепиано:").grid(row=0, column=2, sticky="e", padx=6)
        tip(ttk.Combobox(tr, textvariable=self.piano_engine,
                         values=["basic-pitch", "onsets-frames"], width=13, state="readonly"),
            "Движок транскрипции для стема фортепиано. onsets-frames — модель, "
            "обученная на сольном фортепиано (точнее на плотной полифонии); "
            "требует загруженной модели O&F.").grid(row=0, column=3, sticky="w")

        tip(ttk.Combobox(tr, textvariable=self.key_filter, values=["off", "auto", "manual"],
                         width=8, state="readonly"),
            "Удаление нот вне тональности. auto — определить автоматически и "
            "удалить ноты вне диатоники. manual — указать вручную ниже.") \
            .grid(row=1, column=1, sticky="w", padx=4)
        ttk.Label(tr, text="Тональность:").grid(row=1, column=0, sticky="e", padx=6, pady=4)
        self.key_entry = ttk.Entry(tr, textvariable=self.key, width=9)
        self.key_entry.grid(row=1, column=2, sticky="w", padx=4)
        tip(self.key_entry, "Тональность для режима manual: G:min, Bb:maj, Am. "
                            "Предохранитель: при потере партией более 35% нот фильтр "
                            "не применяется.")
        tr.grid_columnconfigure(3, weight=1)

        # MIDI cleanup row
        tip(ttk.Checkbutton(tr, text="Убирать октавные дубли",
                            variable=self.clean_octaves),
            "Удалять ноты-призраки на октаву выше реальной (частый артефакт "
            "спектральной транскрипции — обертон принимается за отдельную ноту). "
            "Настоящие октавные ходы сохраняются.").grid(
            row=2, column=0, columnspan=2, sticky="w", padx=6, pady=2)
        ttk.Label(tr, text="Макс. полифония:").grid(row=2, column=2, sticky="e", padx=6)
        tip(ttk.Spinbox(tr, from_=0, to=8, textvariable=self.max_polyphony, width=5),
            "Ограничение числа одновременных нот в партии (0 — без ограничения). "
            "Лишние, самые короткие ноты удаляются — против полифонической грязи.") \
            .grid(row=2, column=3, sticky="w")

        tip(ttk.Checkbutton(tr, text="Ручная настройка порогов",
                            variable=self.tweak_thresholds, command=self._sync_thresholds),
            "Включено — пороги задаются вручную ниже. Выключено — применяются "
            "профили под каждый инструмент.").grid(row=3, column=0, columnspan=4,
                                                   sticky="w", padx=6, pady=2)

        # Conditional: manual threshold sliders
        self.thr_row = ttk.Frame(tr)
        self.thr_row.grid(row=4, column=0, columnspan=4, sticky="w", padx=2)
        ttk.Label(self.thr_row, text="Атака:").grid(row=0, column=0, sticky="e", padx=4)
        self.onset_scale = ttk.Scale(self.thr_row, from_=0.05, to=0.95, variable=self.onset,
                                     orient="horizontal", length=110,
                                     command=lambda _v: self._refresh_threshold_labels())
        self.onset_scale.grid(row=0, column=1)
        self.onset_lbl = ttk.Label(self.thr_row, text="0.50", width=5)
        self.onset_lbl.grid(row=0, column=2)
        tip(self.onset_scale, "Порог обнаружения начала ноты. Повышение снижает "
                              "ложные срабатывания на плотном материале.")
        ttk.Label(self.thr_row, text="Удержание:").grid(row=0, column=3, sticky="e", padx=4)
        self.frame_scale = ttk.Scale(self.thr_row, from_=0.05, to=0.95, variable=self.frame,
                                     orient="horizontal", length=110,
                                     command=lambda _v: self._refresh_threshold_labels())
        self.frame_scale.grid(row=0, column=4)
        self.frame_lbl = ttk.Label(self.thr_row, text="0.30", width=5)
        self.frame_lbl.grid(row=0, column=5)
        tip(self.frame_scale, "Порог удержания ноты во времени.")
        ttk.Label(self.thr_row, text="Мин. нота (мс):").grid(row=1, column=0, sticky="e", padx=4, pady=2)
        self.minlen_spin = ttk.Spinbox(self.thr_row, from_=10, to=500, increment=10,
                                       textvariable=self.min_note_ms, width=6)
        self.minlen_spin.grid(row=1, column=1, sticky="w")
        tip(self.minlen_spin, "Ноты короче отбрасываются как шум.")

        # pYIN per-stem
        mono_row = ttk.Frame(tr)
        mono_row.grid(row=5, column=0, columnspan=4, sticky="w", padx=2, pady=2)
        ttk.Label(mono_row, text="Одноголосо (pYIN):").pack(side="left", padx=4)
        for stem, var in self.mono_stems.items():
            tip(ttk.Checkbutton(mono_row, text=stem, variable=var),
                f"Транскрибировать «{stem}» одной линией (pYIN). Подходит для чистых "
                "сольных партий; на дисторшне/полифонии точность ниже.").pack(side="left")

        # ===== Drums (collapsible) =====
        self.drums_box = ttk.LabelFrame(left, text="Барабаны")
        self.drums_box.pack(fill="x", **pad)
        tip(ttk.Checkbutton(self.drums_box, text="Обрабатывать барабаны",
                            variable=self.include_drums, command=self._sync_drums),
            "Транскрибировать барабаны детекцией ударов в перкуссию General MIDI. "
            "Тарелки распознаются методом superflux.").grid(row=0, column=0, columnspan=4,
                                                            sticky="w", padx=6, pady=2)
        self.drum_opts = ttk.Frame(self.drums_box)
        self.drum_opts.grid(row=1, column=0, columnspan=5, sticky="w")
        d = self.drum_opts
        ttk.Label(d, text="Чувствительность:").grid(row=0, column=0, sticky="e", padx=4, pady=2)
        self.drum_scale = ttk.Scale(d, from_=0.2, to=0.95, variable=self.drum_sensitivity,
                                    orient="horizontal", length=110,
                                    command=lambda _v: self._refresh_drum_label())
        self.drum_scale.grid(row=0, column=1)
        self.drum_lbl = ttk.Label(d, text="0.60", width=5)
        self.drum_lbl.grid(row=0, column=2)
        tip(self.drum_scale, "Чувствительность детектора ударов. Выше — больше ударов "
                             "(включая тихие) и больше ложных.")
        ttk.Label(d, text="Темп (BPM):").grid(row=0, column=3, sticky="e", padx=4)
        tip(ttk.Entry(d, textvariable=self.bpm, width=7),
            "Темп для ритмической сетки. Пусто — автоопределение с коррекцией "
            "удвоения.").grid(row=0, column=4, sticky="w")
        ttk.Label(d, text="Сдвиг сетки (мс):").grid(row=1, column=0, sticky="e", padx=4, pady=2)
        tip(ttk.Spinbox(d, from_=-250, to=250, increment=5, textvariable=self.grid_offset_ms,
                        width=6),
            "Ручная коррекция фазы ритмической сетки, мс. Фаза определяется "
            "автоматически по позициям долей; поле смещает её при неточности.") \
            .grid(row=1, column=1, sticky="w")
        ttk.Label(d, text="Отсев тарелок:").grid(row=1, column=3, sticky="e", padx=4)
        self.gate_scale = ttk.Scale(d, from_=1.0, to=1.6, variable=self.cymbal_gate,
                                    orient="horizontal", length=110,
                                    command=lambda _v: self._refresh_drum_label())
        self.gate_scale.grid(row=1, column=4)
        self.gate_lbl = ttk.Label(d, text="1.05", width=5)
        self.gate_lbl.grid(row=1, column=5)
        tip(self.gate_scale, "Порог превышения тарелкой локального фона. Выше — "
                             "меньше ложных, но можно потерять тихие тарелки.")
        tip(ttk.Checkbutton(d, text="Достраивать тарелки по сетке такта",
                            variable=self.drum_grid_fill),
            "В секциях с регулярными тарелками добавлять пропущенные удары по сетке. "
            "В паузах не добавляются.").grid(row=2, column=0, columnspan=3, sticky="w", padx=4)
        tip(ttk.Checkbutton(d, text="Выделять том-томы из малого барабана",
                            variable=self.detect_toms),
            "Отделять том-томы от малого барабана по высоте и затуханию. Метод "
            "приблизительный.").grid(row=2, column=3, columnspan=3, sticky="w", padx=4)
        self.drum_btn = ttk.Button(d, text="Только барабаны", command=self._run_drums_only)
        self.drum_btn.grid(row=3, column=4, columnspan=2, sticky="e", padx=4, pady=2)
        tip(self.drum_btn, "Повторно транскрибировать только барабаны из готового "
                           "стема, без повторного разделения.")

        # ===== Stereo split (collapsible) =====
        self.split_box = ttk.LabelFrame(left, text="Разделение по стереопанораме")
        self.split_box.pack(fill="x", **pad)
        tip(ttk.Checkbutton(self.split_box, text="Делить стемы по стереопанораме",
                            variable=self.split_enable, command=self._sync_split),
            "Разделять выбранные стемы на две партии по положению в стереополе "
            "(например, две запанорамированные гитары).").grid(
            row=0, column=0, columnspan=6, sticky="w", padx=6, pady=2)
        self.split_opts = ttk.Frame(self.split_box)
        self.split_opts.grid(row=1, column=0, columnspan=6, sticky="w")
        s = self.split_opts
        ttk.Label(s, text="Стемы:").grid(row=0, column=0, sticky="e", padx=4, pady=2)
        scol = 1
        for stem, var in self.split_stems.items():
            tip(ttk.Checkbutton(s, text=stem, variable=var),
                f"Разделить стем «{stem}» по сторонам панорамы.").grid(
                row=0, column=scol, sticky="w")
            scol += 1
        ttk.Label(s, text="Метод:").grid(row=1, column=0, sticky="e", padx=4, pady=2)
        tip(ttk.Combobox(s, textvariable=self.split_method, values=["pan", "lr", "midside"],
                         width=9, state="readonly"),
            "pan — маска по панораме (лучшее качество). lr — каналы (грубо). "
            "midside — центр/стороны.").grid(row=1, column=1, columnspan=2, sticky="w")
        ttk.Label(s, text="Имена:").grid(row=1, column=3, sticky="e", padx=4)
        tip(ttk.Combobox(s, textvariable=self.split_naming,
                         values=["LR", "solo-rhythm", "hybrid"], width=11, state="readonly"),
            "LR — _L/_R. solo-rhythm — _lead/_rhythm по эвристике. hybrid — _L/_R "
            "плюс предполагаемая роль.").grid(row=1, column=4, columnspan=2, sticky="w")

        # ===== Run buttons =====
        btns = ttk.Frame(left)
        btns.pack(fill="x", **pad)
        self.run_btn = ttk.Button(btns, text="▶ Запустить обработку", command=self._run)
        self.run_btn.pack(side="left")
        tip(self.run_btn, "Запустить полный цикл: разделение, транскрипция, MIDI.")
        self.cancel_btn = ttk.Button(btns, text="■ Остановить", command=self._cancel,
                                     state="disabled")
        self.cancel_btn.pack(side="left", padx=6)
        tip(self.cancel_btn, "Прервать текущую обработку.")

        self.progress = ttk.Progressbar(left, mode="indeterminate")
        self.progress.pack(fill="x", padx=8)
        self.status = tk.StringVar(value="Готово к запуску.")
        ttk.Label(left, textvariable=self.status, anchor="w").pack(fill="x", padx=8, pady=2)

        # ===== Right column: log (top) + results (bottom) =====
        right = ttk.Frame(self.root)
        right.pack(side="right", fill="both", expand=True, padx=(0, 8), pady=8)

        ttk.Label(right, text="Журнал обработки:").pack(anchor="w")
        logfrm = ttk.Frame(right)
        logfrm.pack(fill="both", expand=True)
        self.log = tk.Text(logfrm, wrap="none", height=16, bg="#11151c", fg="#cfe3ff",
                           insertbackground="#cfe3ff", font=("Consolas", 9))
        yscroll = ttk.Scrollbar(logfrm, orient="vertical", command=self.log.yview)
        self.log.configure(yscrollcommand=yscroll.set)
        self.log.pack(side="left", fill="both", expand=True)
        yscroll.pack(side="right", fill="y")

        resfrm = ttk.LabelFrame(right, text="Результаты (двойной клик — прослушать)")
        resfrm.pack(fill="both", expand=True, pady=(6, 0))
        self.results = tk.Listbox(resfrm, height=8, activestyle="dotbox")
        self.results.pack(side="left", fill="both", expand=True, padx=4, pady=4)
        self.results.bind("<Double-Button-1>", lambda _e: self._play_selected())
        tip(self.results, "Двойной клик — открыть стем или MIDI в системном плеере.")
        rb = ttk.Frame(resfrm)
        rb.pack(side="right", fill="y", padx=4, pady=4)
        tip(ttk.Button(rb, text="▶ Прослушать", command=self._play_selected),
            "Открыть выбранный файл в системном плеере.").pack(fill="x")
        tip(ttk.Button(rb, text="Папка", command=self._open_result_dir),
            "Открыть папку с результатами.").pack(fill="x", pady=4)

        self._refresh_threshold_labels()
        self._refresh_drum_label()
        self._sync_separation()
        self._sync_thresholds()
        self._sync_drums()
        self._sync_split()

    # ---------- progressive disclosure ----------
    # All conditional blocks are placed with .grid(); toggle them with
    # grid()/grid_remove(), which preserves their grid configuration. We don't
    # rely on winfo_manager() (it's empty before the widget is realized).
    def _show(self, widget, visible: bool) -> None:
        if visible:
            widget.grid()
        else:
            widget.grid_remove()

    def _sync_separation(self) -> None:
        mode = self.separation.get()
        self._show(self.model_row, mode == "single")
        self._show(self.bestof_row, mode == "bestof")

    def _sync_thresholds(self) -> None:
        self._show(self.thr_row, self.tweak_thresholds.get())

    def _sync_drums(self) -> None:
        self._show(self.drum_opts, self.include_drums.get())

    def _sync_split(self) -> None:
        self._show(self.split_opts, self.split_enable.get())

    def _refresh_threshold_labels(self) -> None:
        self.onset_lbl.config(text=f"{float(self.onset.get()):.2f}")
        self.frame_lbl.config(text=f"{float(self.frame.get()):.2f}")

    def _refresh_drum_label(self) -> None:
        self.drum_lbl.config(text=f"{float(self.drum_sensitivity.get()):.2f}")
        self.gate_lbl.config(text=f"{float(self.cymbal_gate.get()):.2f}")

    # ---------- best-of dialog ----------
    def _edit_bestof(self) -> None:
        win = tk.Toplevel(self.root)
        win.title("best-of: источник каждого стема")
        win.transient(self.root)
        win.resizable(False, False)
        sources = ["ft", "6s", "htdemucs", "cascade"]
        hint = {"vocals": "ft — наилучший вокал", "bass": "htdemucs — плотнее низ",
                "drums": "htdemucs — чётче транзиенты", "guitar": "6s (только тут есть гитара)",
                "other": "6s по оригиналу", "piano": "cascade — без вокальных обертонов"}
        ttk.Label(win, text="Источник для каждого стема:", padding=8).grid(
            row=0, column=0, columnspan=3, sticky="w")
        for i, stem in enumerate(("vocals", "bass", "drums", "guitar", "other", "piano"), 1):
            ttk.Label(win, text=stem + ":").grid(row=i, column=0, sticky="e", padx=8, pady=3)
            vals = ["6s", "cascade"] if stem in ("guitar", "piano") else sources
            ttk.Combobox(win, textvariable=self.bestof_plan[stem], values=vals,
                         width=10, state="readonly").grid(row=i, column=1, sticky="w", padx=6)
            ttk.Label(win, text=hint.get(stem, ""), foreground="#666").grid(
                row=i, column=2, sticky="w", padx=6)
        ttk.Button(win, text="Готово", command=win.destroy).grid(
            row=99, column=0, columnspan=3, pady=10)
        win.grab_set()

    def _run_drums_only(self) -> None:
        if not self.last_result_dir:
            out, audio = self.out_path.get().strip(), self.in_path.get().strip()
            if not out or not audio:
                messagebox.showinfo("Нет данных", "Сначала выполните полный прогон трека.")
                return
            self.last_result_dir = Path(out) / Path(audio).stem
        drums_wav = self.last_result_dir / "audio" / "drums.wav"
        if not drums_wav.is_file():
            messagebox.showerror("Нет стема", f"Не найден {drums_wav}\n"
                                 "Сначала выполните прогон с сохранением стемов.")
            return
        cmd = [sys.executable, "-u", "-c",
               "import sys; from drum_transcribe import transcribe_drums; "
               "r=transcribe_drums(sys.argv[1], sys.argv[2], sensitivity=float(sys.argv[3]), "
               "bpm=(float(sys.argv[4]) if sys.argv[4] else None), grid_fill=(sys.argv[5]=='1'), "
               "cymbal_gate=float(sys.argv[6]), detect_toms=(sys.argv[7]=='1'), "
               "grid_offset_ms=float(sys.argv[8])); "
               "print('tempo ~%s BPM, %s hits' % (r.get('_tempo','?'), r.get('_total',0))); "
               "[print('  %-9s %s'%(k,v)) for k,v in sorted(r.items()) if not k.startswith('_')]",
               str(drums_wav), str(self.last_result_dir / "drums.mid"),
               f"{float(self.drum_sensitivity.get()):.2f}", self.bpm.get().strip(),
               "1" if self.drum_grid_fill.get() else "0",
               f"{float(self.cymbal_gate.get()):.2f}",
               "1" if self.detect_toms.get() else "0",
               str(int(self.grid_offset_ms.get()))]
        self._launch(cmd, label="барабаны")

    # ---------- file pickers / playback ----------
    def _pick_input(self) -> None:
        p = filedialog.askopenfilename(title="Выберите аудиофайл",
                                       initialdir=str(HERE / "input"), filetypes=AUDIO_TYPES)
        if p:
            self.in_path.set(p)

    def _pick_output(self) -> None:
        p = filedialog.askdirectory(title="Куда сохранять MIDI",
                                    initialdir=self.out_path.get() or str(HERE))
        if p:
            self.out_path.set(p)

    def _open_out(self) -> None:
        out = Path(self.out_path.get())
        out.mkdir(parents=True, exist_ok=True)
        os.startfile(str(out))

    def _open_result_dir(self) -> None:
        if self.last_result_dir and self.last_result_dir.is_dir():
            os.startfile(str(self.last_result_dir))
        else:
            self._open_out()

    def _play_selected(self) -> None:
        sel = self.results.curselection()
        if not sel or not self.last_result_dir:
            return
        name = self.results.get(sel[0]).strip().lstrip("♪⏵ ").strip()
        for cand in (self.last_result_dir / name, self.last_result_dir / "audio" / name):
            if cand.is_file():
                try:
                    os.startfile(str(cand))
                except Exception as e:  # noqa: BLE001
                    messagebox.showerror("Не удалось открыть", f"{cand}\n\n{e!r}")
                return

    def _populate_results(self) -> None:
        self.results.delete(0, "end")
        d = self.last_result_dir
        if not d or not d.is_dir():
            return
        mids = sorted(d.glob("*.mid"))
        wavs = sorted((d / "audio").glob("*.wav")) if (d / "audio").is_dir() else []
        if mids:
            self.results.insert("end", "— MIDI —")
            for m in mids:
                self.results.insert("end", f"  ⏵ {m.name}")
        if wavs:
            self.results.insert("end", "— Аудио —")
            for w in wavs:
                self.results.insert("end", f"  ♪ {w.name}")
        if not mids and not wavs:
            self.results.insert("end", "(пусто)")

    def _log(self, text: str) -> None:
        self.log.insert("end", text)
        self.log.see("end")

    # ---------- run ----------
    def _run(self) -> None:
        audio = self.in_path.get().strip()
        if not audio or not Path(audio).is_file():
            messagebox.showerror("Нет файла", "Сначала выберите существующий аудиофайл.")
            return
        out = self.out_path.get().strip() or str(HERE / "output")
        Path(out).mkdir(parents=True, exist_ok=True)
        self._save_settings()
        self.last_result_dir = Path(out) / Path(audio).stem

        cmd = [sys.executable, "-u", str(PIPELINE), audio,
               "--out", out, "--device", self.device.get(),
               "--separation", self.separation.get(), "--model", self.model.get(),
               "--segment", str(self.segment.get()),
               "--save-audio" if self.save_audio.get() else "--no-save-audio",
               "--freq-bounds", self.freq_bounds.get(),
               "--piano-engine", self.piano_engine.get(),
               "--clean-octaves" if self.clean_octaves.get() else "--no-clean-octaves"]
        if int(self.max_polyphony.get()) > 0:
            cmd += ["--max-polyphony", str(int(self.max_polyphony.get()))]
        if self.separation.get() == "bestof":
            cmd += ["--bestof-plan",
                    ",".join(f"{k}:{v.get()}" for k, v in self.bestof_plan.items())]
        if self.include_drums.get():
            cmd += ["--include-drums",
                    "--drum-sensitivity", f"{float(self.drum_sensitivity.get()):.2f}",
                    "--cymbal-gate", f"{float(self.cymbal_gate.get()):.2f}"]
            if self.bpm.get().strip():
                cmd += ["--bpm", self.bpm.get().strip()]
            if self.drum_grid_fill.get():
                cmd.append("--drum-grid-fill")
            if self.detect_toms.get():
                cmd.append("--detect-toms")
            if int(self.grid_offset_ms.get()) != 0:
                cmd += ["--grid-offset-ms", str(int(self.grid_offset_ms.get()))]
        if self.tweak_thresholds.get():
            cmd += ["--onset", f"{float(self.onset.get()):.2f}",
                    "--frame", f"{float(self.frame.get()):.2f}",
                    "--min-note-ms", str(int(self.min_note_ms.get()))]
        if self.key_filter.get() != "off":
            cmd += ["--key-filter", self.key_filter.get()]
            if self.key_filter.get() == "manual":
                cmd += ["--key", self.key.get().strip() or "C:maj"]
        mono = [s for s, v in self.mono_stems.items() if v.get()]
        if mono:
            cmd += ["--mono-stems", ",".join(mono)]
        split = [s for s, v in self.split_stems.items() if v.get()] if self.split_enable.get() else []
        if split:
            cmd += ["--split-stem", ",".join(split),
                    "--split-method", self.split_method.get(),
                    "--split-naming", self.split_naming.get()]
        self._launch(cmd, label="трек")

    def _launch(self, cmd: list[str], label: str = "") -> None:
        if self.proc and self.proc.poll() is None:
            messagebox.showinfo("Занято", "Дождитесь завершения текущей обработки.")
            return
        self.log.delete("1.0", "end")
        self.results.delete(0, "end")
        self._log("$ " + " ".join(f'"{c}"' if " " in c else c for c in cmd) + "\n\n")
        self.status.set(f"Обработка… ({label})" if label else "Обработка…")
        self.run_btn.config(state="disabled")
        self.cancel_btn.config(state="normal")
        self.progress.start(12)
        env = dict(os.environ)
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"

        def worker() -> None:
            try:
                self.proc = subprocess.Popen(
                    cmd, cwd=str(HERE), env=env,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, encoding="utf-8", errors="replace", bufsize=1,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                assert self.proc.stdout is not None
                for line in self.proc.stdout:
                    self.q.put(line)
                code = self.proc.wait()
                self.q.put(f"\n[код завершения {code}]\n")
                self.q.put(f"__DONE__{code}")
            except Exception as e:  # noqa: BLE001
                self.q.put(f"\n[ошибка GUI] {e!r}\n")
                self.q.put("__DONE__1")

        threading.Thread(target=worker, daemon=True).start()

    def _cancel(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            self.status.set("Остановлено пользователем.")
            self._log("\n[остановлено]\n")

    def _drain_log(self) -> None:
        try:
            while True:
                item = self.q.get_nowait()
                if item.startswith("__DONE__"):
                    self._finish(item.replace("__DONE__", ""))
                else:
                    self._log(item)
        except queue.Empty:
            pass
        self.root.after(100, self._drain_log)

    def _finish(self, code: str) -> None:
        self.progress.stop()
        self.run_btn.config(state="normal")
        self.cancel_btn.config(state="disabled")
        self.proc = None
        if code == "0":
            self.status.set("Готово ✓  Двойной клик по файлу справа — прослушать.")
            self._populate_results()
        else:
            self.status.set(f"Завершено с ошибкой (код {code}). См. журнал.")

    def _on_close(self) -> None:
        if self.proc and self.proc.poll() is None:
            if not messagebox.askyesno("Идёт обработка", "Прервать и выйти?"):
                return
            self.proc.terminate()
        self._save_settings()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    try:
        ttk.Style().theme_use("vista")
    except tk.TclError:
        pass
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
