"""
M-A Splitter GUI.

Desktop shell for the audio -> stems -> MIDI pipeline. The interface stays on
Tkinter so the app keeps its zero-extra-GUI-dependency packaging, but the layout
is styled like a dark metallic audio tool with cyberpunk accents.
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import ctypes
import random
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import tkinter.font as tkfont

HERE = Path(__file__).resolve().parent
PIPELINE = HERE / "pipeline.py"
SETTINGS = HERE / "gui_settings.json"
FONT_FILE = HERE / "assets" / "fonts" / "Oxanium.ttf"

AUDIO_TYPES = [
    ("Audio", "*.wav *.mp3 *.flac *.m4a *.ogg *.aac *.wma *.aiff"),
    ("WAV", "*.wav"),
    ("MP3", "*.mp3"),
    ("FLAC", "*.flac"),
    ("All files", "*.*"),
]

DEMUCS_MODELS = ["htdemucs_6s", "htdemucs", "htdemucs_ft", "mdx_extra", "mdx_extra_q"]

COLORS = {
    "bg": "#05070A",
    "panel": "#0D1117",
    "panel_2": "#131820",
    "metal": "#1A2028",
    "metal_dark": "#080B10",
    "ridge": "#303844",
    "ridge_hi": "#667385",
    "text": "#AFC3CF",
    "text_hi": "#C7D8E0",
    "muted": "#748899",
    "cyan": "#22E6FF",
    "magenta": "#FF2D95",
    "amber": "#FFD166",
    "green": "#74FFB1",
    "entry": "#090D13",
    "entry_edge": "#465463",
    "log": "#060A10",
}

DISPLAY_FONT = "Oxanium"


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
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 8
        self._tip = tk.Toplevel(self.widget)
        self._tip.wm_overrideredirect(True)
        self._tip.wm_geometry(f"+{x}+{y}")
        tk.Label(
            self._tip,
            text=self.text,
            justify="left",
            bg="#121820",
            fg=COLORS["text"],
            relief="solid",
            borderwidth=1,
            font=("Segoe UI", 9),
            wraplength=360,
            padx=8,
            pady=5,
            highlightthickness=1,
            highlightbackground=COLORS["cyan"],
        ).pack()

    def _hide(self, _e: object = None) -> None:
        self._cancel()
        if self._tip:
            self._tip.destroy()
            self._tip = None


def tip(widget: tk.Widget, text: str) -> tk.Widget:
    Tooltip(widget, text)
    return widget


class ScrollFrame(ttk.Frame):
    """A dark scrollable frame for dense controls."""

    def __init__(self, parent: tk.Widget) -> None:
        super().__init__(parent, style="Shell.TFrame")
        self.canvas = tk.Canvas(
            self,
            bg=COLORS["bg"],
            highlightthickness=0,
            bd=0,
            relief="flat",
        )
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.inner = ttk.Frame(self.canvas, style="Shell.TFrame")
        self._win = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        self.inner.bind("<Configure>", self._configure)
        self.canvas.bind("<Configure>", self._resize)
        self.canvas.bind_all("<MouseWheel>", self._wheel, add="+")

    def _configure(self, _event: tk.Event) -> None:
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _resize(self, event: tk.Event) -> None:
        self.canvas.itemconfigure(self._win, width=event.width)

    def _wheel(self, event: tk.Event) -> None:
        if self.winfo_containing(event.x_root, event.y_root) is None:
            return
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")


class App:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        root.title("M-A Splitter")
        root.geometry("1180x720")
        root.minsize(980, 620)
        root.configure(bg=COLORS["bg"])
        root.overrideredirect(True)

        self.proc: subprocess.Popen | None = None
        self.q: "queue.Queue[str]" = queue.Queue()
        self.last_result_dir: Path | None = None
        self._drag_xy: tuple[int, int] | None = None
        self._maximized = False
        self._texture_cache: dict[tuple[int, int, str], tk.PhotoImage] = {}
        self.display_font = self._pick_display_font()

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
        self.adaptive_midi = tk.BooleanVar(value=cfg.get("adaptive_midi", True))
        self.midi_preprocess = tk.BooleanVar(value=cfg.get("midi_preprocess", True))
        self.smart_clean = tk.BooleanVar(value=cfg.get("smart_clean", True))
        self.clean_octaves = tk.BooleanVar(value=cfg.get("clean_octaves", True))
        self.max_polyphony = tk.IntVar(value=cfg.get("max_polyphony", 0))
        saved_mono = set(cfg.get("mono_stems", []))
        self.mono_stems = {
            s: tk.BooleanVar(value=s in saved_mono)
            for s in ("bass", "vocals", "other", "guitar", "piano")
        }
        self.drum_sensitivity = tk.DoubleVar(value=cfg.get("drum_sensitivity", 0.6))
        self.bpm = tk.StringVar(value=cfg.get("bpm", ""))
        self.drum_grid_fill = tk.BooleanVar(value=cfg.get("drum_grid_fill", False))
        self.cymbal_gate = tk.DoubleVar(value=cfg.get("cymbal_gate", 1.05))
        self.detect_toms = tk.BooleanVar(value=cfg.get("detect_toms", False))
        self.grid_offset_ms = tk.IntVar(value=cfg.get("grid_offset_ms", 0))
        default_bestof = {
            "vocals": "ft",
            "bass": "htdemucs",
            "drums": "htdemucs",
            "guitar": "6s",
            "other": "6s",
            "piano": "cascade",
        }
        saved = cfg.get("bestof_plan", {})
        self.bestof_plan = {
            k: tk.StringVar(value=saved.get(k, v)) for k, v in default_bestof.items()
        }
        saved_split = set(cfg.get("split_stems", []))
        self.split_stems = {
            s: tk.BooleanVar(value=s in saved_split)
            for s in ("guitar", "other", "vocals", "piano")
        }
        self.split_method = tk.StringVar(value=cfg.get("split_method", "pan"))
        self.split_naming = tk.StringVar(value=cfg.get("split_naming", "LR"))
        self.split_enable = tk.BooleanVar(value=bool(saved_split))
        self.status = tk.StringVar(value="Готово к запуску")

        self._configure_styles()
        self._build()
        self.root.after(100, self._drain_log)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ---------- settings ----------
    def _pick_display_font(self) -> str:
        if FONT_FILE.is_file() and os.name == "nt":
            try:
                ctypes.windll.gdi32.AddFontResourceExW(str(FONT_FILE), 0x10, 0)
            except Exception:
                pass
        fonts = {name.lower(): name for name in tkfont.families(self.root)}
        for candidate in ("Oxanium", "ST MicroSquare Ex", "Furore", "Bahnschrift SemiBold", "Bahnschrift"):
            found = fonts.get(candidate.lower())
            if found:
                return found
        return DISPLAY_FONT

    def _load_settings(self) -> dict:
        try:
            return json.loads(SETTINGS.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _save_settings(self) -> None:
        data = {
            "in_path": self.in_path.get(),
            "out_path": self.out_path.get(),
            "device": self.device.get(),
            "separation": self.separation.get(),
            "model": self.model.get(),
            "segment": int(self.segment.get()),
            "include_drums": bool(self.include_drums.get()),
            "save_audio": bool(self.save_audio.get()),
            "tweak_thresholds": bool(self.tweak_thresholds.get()),
            "onset": round(float(self.onset.get()), 2),
            "frame": round(float(self.frame.get()), 2),
            "min_note_ms": int(self.min_note_ms.get()),
            "freq_bounds": self.freq_bounds.get(),
            "key_filter": self.key_filter.get(),
            "key": self.key.get(),
            "piano_engine": self.piano_engine.get(),
            "adaptive_midi": bool(self.adaptive_midi.get()),
            "midi_preprocess": bool(self.midi_preprocess.get()),
            "smart_clean": bool(self.smart_clean.get()),
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
            SETTINGS.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

    # ---------- theme ----------
    def _configure_styles(self) -> None:
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        self.root.option_add("*Background", COLORS["panel"])
        self.root.option_add("*Foreground", COLORS["text_hi"])
        self.root.option_add("*Entry.Background", COLORS["entry"])
        self.root.option_add("*Entry.Foreground", COLORS["text_hi"])
        self.root.option_add("*Entry.InsertBackground", COLORS["cyan"])
        self.root.option_add("*Listbox.Background", COLORS["entry"])
        self.root.option_add("*Listbox.Foreground", COLORS["text"])
        self.root.option_add("*selectBackground", "#183F4A")
        self.root.option_add("*selectForeground", COLORS["cyan"])

        style.configure(".", font=("Segoe UI", 10))
        style.configure("Shell.TFrame", background=COLORS["bg"])
        style.configure("Panel.TFrame", background=COLORS["panel"])
        style.configure("Card.TFrame", background=COLORS["panel_2"], relief="flat")
        style.configure("TLabel", background=COLORS["bg"], foreground=COLORS["text"])
        style.configure("Muted.TLabel", background=COLORS["bg"], foreground=COLORS["muted"])
        style.configure("Panel.TLabel", background=COLORS["panel"], foreground=COLORS["text"])
        style.configure("Card.TLabel", background=COLORS["panel_2"], foreground=COLORS["text"])
        style.configure("MutedCard.TLabel", background=COLORS["panel_2"], foreground=COLORS["muted"])
        style.configure(
            "Rack.TLabelframe",
            background=COLORS["panel"],
            foreground=COLORS["cyan"],
            bordercolor=COLORS["ridge_hi"],
            relief="solid",
            borderwidth=1,
        )
        style.configure(
            "Rack.TLabelframe.Label",
            background=COLORS["panel"],
            foreground=COLORS["cyan"],
            font=("Segoe UI Semibold", 10),
        )
        style.configure(
            "TEntry",
            fieldbackground=COLORS["entry"],
            foreground=COLORS["text_hi"],
            insertcolor=COLORS["cyan"],
            bordercolor=COLORS["entry_edge"],
            lightcolor=COLORS["entry_edge"],
            darkcolor=COLORS["ridge"],
            padding=5,
        )
        style.configure(
            "TCombobox",
            fieldbackground=COLORS["entry"],
            background=COLORS["panel_2"],
            foreground=COLORS["text_hi"],
            arrowcolor=COLORS["cyan"],
            bordercolor=COLORS["entry_edge"],
            padding=4,
        )
        style.map(
            "TCombobox",
            fieldbackground=[("readonly", COLORS["entry"]), ("!disabled", COLORS["entry"])],
            foreground=[("readonly", COLORS["text_hi"]), ("!disabled", COLORS["text_hi"])],
            background=[("active", "#18222D"), ("readonly", COLORS["entry"])],
        )
        style.configure(
            "TSpinbox",
            fieldbackground=COLORS["entry"],
            foreground=COLORS["text_hi"],
            arrowsize=13,
            bordercolor=COLORS["entry_edge"],
        )
        style.configure("TCheckbutton", background=COLORS["panel"], foreground=COLORS["text"])
        style.map(
            "TCheckbutton",
            foreground=[("active", COLORS["cyan"]), ("selected", COLORS["green"])],
            background=[("active", COLORS["panel"])],
            indicatorcolor=[("selected", COLORS["cyan"]), ("!selected", COLORS["entry"])],
        )
        style.configure(
            "Accent.TButton",
            background=COLORS["cyan"],
            foreground="#031016",
            bordercolor=COLORS["cyan"],
            focusthickness=0,
            padding=(12, 7),
            font=("Segoe UI Semibold", 10),
        )
        style.map("Accent.TButton", background=[("active", "#69F2FF"), ("disabled", "#33404A")])
        style.configure(
            "Danger.TButton",
            background=COLORS["magenta"],
            foreground="#17040D",
            bordercolor=COLORS["magenta"],
            focusthickness=0,
            padding=(12, 7),
            font=("Segoe UI Semibold", 10),
        )
        style.map("Danger.TButton", background=[("active", "#FF70B4"), ("disabled", "#3D2732")])
        style.configure(
            "Metal.TButton",
            background=COLORS["panel_2"],
            foreground=COLORS["text"],
            bordercolor=COLORS["ridge_hi"],
            padding=(10, 6),
        )
        style.map("Metal.TButton", background=[("active", "#222A36")], foreground=[("active", COLORS["cyan"])])
        style.configure(
            "Cyber.Horizontal.TProgressbar",
            troughcolor=COLORS["entry"],
            background=COLORS["magenta"],
            bordercolor=COLORS["ridge"],
            lightcolor=COLORS["cyan"],
            darkcolor=COLORS["magenta"],
        )
        style.configure("Vertical.TScrollbar", background=COLORS["panel_2"], troughcolor=COLORS["bg"])

    # ---------- UI ----------
    def _build(self) -> None:
        shell = tk.Frame(
            self.root,
            bg=COLORS["bg"],
            highlightthickness=1,
            highlightbackground=COLORS["cyan"],
            highlightcolor=COLORS["cyan"],
        )
        shell.pack(fill="both", expand=True)

        self._build_titlebar(shell)
        self._build_header(shell)

        body = ttk.Frame(shell, style="Shell.TFrame")
        body.pack(fill="both", expand=True, padx=12, pady=(4, 12))
        body.columnconfigure(0, minsize=455)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        scroll = ScrollFrame(body)
        scroll.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        left = scroll.inner

        self._build_file_card(left)
        self._build_separation_card(left)
        self._build_transcription_card(left)
        self._build_drums_card(left)
        self._build_split_card(left)
        self._build_transport(left)
        self._build_right_panel(body)

        self._refresh_threshold_labels()
        self._refresh_drum_label()
        self._sync_separation()
        self._sync_thresholds()
        self._sync_drums()
        self._sync_split()

    def _paint_metal(self, canvas: tk.Canvas, width: int, height: int, base: str | None = None, shine: float = 0.35) -> None:
        base = base or COLORS["metal"]
        canvas.delete("texture")
        img = self._brushed_texture(max(1, width), max(1, height), base, shine)
        canvas.create_image(0, 0, anchor="nw", image=img, tags="texture")
        canvas._metal_img = img  # keep Tk image alive
        canvas.tag_lower("texture")

    def _brushed_texture(self, width: int, height: int, base: str, shine: float) -> tk.PhotoImage:
        key = (min(width, 1600), min(height, 900), f"{base}:{shine:.2f}")
        if key in self._texture_cache:
            return self._texture_cache[key]

        def hex_to_rgb(value: str) -> tuple[int, int, int]:
            value = value.lstrip("#")
            return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)

        def clamp(v: float) -> int:
            return max(0, min(255, int(v)))

        br, bg, bb = hex_to_rgb(base)
        rng = random.Random(width * 92821 + height * 68917 + sum(ord(c) for c in base))
        img = tk.PhotoImage(width=width, height=height)
        rows = []
        for y in range(height):
            row_noise = rng.randint(-22, 20)
            fine = rng.randint(-7, 7)
            scratch = rng.random() < 0.18
            scratch_delta = rng.choice((-42, -28, 26, 38)) if scratch else 0
            row = []
            for x in range(width):
                cx = (x / max(1, width - 1)) - 0.5
                cy = (y / max(1, height - 1)) - 0.5
                horizontal_grain = rng.randint(-9, 9)
                long_streak = rng.randint(-4, 4) if x % 9 == 0 else 0
                center_glow = max(0.0, 1.0 - (abs(cx) * 2.2 + abs(cy) * 0.75)) * 70 * shine
                edge_vignette = (abs(cx) ** 1.7) * 48 + (abs(cy) ** 1.4) * 20
                delta = row_noise + fine + horizontal_grain + long_streak + scratch_delta + center_glow - edge_vignette
                row.append("#{0:02x}{1:02x}{2:02x}".format(
                    clamp(br + delta),
                    clamp(bg + delta),
                    clamp(bb + delta),
                ))
            rows.append("{" + " ".join(row) + "}")
        img.put(" ".join(rows))
        self._texture_cache[key] = img
        if len(self._texture_cache) > 32:
            self._texture_cache.pop(next(iter(self._texture_cache)))
        return img

    def _metal_button(self, parent: tk.Widget, text: str, command, width: int | None = None) -> tk.Button:
        return tk.Button(
            parent,
            text=text,
            command=command,
            width=width or 0,
            bg="#111720",
            fg=COLORS["text_hi"],
            activebackground="#1F2834",
            activeforeground=COLORS["cyan"],
            relief="flat",
            bd=0,
            highlightthickness=1,
            highlightbackground=COLORS["entry_edge"],
            highlightcolor=COLORS["cyan"],
            padx=10,
            pady=5,
            font=("Segoe UI Semibold", 9),
            cursor="hand2",
        )

    def _accent_button(self, parent: tk.Widget, text: str, command, danger: bool = False) -> tk.Button:
        bg = COLORS["magenta"] if danger else COLORS["cyan"]
        fg = "#190511" if danger else "#031016"
        active = "#FF70B4" if danger else "#69F2FF"
        return tk.Button(
            parent,
            text=text,
            command=command,
            bg=bg,
            fg=fg,
            activebackground=active,
            activeforeground=fg,
            disabledforeground="#6E7885",
            relief="flat",
            bd=0,
            padx=16,
            pady=8,
            font=(self.display_font, 10),
            cursor="hand2",
        )

    def _check(self, parent: tk.Widget, text: str, variable: tk.Variable, command=None) -> tk.Checkbutton:
        return tk.Checkbutton(
            parent,
            text=text,
            variable=variable,
            command=command,
            indicatoron=False,
            bg=COLORS["panel"],
            fg=COLORS["text_hi"],
            activebackground="#18222B",
            activeforeground=COLORS["cyan"],
            selectcolor=COLORS["entry"],
            relief="flat",
            bd=0,
            highlightthickness=1,
            highlightbackground=COLORS["ridge"],
            highlightcolor=COLORS["cyan"],
            padx=7,
            pady=2,
            font=("Segoe UI Semibold", 8),
            cursor="hand2",
        )

    def _entry(self, parent: tk.Widget, variable: tk.StringVar, width: int = 12) -> tk.Entry:
        return tk.Entry(
            parent,
            textvariable=variable,
            width=width,
            bg=COLORS["entry"],
            fg=COLORS["text"],
            insertbackground=COLORS["cyan"],
            selectbackground="#183F4A",
            selectforeground=COLORS["cyan"],
            relief="flat",
            bd=0,
            highlightthickness=1,
            highlightbackground=COLORS["entry_edge"],
            highlightcolor=COLORS["cyan"],
            font=("Segoe UI Semibold", 9),
        )

    def _spinbox(self, parent: tk.Widget, variable: tk.Variable, from_: int | float, to: int | float, width: int = 6, increment: int | float = 1) -> tk.Spinbox:
        return tk.Spinbox(
            parent,
            from_=from_,
            to=to,
            increment=increment,
            textvariable=variable,
            width=width,
            bg=COLORS["entry"],
            fg=COLORS["text"],
            insertbackground=COLORS["cyan"],
            buttonbackground=COLORS["metal"],
            relief="flat",
            bd=0,
            highlightthickness=1,
            highlightbackground=COLORS["entry_edge"],
            highlightcolor=COLORS["cyan"],
            font=("Segoe UI Semibold", 9),
        )

    def _combo(self, parent: tk.Widget, variable: tk.StringVar, values: list[str], width: int = 12, command=None) -> tk.OptionMenu:
        menu = tk.OptionMenu(parent, variable, *values, command=command)
        menu.configure(
            width=width,
            anchor="w",
            bg=COLORS["entry"],
            fg=COLORS["text"],
            activebackground="#18222B",
            activeforeground=COLORS["cyan"],
            relief="flat",
            bd=0,
            highlightthickness=1,
            highlightbackground=COLORS["entry_edge"],
            highlightcolor=COLORS["cyan"],
            font=("Segoe UI Semibold", 9),
            indicatoron=False,
            padx=6,
            pady=2,
        )
        menu["menu"].configure(
            bg=COLORS["panel_2"],
            fg=COLORS["text_hi"],
            activebackground="#183F4A",
            activeforeground=COLORS["cyan"],
            bd=0,
            tearoff=False,
            font=("Segoe UI", 9),
        )
        return menu

    def _build_titlebar(self, parent: tk.Widget) -> None:
        bar = tk.Canvas(parent, height=34, bg=COLORS["metal_dark"], bd=0, highlightthickness=0)
        bar.pack(fill="x")
        bar.bind("<Configure>", lambda e: self._paint_metal(bar, e.width, e.height, "#313740", 0.62))
        bar.bind("<ButtonPress-1>", self._start_window_drag)
        bar.bind("<B1-Motion>", self._move_window)
        bar.bind("<Double-Button-1>", lambda _e: self._toggle_maximize())
        bar.create_text(
            14,
            17,
            anchor="w",
            text="M-A SPLITTER // CYBER RACK",
            fill=COLORS["cyan"],
            font=(self.display_font, 10),
            tags="title",
        )
        controls = tk.Frame(bar, bg=COLORS["metal_dark"])
        controls.place(relx=1.0, x=-6, y=5, anchor="ne")
        self._metal_button(controls, "_", self._minimize, width=3).pack(side="left", padx=2)
        self._metal_button(controls, "□", self._toggle_maximize, width=3).pack(side="left", padx=2)
        close = self._metal_button(controls, "X", self._on_close, width=3)
        close.configure(bg="#23111A", activebackground=COLORS["magenta"], fg=COLORS["text"])
        close.pack(side="left", padx=2)

    def _build_header(self, parent: tk.Widget) -> None:
        header = tk.Canvas(parent, height=86, bg=COLORS["bg"], bd=0, highlightthickness=0)
        header.pack(fill="x")
        header.bind("<Configure>", lambda e: self._paint_header(header, e.width, e.height))
        header.create_rectangle(0, 0, 4000, 86, fill=COLORS["bg"], outline="", tags="texture")
        header.create_line(18, 74, 1160, 74, fill=COLORS["ridge_hi"], width=1)
        header.create_line(18, 75, 260, 75, fill=COLORS["cyan"], width=2)
        header.create_line(266, 75, 390, 75, fill=COLORS["magenta"], width=2)
        header.create_text(
            22,
            20,
            anchor="nw",
            text="M-A SPLITTER",
            fill=COLORS["text"],
            font=(self.display_font, 28),
        )
        header.create_text(
            24,
            55,
            anchor="nw",
            text="OFFLINE STEM SEPARATION / MIDI EXTRACTION",
            fill=COLORS["muted"],
            font=("Consolas", 9),
        )
        tk.Label(
            header,
            textvariable=self.status,
            bg=COLORS["bg"],
            fg=COLORS["green"],
            font=("Segoe UI", 10),
        ).place(relx=1.0, x=-38, y=28, anchor="ne")

    def _paint_header(self, canvas: tk.Canvas, width: int, height: int) -> None:
        canvas.delete("texture")
        self._paint_metal(canvas, width, height, COLORS["bg"])
        canvas.create_rectangle(0, 0, width, height, fill="", outline=COLORS["ridge"], tags="texture")
        canvas.tag_lower("texture")

    def _section(self, parent: tk.Widget, title: str) -> tk.Frame:
        outer = tk.Frame(
            parent,
            bg=COLORS["panel"],
            highlightthickness=1,
            highlightbackground=COLORS["ridge_hi"],
        )
        outer.pack(fill="x", padx=2, pady=7)
        titlebar = tk.Canvas(outer, height=25, bg=COLORS["metal"], bd=0, highlightthickness=0)
        titlebar.pack(fill="x")
        titlebar.bind("<Configure>", lambda e, c=titlebar: self._paint_metal(c, e.width, e.height, "#343B45", 0.58))
        titlebar.create_text(
            9,
            13,
            anchor="w",
            text=title,
            fill=COLORS["cyan"],
            font=(self.display_font, 9),
        )
        frame = tk.Frame(outer, bg=COLORS["panel"], padx=10, pady=10)
        frame.pack(fill="x")
        rail = tk.Canvas(outer, height=7, bg=COLORS["panel"], bd=0, highlightthickness=0)
        rail.pack(fill="x")
        rail.bind("<Configure>", lambda e, c=rail: self._paint_metal(c, e.width, e.height, "#222831", 0.36))
        for col in range(6):
            frame.columnconfigure(col, weight=1 if col in (1, 3, 5) else 0)
        return frame

    def _label(self, parent: tk.Widget, text: str, row: int, col: int = 0) -> None:
        tk.Label(parent, text=text, bg=COLORS["panel"], fg=COLORS["text_hi"], font=("Segoe UI Semibold", 9)).grid(
            row=row, column=col, sticky="e", padx=(0, 7), pady=4
        )

    def _start_window_drag(self, event: tk.Event) -> None:
        if self._maximized:
            return
        self._drag_xy = (event.x_root - self.root.winfo_x(), event.y_root - self.root.winfo_y())

    def _move_window(self, event: tk.Event) -> None:
        if not self._drag_xy or self._maximized:
            return
        dx, dy = self._drag_xy
        self.root.geometry(f"+{event.x_root - dx}+{event.y_root - dy}")

    def _minimize(self) -> None:
        self.root.overrideredirect(False)
        self.root.iconify()
        self.root.after(200, lambda: self.root.overrideredirect(True))

    def _toggle_maximize(self) -> None:
        if self._maximized:
            self.root.state("normal")
            self.root.geometry("1180x720")
            self._maximized = False
        else:
            self.root.state("zoomed")
            self._maximized = True

    def _build_file_card(self, left: tk.Widget) -> None:
        frm = self._section(left, "SOURCE / OUTPUT")
        self._label(frm, "Аудио", 0)
        e_in = self._entry(frm, self.in_path, width=36)
        e_in.grid(row=0, column=1, columnspan=4, sticky="ew", pady=4)
        tip(e_in, "Исходный файл: wav, mp3, flac, m4a и другие форматы через ffmpeg.")
        tip(self._metal_button(frm, "Browse", self._pick_input),
            "Выбрать аудиофайл.").grid(row=0, column=5, sticky="e", padx=(7, 0))

        self._label(frm, "Вывод", 1)
        e_out = self._entry(frm, self.out_path, width=36)
        e_out.grid(row=1, column=1, columnspan=4, sticky="ew", pady=4)
        tip(e_out, "Папка для MIDI и, если включено, аудио-стемов.")
        tip(self._metal_button(frm, "Folder", self._pick_output),
            "Выбрать папку результатов.").grid(row=1, column=5, sticky="e", padx=(7, 0))

    def _build_separation_card(self, left: tk.Widget) -> None:
        sep = self._section(left, "DEMIX ENGINE")
        self._label(sep, "Device", 0)
        tip(self._combo(sep, self.device, ["auto", "cuda", "cpu"], width=9),
            "auto выберет CUDA при наличии NVIDIA GPU, иначе CPU.").grid(row=0, column=1, sticky="w")
        self._label(sep, "Segment", 0, 2)
        tip(self._spinbox(sep, self.segment, from_=1, to=60, width=6),
            "Длина окна Demucs в секундах. Меньше - ниже расход VRAM.").grid(row=0, column=3, sticky="w")

        self._label(sep, "Mode", 1)
        cb_sep = self._combo(
            sep,
            self.separation,
            ["cascade", "parallel", "parallel-deep", "bestof", "bestof-auto", "single"],
            width=16,
            command=lambda _v: self._sync_separation(),
        )
        cb_sep.grid(row=1, column=1, columnspan=2, sticky="w", pady=4)
        tip(cb_sep, "Режим разделения: cascade быстрый и чистый; bestof-auto самый тяжёлый, но анализирует кандидатов.")

        self.model_row = ttk.Frame(sep, style="Panel.TFrame")
        self.model_row.grid(row=2, column=0, columnspan=6, sticky="w", pady=(2, 0))
        ttk.Label(self.model_row, text="Model", style="Panel.TLabel").pack(side="left", padx=(0, 7))
        tip(self._combo(self.model_row, self.model, DEMUCS_MODELS, width=16),
            "Модель Demucs для режима single.").pack(side="left")

        self.bestof_row = ttk.Frame(sep, style="Panel.TFrame")
        self.bestof_row.grid(row=3, column=0, columnspan=6, sticky="w", pady=(2, 0))
        tip(self._metal_button(self.bestof_row, "Stem Sources", self._edit_bestof),
            "Назначить источник для каждого стема в режиме bestof.").pack(side="left")

        tip(self._check(sep, "Сохранять WAV-стемы рядом с MIDI", self.save_audio),
            "Удобно для прослушивания того, что реально услышал транскрайбер.").grid(
            row=4, column=0, columnspan=6, sticky="w", pady=(8, 0)
        )

    def _build_transcription_card(self, left: tk.Widget) -> None:
        tr = self._section(left, "MIDI TRANSCRIBE")
        self._label(tr, "Range", 0)
        tip(self._combo(tr, self.freq_bounds, ["per-part", "off"], width=10),
            "per-part отсекает ноты вне физического диапазона инструмента.").grid(row=0, column=1, sticky="w")
        self._label(tr, "Piano", 0, 2)
        tip(self._combo(tr, self.piano_engine, ["basic-pitch", "onsets-frames"], width=14),
            "onsets-frames точнее на сольном пианино, но требует отдельную модель.").grid(row=0, column=3, columnspan=2, sticky="w")

        self._label(tr, "Key filter", 1)
        tip(self._combo(tr, self.key_filter, ["off", "auto", "manual"], width=10),
            "Удаление нот вне тональности. Есть предохранитель от слишком сильной потери нот.").grid(row=1, column=1, sticky="w")
        self._label(tr, "Key", 1, 2)
        self.key_entry = self._entry(tr, self.key, width=10)
        self.key_entry.grid(row=1, column=3, sticky="w")
        tip(self.key_entry, "Форматы: G:min, Bb:maj, Am.")

        tip(self._check(tr, "Убирать октавные призраки", self.clean_octaves),
            "Удаляет типичный артефакт, когда обертон принят за отдельную ноту.").grid(
            row=2, column=0, columnspan=3, sticky="w", pady=(6, 0)
        )
        self._label(tr, "Max poly", 2, 3)
        tip(self._spinbox(tr, self.max_polyphony, from_=0, to=8, width=5),
            "0 - без ограничения.").grid(row=2, column=4, sticky="w", pady=(6, 0))

        adaptive_row = ttk.Frame(tr, style="Panel.TFrame")
        adaptive_row.grid(row=3, column=0, columnspan=6, sticky="w", pady=(7, 0))
        tip(self._check(adaptive_row, "Adaptive", self.adaptive_midi),
            "Анализ стема и автоподбор профиля MIDI.").pack(side="left", padx=(0, 5))
        tip(self._check(adaptive_row, "Preprocess", self.midi_preprocess),
            "Helper-WAV для MIDI: bandpass, HPSS, noise gate.").pack(side="left", padx=(0, 5))
        tip(self._check(adaptive_row, "Smart clean", self.smart_clean),
            "Склейка микроразрывов и удаление тихих микронот.").pack(side="left")

        tip(self._check(tr, "Ручные пороги атаки и удержания", self.tweak_thresholds, command=self._sync_thresholds),
            "Если выключено, используются профили под каждый инструмент.").grid(
            row=4, column=0, columnspan=6, sticky="w", pady=(6, 0)
        )

        self.thr_row = ttk.Frame(tr, style="Panel.TFrame")
        self.thr_row.grid(row=5, column=0, columnspan=6, sticky="ew", pady=(4, 0))
        ttk.Label(self.thr_row, text="Attack", style="Panel.TLabel").grid(row=0, column=0, padx=(0, 6))
        self.onset_scale = ttk.Scale(
            self.thr_row, from_=0.05, to=0.95, variable=self.onset, orient="horizontal",
            length=115, command=lambda _v: self._refresh_threshold_labels()
        )
        self.onset_scale.grid(row=0, column=1)
        self.onset_lbl = ttk.Label(self.thr_row, text="0.50", width=5, style="Panel.TLabel")
        self.onset_lbl.grid(row=0, column=2, padx=(4, 12))
        ttk.Label(self.thr_row, text="Hold", style="Panel.TLabel").grid(row=0, column=3, padx=(0, 6))
        self.frame_scale = ttk.Scale(
            self.thr_row, from_=0.05, to=0.95, variable=self.frame, orient="horizontal",
            length=115, command=lambda _v: self._refresh_threshold_labels()
        )
        self.frame_scale.grid(row=0, column=4)
        self.frame_lbl = ttk.Label(self.thr_row, text="0.30", width=5, style="Panel.TLabel")
        self.frame_lbl.grid(row=0, column=5, padx=(4, 0))
        ttk.Label(self.thr_row, text="Min note ms", style="Panel.TLabel").grid(row=1, column=0, padx=(0, 6), pady=(5, 0))
        self.minlen_spin = self._spinbox(self.thr_row, self.min_note_ms, from_=10, to=500, increment=10, width=7)
        self.minlen_spin.grid(row=1, column=1, sticky="w", pady=(5, 0))

        mono_row = ttk.Frame(tr, style="Panel.TFrame")
        mono_row.grid(row=6, column=0, columnspan=6, sticky="w", pady=(8, 0))
        ttk.Label(mono_row, text="pYIN mono:", style="Panel.TLabel").pack(side="left", padx=(0, 6))
        for stem, var in self.mono_stems.items():
            tip(self._check(mono_row, stem, var),
                f"Транскрибировать {stem} как одну мелодическую линию.").pack(side="left", padx=(0, 4))

    def _build_drums_card(self, left: tk.Widget) -> None:
        self.drums_box = self._section(left, "DRUM DETECTOR")
        tip(self._check(self.drums_box, "Обрабатывать барабаны в GM percussion MIDI", self.include_drums, command=self._sync_drums),
            "Отдельная onset-детекция вместо pitch-транскрипции.").grid(row=0, column=0, columnspan=6, sticky="w")

        self.drum_opts = ttk.Frame(self.drums_box, style="Panel.TFrame")
        self.drum_opts.grid(row=1, column=0, columnspan=6, sticky="ew", pady=(7, 0))
        d = self.drum_opts
        ttk.Label(d, text="Sensitivity", style="Panel.TLabel").grid(row=0, column=0, sticky="e", padx=(0, 6))
        self.drum_scale = ttk.Scale(d, from_=0.2, to=0.95, variable=self.drum_sensitivity, orient="horizontal", length=115, command=lambda _v: self._refresh_drum_label())
        self.drum_scale.grid(row=0, column=1)
        self.drum_lbl = ttk.Label(d, text="0.60", width=5, style="Panel.TLabel")
        self.drum_lbl.grid(row=0, column=2, padx=(4, 12))
        ttk.Label(d, text="BPM", style="Panel.TLabel").grid(row=0, column=3, padx=(0, 6))
        tip(self._entry(d, self.bpm, width=8), "Пусто - автоопределение темпа.").grid(row=0, column=4, sticky="w")
        ttk.Label(d, text="Grid ms", style="Panel.TLabel").grid(row=1, column=0, sticky="e", padx=(0, 6), pady=(5, 0))
        tip(self._spinbox(d, self.grid_offset_ms, from_=-250, to=250, increment=5, width=7),
            "Ручной сдвиг фазы ритмической сетки.").grid(row=1, column=1, sticky="w", pady=(5, 0))
        ttk.Label(d, text="Cym gate", style="Panel.TLabel").grid(row=1, column=3, padx=(0, 6), pady=(5, 0))
        self.gate_scale = ttk.Scale(d, from_=1.0, to=1.6, variable=self.cymbal_gate, orient="horizontal", length=115, command=lambda _v: self._refresh_drum_label())
        self.gate_scale.grid(row=1, column=4, pady=(5, 0))
        self.gate_lbl = ttk.Label(d, text="1.05", width=5, style="Panel.TLabel")
        self.gate_lbl.grid(row=1, column=5, padx=(4, 0), pady=(5, 0))
        tip(self._check(d, "Достраивать тарелки по сетке", self.drum_grid_fill),
            "Добавляет пропущенные регулярные хэты/райды только в активных секциях.").grid(row=2, column=0, columnspan=3, sticky="w", pady=(7, 0))
        tip(self._check(d, "Пробовать выделять томы", self.detect_toms),
            "Приблизительная эвристика по высоте и затуханию.").grid(row=2, column=3, columnspan=3, sticky="w", pady=(7, 0))
        self.drum_btn = self._metal_button(d, "Only Drums", self._run_drums_only)
        self.drum_btn.grid(row=3, column=4, columnspan=2, sticky="e", pady=(8, 0))

    def _build_split_card(self, left: tk.Widget) -> None:
        self.split_box = self._section(left, "STEREO FIELD SPLIT")
        tip(self._check(self.split_box, "Делить выбранные стемы по панораме", self.split_enable, command=self._sync_split),
            "Полезно для двух гитар или партий, разведённых по стереополю.").grid(row=0, column=0, columnspan=6, sticky="w")

        self.split_opts = ttk.Frame(self.split_box, style="Panel.TFrame")
        self.split_opts.grid(row=1, column=0, columnspan=6, sticky="ew", pady=(7, 0))
        s = self.split_opts
        ttk.Label(s, text="Stems", style="Panel.TLabel").grid(row=0, column=0, sticky="e", padx=(0, 6))
        col = 1
        for stem, var in self.split_stems.items():
            tip(self._check(s, stem, var), f"Разделить {stem} на две партии.").grid(row=0, column=col, sticky="w")
            col += 1
        ttk.Label(s, text="Method", style="Panel.TLabel").grid(row=1, column=0, sticky="e", padx=(0, 6), pady=(5, 0))
        tip(self._combo(s, self.split_method, ["pan", "lr", "midside"], width=10),
            "pan - мягкая маска по панораме; lr - каналы; midside - центр/стороны.").grid(row=1, column=1, columnspan=2, sticky="w", pady=(5, 0))
        ttk.Label(s, text="Names", style="Panel.TLabel").grid(row=1, column=3, sticky="e", padx=(0, 6), pady=(5, 0))
        tip(self._combo(s, self.split_naming, ["LR", "solo-rhythm", "hybrid"], width=13),
            "Схема имён для двух полученных партий.").grid(row=1, column=4, columnspan=2, sticky="w", pady=(5, 0))

    def _build_transport(self, left: tk.Widget) -> None:
        bar = ttk.Frame(left, style="Shell.TFrame")
        bar.pack(fill="x", padx=2, pady=(8, 4))
        self.run_btn = self._accent_button(bar, "RUN SPLIT", self._run)
        self.run_btn.pack(side="left")
        self.cancel_btn = self._accent_button(bar, "STOP", self._cancel, danger=True)
        self.cancel_btn.config(state="disabled")
        self.cancel_btn.pack(side="left", padx=8)
        self.progress = ttk.Progressbar(left, mode="indeterminate", style="Cyber.Horizontal.TProgressbar")
        self.progress.pack(fill="x", padx=2, pady=(0, 8))

    def _build_right_panel(self, body: tk.Widget) -> None:
        right = ttk.Frame(body, style="Shell.TFrame")
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(1, weight=3)
        right.rowconfigure(3, weight=2)
        right.columnconfigure(0, weight=1)

        ttk.Label(right, text="PROCESS LOG", style="Muted.TLabel").grid(row=0, column=0, sticky="w", pady=(0, 5))
        logfrm = ttk.Frame(right, style="Card.TFrame", padding=1)
        logfrm.grid(row=1, column=0, sticky="nsew")
        logfrm.rowconfigure(0, weight=1)
        logfrm.columnconfigure(0, weight=1)
        self.log = tk.Text(
            logfrm,
            wrap="none",
            height=18,
            bg=COLORS["log"],
            fg="#C8F7FF",
            insertbackground=COLORS["cyan"],
            selectbackground="#24414A",
            font=("Cascadia Mono", 9),
            relief="flat",
            bd=0,
            padx=10,
            pady=10,
        )
        yscroll = ttk.Scrollbar(logfrm, orient="vertical", command=self.log.yview)
        self.log.configure(yscrollcommand=yscroll.set)
        self.log.grid(row=0, column=0, sticky="nsew")
        yscroll.grid(row=0, column=1, sticky="ns")

        ttk.Label(right, text="RENDERED FILES", style="Muted.TLabel").grid(row=2, column=0, sticky="w", pady=(10, 5))
        resfrm = ttk.Frame(right, style="Card.TFrame", padding=8)
        resfrm.grid(row=3, column=0, sticky="nsew")
        resfrm.rowconfigure(0, weight=1)
        resfrm.columnconfigure(0, weight=1)
        self.results = tk.Listbox(
            resfrm,
            height=8,
            activestyle="none",
            bg=COLORS["entry"],
            fg=COLORS["text"],
            selectbackground="#183F4A",
            selectforeground=COLORS["cyan"],
            highlightthickness=1,
            highlightbackground=COLORS["ridge"],
            relief="flat",
            font=("Segoe UI", 10),
        )
        self.results.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        self.results.bind("<Double-Button-1>", lambda _e: self._play_selected())
        tip(self.results, "Двойной клик откроет MIDI или WAV системным приложением.")
        rb = ttk.Frame(resfrm, style="Card.TFrame")
        rb.grid(row=0, column=1, sticky="ns")
        tip(self._metal_button(rb, "Play", self._play_selected),
            "Открыть выбранный файл.").pack(fill="x")
        tip(self._metal_button(rb, "Open Dir", self._open_result_dir),
            "Открыть папку результатов.").pack(fill="x", pady=6)

    # ---------- progressive disclosure ----------
    def _show(self, widget: tk.Widget, visible: bool) -> None:
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
        win.title("Best-of stem sources")
        win.configure(bg=COLORS["bg"])
        win.transient(self.root)
        win.resizable(False, False)
        body = ttk.Frame(win, style="Shell.TFrame", padding=14)
        body.pack(fill="both", expand=True)
        sources = ["ft", "6s", "htdemucs", "cascade"]
        hint = {
            "vocals": "ft - самый чистый вокал",
            "bass": "htdemucs - плотнее низ",
            "drums": "htdemucs - чётче транзиенты",
            "guitar": "6s или cascade",
            "other": "6s по оригиналу",
            "piano": "cascade - меньше вокальных обертонов",
        }
        ttk.Label(body, text="Источник для каждого стема", style="Muted.TLabel").grid(
            row=0, column=0, columnspan=3, sticky="w", pady=(0, 8)
        )
        for i, stem in enumerate(("vocals", "bass", "drums", "guitar", "other", "piano"), 1):
            ttk.Label(body, text=stem + ":", style="Panel.TLabel").grid(row=i, column=0, sticky="e", padx=(0, 8), pady=3)
            vals = ["6s", "cascade"] if stem in ("guitar", "piano") else sources
            self._combo(body, self.bestof_plan[stem], vals, width=12).grid(
                row=i, column=1, sticky="w", padx=(0, 8), pady=3
            )
            ttk.Label(body, text=hint.get(stem, ""), style="Muted.TLabel").grid(row=i, column=2, sticky="w", pady=3)
        self._accent_button(body, "Done", win.destroy).grid(
            row=99, column=0, columnspan=3, pady=(12, 0)
        )
        win.grab_set()

    # ---------- drum only ----------
    def _run_drums_only(self) -> None:
        if not self.last_result_dir:
            out, audio = self.out_path.get().strip(), self.in_path.get().strip()
            if not out or not audio:
                messagebox.showinfo("Нет данных", "Сначала выполните полный прогон трека.")
                return
            self.last_result_dir = Path(out) / Path(audio).stem
        drums_wav = self.last_result_dir / "audio" / "drums.wav"
        if not drums_wav.is_file():
            messagebox.showerror(
                "Нет стема",
                f"Не найден {drums_wav}\nСначала выполните прогон с сохранением стемов.",
            )
            return
        cmd = [
            sys.executable,
            "-u",
            "-c",
            "import sys; from drum_transcribe import transcribe_drums; "
            "r=transcribe_drums(sys.argv[1], sys.argv[2], sensitivity=float(sys.argv[3]), "
            "bpm=(float(sys.argv[4]) if sys.argv[4] else None), grid_fill=(sys.argv[5]=='1'), "
            "cymbal_gate=float(sys.argv[6]), detect_toms=(sys.argv[7]=='1'), "
            "grid_offset_ms=float(sys.argv[8])); "
            "print('tempo ~%s BPM, %s hits' % (r.get('_tempo','?'), r.get('_total',0))); "
            "[print('  %-9s %s'%(k,v)) for k,v in sorted(r.items()) if not k.startswith('_')]",
            str(drums_wav),
            str(self.last_result_dir / "drums.mid"),
            f"{float(self.drum_sensitivity.get()):.2f}",
            self.bpm.get().strip(),
            "1" if self.drum_grid_fill.get() else "0",
            f"{float(self.cymbal_gate.get()):.2f}",
            "1" if self.detect_toms.get() else "0",
            str(int(self.grid_offset_ms.get())),
        ]
        self._launch(cmd, label="барабаны")

    # ---------- file pickers / playback ----------
    def _pick_input(self) -> None:
        p = filedialog.askopenfilename(
            title="Выберите аудиофайл",
            initialdir=str(HERE / "input"),
            filetypes=AUDIO_TYPES,
        )
        if p:
            self.in_path.set(p)

    def _pick_output(self) -> None:
        p = filedialog.askdirectory(title="Куда сохранять MIDI", initialdir=self.out_path.get() or str(HERE))
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
        name = self.results.get(sel[0]).strip()
        if name.startswith("--") or not name:
            return
        name = name.replace("MIDI  ", "").replace("WAV   ", "").strip()
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
            self.results.insert("end", "-- MIDI --")
            for m in mids:
                self.results.insert("end", f"MIDI  {m.name}")
        if wavs:
            self.results.insert("end", "-- AUDIO --")
            for w in wavs:
                self.results.insert("end", f"WAV   {w.name}")
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

        cmd = [
            sys.executable,
            "-u",
            str(PIPELINE),
            audio,
            "--out",
            out,
            "--device",
            self.device.get(),
            "--separation",
            self.separation.get(),
            "--model",
            self.model.get(),
            "--segment",
            str(self.segment.get()),
            "--save-audio" if self.save_audio.get() else "--no-save-audio",
            "--freq-bounds",
            self.freq_bounds.get(),
            "--piano-engine",
            self.piano_engine.get(),
            "--clean-octaves" if self.clean_octaves.get() else "--no-clean-octaves",
        ]
        if int(self.max_polyphony.get()) > 0:
            cmd += ["--max-polyphony", str(int(self.max_polyphony.get()))]
        if not self.adaptive_midi.get():
            cmd.append("--no-adaptive-midi")
        if not self.midi_preprocess.get():
            cmd.append("--no-midi-preprocess")
        if not self.smart_clean.get():
            cmd.append("--no-smart-clean")
        if self.separation.get() == "bestof":
            cmd += ["--bestof-plan", ",".join(f"{k}:{v.get()}" for k, v in self.bestof_plan.items())]
        if self.include_drums.get():
            cmd += [
                "--include-drums",
                "--drum-sensitivity",
                f"{float(self.drum_sensitivity.get()):.2f}",
                "--cymbal-gate",
                f"{float(self.cymbal_gate.get()):.2f}",
            ]
            if self.bpm.get().strip():
                cmd += ["--bpm", self.bpm.get().strip()]
            if self.drum_grid_fill.get():
                cmd.append("--drum-grid-fill")
            if self.detect_toms.get():
                cmd.append("--detect-toms")
            if int(self.grid_offset_ms.get()) != 0:
                cmd += ["--grid-offset-ms", str(int(self.grid_offset_ms.get()))]
        if self.tweak_thresholds.get():
            cmd += [
                "--onset",
                f"{float(self.onset.get()):.2f}",
                "--frame",
                f"{float(self.frame.get()):.2f}",
                "--min-note-ms",
                str(int(self.min_note_ms.get())),
            ]
        if self.key_filter.get() != "off":
            cmd += ["--key-filter", self.key_filter.get()]
            if self.key_filter.get() == "manual":
                cmd += ["--key", self.key.get().strip() or "C:maj"]
        mono = [s for s, v in self.mono_stems.items() if v.get()]
        if mono:
            cmd += ["--mono-stems", ",".join(mono)]
        split = [s for s, v in self.split_stems.items() if v.get()] if self.split_enable.get() else []
        if split:
            cmd += ["--split-stem", ",".join(split), "--split-method", self.split_method.get(), "--split-naming", self.split_naming.get()]
        self._launch(cmd, label="трек")

    def _launch(self, cmd: list[str], label: str = "") -> None:
        if self.proc and self.proc.poll() is None:
            messagebox.showinfo("Занято", "Дождитесь завершения текущей обработки.")
            return
        self.log.delete("1.0", "end")
        self.results.delete(0, "end")
        self._log("$ " + " ".join(f'"{c}"' if " " in c else c for c in cmd) + "\n\n")
        self.status.set(f"Обработка... ({label})" if label else "Обработка...")
        self.run_btn.config(state="disabled")
        self.cancel_btn.config(state="normal")
        self.progress.start(12)
        env = dict(os.environ)
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"

        def worker() -> None:
            try:
                self.proc = subprocess.Popen(
                    cmd,
                    cwd=str(HERE),
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                    creationflags=(
                        getattr(subprocess, "CREATE_NO_WINDOW", 0)
                        | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                    ),
                )
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
            self._terminate_process_tree(self.proc)
            self.status.set("Остановлено пользователем.")
            self._log("\n[остановлено]\n")

    def _terminate_process_tree(self, proc: subprocess.Popen) -> None:
        if proc.poll() is not None:
            return
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                check=False,
            )
            return
        proc.terminate()

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
            self.status.set("Готово. Двойной клик по файлу справа - прослушать.")
            self._populate_results()
        else:
            self.status.set(f"Завершено с ошибкой (код {code}). См. журнал.")

    def _on_close(self) -> None:
        if self.proc and self.proc.poll() is None:
            if not messagebox.askyesno("Идёт обработка", "Прервать и выйти?"):
                return
            self._terminate_process_tree(self.proc)
        self._save_settings()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
