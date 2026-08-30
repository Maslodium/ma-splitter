from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pretty_midi
import soundfile as sf

import pipeline


class PipelineSmokeTests(unittest.TestCase):
    def test_parse_key_accepts_common_labels(self) -> None:
        self.assertEqual(pipeline.parse_key("G:min"), (7, "min"))
        self.assertEqual(pipeline.parse_key("Bb:maj"), (10, "maj"))
        self.assertEqual(pipeline.parse_key("Am"), (9, "min"))

    def test_clean_midi_removes_octave_ghost(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            midi_path = Path(td) / "ghost.mid"
            pm = pretty_midi.PrettyMIDI()
            inst = pretty_midi.Instrument(program=0)
            inst.notes.append(pretty_midi.Note(velocity=90, pitch=60, start=0.0, end=1.0))
            inst.notes.append(pretty_midi.Note(velocity=50, pitch=72, start=0.01, end=0.8))
            pm.instruments.append(inst)
            pm.write(str(midi_path))

            kept, removed = pipeline.clean_midi(midi_path, drop_octave_ghosts=True)

            self.assertEqual((kept, removed), (1, 1))
            cleaned = pretty_midi.PrettyMIDI(str(midi_path))
            self.assertEqual(cleaned.instruments[0].notes[0].pitch, 60)

    def test_clean_midi_polyphony_cap_keeps_longest_notes(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            midi_path = Path(td) / "poly.mid"
            pm = pretty_midi.PrettyMIDI()
            inst = pretty_midi.Instrument(program=0)
            inst.notes.append(pretty_midi.Note(velocity=90, pitch=60, start=0.0, end=1.0))
            inst.notes.append(pretty_midi.Note(velocity=90, pitch=64, start=0.0, end=0.2))
            inst.notes.append(pretty_midi.Note(velocity=90, pitch=67, start=0.0, end=0.9))
            pm.instruments.append(inst)
            pm.write(str(midi_path))

            kept, removed = pipeline.clean_midi(
                midi_path,
                drop_octave_ghosts=False,
                max_polyphony=2,
            )

            self.assertEqual((kept, removed), (2, 1))
            cleaned = pretty_midi.PrettyMIDI(str(midi_path))
            pitches = sorted(n.pitch for n in cleaned.instruments[0].notes)
            self.assertEqual(pitches, [60, 67])

    def test_smart_clean_merges_gaps_and_drops_quiet_specks(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            midi_path = Path(td) / "smart.mid"
            pm = pretty_midi.PrettyMIDI()
            inst = pretty_midi.Instrument(program=0)
            inst.notes.append(pretty_midi.Note(velocity=80, pitch=60, start=0.0, end=0.20))
            inst.notes.append(pretty_midi.Note(velocity=82, pitch=60, start=0.22, end=0.45))
            inst.notes.append(pretty_midi.Note(velocity=20, pitch=73, start=1.0, end=1.02))
            pm.instruments.append(inst)
            pm.write(str(midi_path))

            stats = pipeline.clean_midi_smart(
                midi_path,
                "guitar",
                {"min_len_ms": 90},
            )

            self.assertEqual(stats["merged"], 1)
            self.assertGreaterEqual(stats["removed"], 1)
            cleaned = pretty_midi.PrettyMIDI(str(midi_path))
            notes = cleaned.instruments[0].notes
            self.assertEqual(len(notes), 1)
            self.assertAlmostEqual(notes[0].end, 0.45)

    def test_analyze_and_preprocess_stem_writes_helper_wav(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            wav = Path(td) / "sine.wav"
            out = Path(td) / "sine.prep.wav"
            sr = 44100
            t = np.linspace(0, 0.25, int(sr * 0.25), endpoint=False)
            y = 0.25 * np.sin(2 * np.pi * 220 * t)
            sf.write(str(wav), y.astype(np.float32), sr)

            metrics = pipeline.analyze_stem(wav)
            profile = pipeline.choose_adaptive_profile(
                "bass",
                dict(pipeline.PART_PROFILES["bass"]),
                metrics,
            )
            result = pipeline.preprocess_stem_for_midi(wav, out, "bass", profile, metrics)

            self.assertEqual(result, out)
            self.assertTrue(out.is_file())
            self.assertGreater(metrics["rms"], 0)


if __name__ == "__main__":
    unittest.main()
