import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import torch
import torchaudio as ta

from scnet.wav import get_wav_datasets


def _write_wav(path, sample_rate=8000, channels=1, samples=8000):
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = torch.zeros(channels, samples, dtype=torch.float32)
    ta.save(str(path), audio, sample_rate)


def _make_dataset(root, sources, sample_rate=8000, channels=1):
    for split in ["train", "test"]:
        track_dir = Path(root) / split / "track1"
        for name in sources + ["mixture"]:
            _write_wav(track_dir / f"{name}.wav", sample_rate, channels)


class TestWavDataset(unittest.TestCase):
    def test_get_wav_datasets_uses_test_dir(self):
        with TemporaryDirectory() as tmp:
            data_root = Path(tmp) / "wav"
            metadata_root = Path(tmp) / "metadata"
            sources = ["s1"]
            _make_dataset(data_root, sources)
            args = SimpleNamespace(
                wav=str(data_root),
                metadata=str(metadata_root),
                sources=sources,
                segment=None,
                shift=None,
                samplerate=8000,
                channels=1,
                normalize=False,
            )

            train_set, valid_set = get_wav_datasets(args)

            self.assertEqual(valid_set.root, Path(args.wav) / "test")
            self.assertEqual(len(train_set), 1)
            self.assertEqual(len(valid_set), 1)

    def test_missing_file_raises(self):
        with TemporaryDirectory() as tmp:
            data_root = Path(tmp) / "wav"
            metadata_root = Path(tmp) / "metadata"
            sources = ["s1"]
            _make_dataset(data_root, sources)
            args = SimpleNamespace(
                wav=str(data_root),
                metadata=str(metadata_root),
                sources=sources,
                segment=None,
                shift=None,
                samplerate=8000,
                channels=1,
                normalize=False,
            )

            _, valid_set = get_wav_datasets(args)
            missing = data_root / "test" / "track1" / "mixture.wav"
            missing.unlink()

            with self.assertRaises(FileNotFoundError):
                _ = valid_set[0]


if __name__ == "__main__":
    unittest.main()
