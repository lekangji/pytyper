import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from experiments.data import SPLIT_SEED, source_hash
from experiments.full_corpus import train
from experiments.full_corpus_data import prepare


class FullCorpusCacheTests(unittest.TestCase):
    def test_prepared_cache_rejects_changed_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "Keystrokes.zip"
            archive.write_bytes(b"original")
            torch.save({"split_seed": SPLIT_SEED, "source_sha256": {archive.name: source_hash(archive)}},
                       root / "manifest.pt")
            archive.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "archive or split"):
                prepare(archive, root, workers=1)

    def test_completed_run_rejects_changed_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "gelu-16d.json").write_text(json.dumps({"source_sha256": {"archive": "previous"}}),
                                               encoding="utf-8")
            manifest = {"source_sha256": {"archive": "updated"}, "split_seed": SPLIT_SEED}
            with patch("experiments.full_corpus.WORK", root):
                with self.assertRaisesRegex(ValueError, "Run inputs changed"):
                    train("gelu-16d", manifest, {}, torch.device("cpu"), amp=False)
