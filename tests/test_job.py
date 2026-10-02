from contextlib import nullcontext
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from krea2_slider_node import job
from krea2_slider_node.config import ModelConfig, TrainConfig
from krea2_slider_node.job import run_training_job


class JobTests(unittest.TestCase):
    def test_gpu_cleanup_failure_releases_job_lock_for_next_attempt(self):
        lock = threading.Lock()
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(job, "_job_lock", lock), \
                patch.object(job, "MemoryBudget", side_effect=lambda *args: nullcontext()), \
                patch.object(job, "probe_training_operators", return_value={}), \
                patch("torch.cuda.empty_cache", side_effect=RuntimeError("GPU cleanup failed")):
            for _ in range(2):
                with self.assertRaisesRegex(RuntimeError, "GPU cleanup failed"):
                    run_training_job(ModelConfig(str(Path(directory) / "missing.safetensors")), [],
                                     TrainConfig(), directory, "slider", device="cuda")
                self.assertFalse(lock.locked())

    def test_invalid_output_name_is_rejected_before_checkpoint_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            # A bad output name must win over the missing checkpoint error.
            with self.assertRaisesRegex(ValueError, "filename"):
                run_training_job(ModelConfig(str(Path(directory) / "missing.safetensors")), [],
                                 TrainConfig(), directory, "../escape", device="cpu")

    def test_failed_load_releases_job_lock_for_next_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            for _ in range(2):
                with self.assertRaises(RuntimeError) as caught:
                    run_training_job(ModelConfig(str(Path(directory) / "missing.safetensors")), [],
                                     TrainConfig(), directory, "slider", device="cpu")
                self.assertNotIn("already running", str(caught.exception))
