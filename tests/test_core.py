import importlib.util
import unittest


class CoreAvailabilityTests(unittest.TestCase):
    def test_training_core_is_importable_without_comfyui(self):
        self.assertIsNotNone(importlib.util.find_spec("krea2_slider_node"))


if __name__ == "__main__":
    unittest.main()
