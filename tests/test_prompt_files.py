import importlib
from pathlib import Path
import tempfile
import unittest


class PromptFileTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec("krea2_slider_node.prompt_files"))
        return importlib.import_module("krea2_slider_node.prompt_files")

    def test_bundled_yaml_choices_are_readable_by_the_training_schema(self):
        api = self.api()
        root = Path(__file__).resolve().parents[1] / "prompts"
        names = api.list_prompt_files(root)
        self.assertEqual(names, ["aging_slider_fullbody.yaml", "breast_size_slider.yaml", "deaging_slider_fullbody.yaml"])
        for name in names:
            records = api.load_prompt_file(root, name)
            self.assertEqual(len(records), 6)
            self.assertTrue(all(r["positive"] != r["negative"] for r in records))

    def test_yaml_blocks_anchors_and_changed_content_are_supported(self):
        api = self.api()
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / "test.yaml"
            p.write_text('- target: &base >-\n    a portrait of\n    a person\n  positive: smiling person\n  negative: *base\n', encoding="utf-8")
            records = api.load_prompt_file(directory, "test.yaml")
            self.assertEqual(records[0]["target"], "a portrait of a person")
            self.assertEqual(records[0]["negative"], records[0]["target"])
            before = api.prompt_file_fingerprint(directory, "test.yaml")
            p.write_text(p.read_text().replace("smiling", "frowning"), encoding="utf-8")
            self.assertNotEqual(before, api.prompt_file_fingerprint(directory, "test.yaml"))

    def test_traversal_unsafe_yaml_and_invalid_records_are_rejected(self):
        api = self.api()
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                api.load_prompt_file(directory, "../outside.yaml")
            p = Path(directory) / "bad.yaml"
            for text in ['!!python/object/apply:os.system [echo unsafe]', '[{target: person}]']:
                p.write_text(text, encoding="utf-8")
                with self.assertRaises(ValueError):
                    api.load_prompt_file(directory, "bad.yaml")
