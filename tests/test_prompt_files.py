import hashlib
import importlib
import json
from pathlib import Path
import re
import tempfile
import unittest

import yaml


PROMPTS_ROOT = Path(__file__).resolve().parents[1] / "prompts"
SLIDER_ROLES = ("target", "positive", "negative", "neutral")


def male_text(text):
    """Change gender terms only; keep the concept, clothing and scene fixed."""
    words = {"woman": "man", "girl": "boy", "She": "He", "she": "he", "Her": "His", "her": "his"}
    text = re.sub(r"\b(woman|girl|She|she|Her|her)\b", lambda match: words[match[0]], text)
    return text.replace("shows his from", "shows him from")


def read_preset(name):
    return yaml.safe_load((PROMPTS_ROOT / name).read_text(encoding="utf-8"))


class PromptFileTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec("krea2_slider_node.prompt_files"))
        return importlib.import_module("krea2_slider_node.prompt_files")

    def test_bundled_yaml_choices_are_readable_by_the_training_schema(self):
        api = self.api()
        root = PROMPTS_ROOT
        names = api.list_prompt_files(root)
        self.assertEqual(names, ["aging_slider_fullbody.yaml", "aging_slider_fullbody_male.yaml",
                                 "breast_size_slider.yaml", "breast_size_slider_v2.yaml",
                                 "deaging_slider_fullbody.yaml", "deaging_slider_fullbody_male.yaml"])
        for name in names:
            records = api.load_prompt_file(root, name)
            self.assertEqual(len(records), 6)
            self.assertTrue(all(r["positive"] != r["negative"] for r in records))

    def test_original_slider_text_is_unchanged_when_anchors_are_added(self):
        # Canonical four-role snapshots from the pre-anchor presets. Adding an
        # anchor must not weaken or silently rewrite an existing concept pair.
        expected = {
            "aging_slider_fullbody.yaml": "9cf796bda11aa85d4b40cd9e7c1a1d010faf15343df73fee9e99e17d2da315b4",
            "breast_size_slider.yaml": "03bb808b69601029b83ad7169e0e85278618446cce4f9a0308da674d22ca8fe0",
            "breast_size_slider_v2.yaml": "eef7a922cb16a218c11f0732c728d533b3f7838cd2a889fb8373e1eb25da70c4",
            "deaging_slider_fullbody.yaml": "aa56cbe3a295fa23384cb0340ba53caabb80ce53d8dc5ebc426c51d7904ae6c1",
        }
        for name, digest in expected.items():
            with self.subTest(preset=name):
                rows = [{role: record[role] for role in SLIDER_ROLES} for record in read_preset(name)]
                payload = json.dumps(rows, sort_keys=True, ensure_ascii=False).encode()
                self.assertEqual(hashlib.sha256(payload).hexdigest(), digest)

    def test_female_age_presets_anchor_the_matching_adult_male_baseline(self):
        for name in ("aging_slider_fullbody.yaml", "deaging_slider_fullbody.yaml"):
            for index, record in enumerate(read_preset(name)):
                with self.subTest(preset=name, record=index):
                    self.assertIn("anchor", record)
                    self.assertEqual(record["anchor"], male_text(record["target"]))
                    self.assertIn("adult man", record["anchor"])
                    self.assertNotIn("toddler", record["anchor"])

    def test_male_age_presets_keep_the_concept_and_anchor_the_female_baseline(self):
        for stem in ("aging_slider_fullbody", "deaging_slider_fullbody"):
            name = stem + "_male.yaml"
            with self.subTest(preset=name):
                self.assertTrue((PROMPTS_ROOT / name).is_file(), f"Missing male preset: {name}")
                male = read_preset(name)
                female = read_preset(stem + ".yaml")
                self.assertEqual(len(male), 6)
                for index, (record, original) in enumerate(zip(male, female)):
                    with self.subTest(record=index):
                        self.assertEqual(set(record), {*SLIDER_ROLES, "anchor"})
                        for role in SLIDER_ROLES:
                            self.assertEqual(record[role], male_text(original[role]))
                        self.assertEqual(record["anchor"], original["target"])
                        self.assertIn("adult woman", record["anchor"])

    def test_deaging_keeps_the_fully_clothed_toddler_goal_for_each_gender(self):
        for suffix, child in (("", "girl"), ("_male", "boy")):
            name = f"deaging_slider_fullbody{suffix}.yaml"
            self.assertTrue((PROMPTS_ROOT / name).is_file(), f"Missing deaging preset: {name}")
            for index, record in enumerate(read_preset(name)):
                with self.subTest(preset=name, record=index):
                    positive = record["positive"]
                    self.assertIn(f"a toddler {child}", positive)
                    self.assertIn("natural toddler proportions", positive)
                    self.assertIn("fully clothed", positive)
                    self.assertIn("nonsexual composition", positive)
                    self.assertIn("clothes are made of opaque matte fabric", positive)
                    self.assertIn("adult", record["negative"])
                    self.assertNotIn("toddler", record["negative"])

    def test_breast_v2_preserves_small_negative_contrast_and_adds_male_anchors(self):
        moderate = (
            "He has medium-sized breasts with moderate bust volume. The opaque shirt follows a moderate rounded "
            "bust contour that projects a short distance forward from the ribcage."
        )
        male_chest = (
            "He has a natural adult male chest. The opaque shirt follows his ordinary chest contour."
        )
        for index, record in enumerate(read_preset("breast_size_slider_v2.yaml")):
            with self.subTest(record=index):
                self.assertIn("anchor", record)
                self.assertEqual(record["anchor"], male_text(record["target"]).replace(moderate, male_chest))
                self.assertIn("adult man", record["anchor"])
                self.assertIn("fully clothed", record["anchor"])
                self.assertNotIn("breasts", record["anchor"])
                self.assertNotIn("bust", record["anchor"])
                self.assertIn("small breasts with very low bust volume", record["negative"])
                self.assertNotEqual(record["target"], record["negative"])
                self.assertEqual(record["target"], record["neutral"])

    def test_breast_v1_stays_byte_for_byte_compatible_without_anchors(self):
        name = "breast_size_slider.yaml"
        self.assertEqual(hashlib.sha256((PROMPTS_ROOT / name).read_bytes()).hexdigest(),
                         "8e06fd501e4448ba223a3fc6a7ca64e5f5ac86995c2c65313138bf87d453aa4a")
        for record in read_preset(name):
            self.assertNotIn("anchor", record)
            self.assertEqual(record["target"], record["negative"])
            self.assertEqual(record["target"], record["neutral"])

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
