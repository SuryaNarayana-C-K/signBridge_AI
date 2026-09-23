"""
Unit tests for custom_signs.py module.
Tests:
- Landmark normalization
- Cosine similarity and Euclidean distance
- Adding custom sign
- Rejection of duplicate or reserved built-in sign names
- Persistence and loading
- Matching live features against custom signs
- Deleting custom sign
"""

import os
import shutil
import tempfile
import unittest
import numpy as np

import custom_signs
from custom_signs import (
    CustomSignStorage,
    normalize_landmarks,
    cosine_similarity,
    euclidean_distance,
    calculate_centroid,
    match_custom_sign,
    validate_sign_name
)


class TestCustomSigns(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.test_file = os.path.join(self.test_dir, "test_custom_signs.json")
        self.storage = CustomSignStorage(self.test_file)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_normalization(self):
        # Create synthetic 21 landmarks
        synthetic_hand = [{"x": float(i) * 0.05, "y": float(i) * 0.04, "z": 0.01} for i in range(21)]
        feat = normalize_landmarks([synthetic_hand])
        self.assertIsNotNone(feat)
        self.assertEqual(len(feat), 63)  # 21 * 3
        # Wrist coordinate (index 0, 1, 2) should be [0.0, 0.0, 0.0]
        self.assertAlmostEqual(feat[0], 0.0, places=5)
        self.assertAlmostEqual(feat[1], 0.0, places=5)

    def test_similarity_metrics(self):
        v1 = [1.0, 2.0, 3.0, 4.0]
        v2 = [1.0, 2.0, 3.0, 4.0]
        v3 = [-1.0, -2.0, -3.0, -4.0]
        self.assertAlmostEqual(cosine_similarity(v1, v2), 1.0, places=5)
        self.assertAlmostEqual(cosine_similarity(v1, v3), -1.0, places=5)
        self.assertAlmostEqual(euclidean_distance(v1, v2), 0.0, places=5)

    def test_add_and_load_custom_sign(self):
        # 25 synthetic samples with small random noise
        base = [float(i) * 0.1 for i in range(63)]
        samples = [[v + np.random.normal(0, 0.01) for v in base] for _ in range(25)]

        ok, msg, record = self.storage.add_sign(
            name="good_morning",
            meaning="Good morning",
            samples=samples,
            num_hands=1
        )
        self.assertTrue(ok, msg)
        self.assertIsNotNone(record)
        self.assertEqual(record["name"], "GOOD_MORNING")
        self.assertEqual(record["meaning"], "Good morning")
        self.assertEqual(record["sample_count"], 25)

        # Load back
        all_signs = self.storage.load_all()
        self.assertEqual(len(all_signs), 1)
        self.assertEqual(all_signs[0]["name"], "GOOD_MORNING")

    def test_reserved_name_rejection(self):
        samples = [[0.1] * 63 for _ in range(10)]
        ok, msg, _ = self.storage.add_sign("HELLO", "Greeting", samples)
        self.assertFalse(ok)
        self.assertIn("already exists", msg)

        is_valid, vmsg = validate_sign_name("HELLO")
        self.assertFalse(is_valid)

        is_valid2, _ = validate_sign_name("MY_CUSTOM_SIGN")
        self.assertTrue(is_valid2)

    def test_matching_above_and_below_threshold(self):
        base = [float(i) * 0.1 for i in range(63)]
        samples = [[v + np.random.normal(0, 0.005) for v in base] for _ in range(20)]

        self.storage.add_sign("PEACE_CUSTOM", "Peace to everyone", samples)
        signs = self.storage.load_all()

        # Frame very close to base
        live_close = [v + np.random.normal(0, 0.005) for v in base]
        match = match_custom_sign(live_close, signs, threshold=0.85)
        self.assertIsNotNone(match)
        self.assertEqual(match["name"], "PEACE_CUSTOM")
        self.assertGreaterEqual(match["confidence"], 0.85)

        # Completely different frame
        live_different = [-v for v in base]
        no_match = match_custom_sign(live_different, signs, threshold=0.85)
        self.assertIsNone(no_match)

    def test_delete_custom_sign(self):
        samples = [[0.1] * 63 for _ in range(10)]
        ok, _, record = self.storage.add_sign("TEST_SIGN", "Test Meaning", samples)
        self.assertTrue(ok)
        self.assertEqual(len(self.storage.load_all()), 1)

        del_ok, _ = self.storage.delete_sign(record["id"])
        self.assertTrue(del_ok)
        self.assertEqual(len(self.storage.load_all()), 0)


if __name__ == "__main__":
    unittest.main()
