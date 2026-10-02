"""Regression: seeded legacy-Keras initializers must run under Python 3.12."""

import unittest
import numpy as np

from building_damage.model import framework, seed_framework


class SeedCompatibility(unittest.TestCase):
    def test_seeded_conv_initializer_and_inference(self):
        tf, _ = framework()
        seed_framework(tf, 42)
        layer = tf.keras.layers.Conv2D(2, 3)
        result = layer(np.zeros((1, 8, 8, 3), dtype=np.float32))
        self.assertEqual(tuple(result.shape), (1, 6, 6, 2))
        self.assertTrue(np.isfinite(result.numpy()).all())


if __name__ == "__main__":
    unittest.main()
