import unittest

import numpy as np

from src.metrics.hd95 import hd95_mm


class HD95Tests(unittest.TestCase):
    def test_identical_and_empty_masks(self):
        mask = np.ones((4, 4, 4), dtype=bool)
        empty = np.zeros_like(mask)
        affine = np.eye(4)
        self.assertEqual(hd95_mm(mask, mask, affine, affine), 0)
        self.assertEqual(hd95_mm(empty, empty, affine, affine), 0)
        self.assertTrue(np.isinf(hd95_mm(mask, empty, affine, affine)))
        self.assertTrue(np.isinf(hd95_mm(empty, mask, affine, affine)))

    def test_anisotropic_spacing_and_origin(self):
        pred = np.zeros((5, 5, 5), dtype=bool)
        ref = pred.copy()
        pred[1, 2, 2] = True
        ref[2, 2, 2] = True
        affine = np.diag([2.5, 1, 0.5, 1])
        self.assertEqual(hd95_mm(pred, ref, affine, affine), 2.5)
        shifted = affine.copy()
        shifted[0, 3] = 2.5
        self.assertEqual(hd95_mm(pred, ref, shifted, affine), 0)

    def test_percentile_rejects_sparse_outlier_and_is_symmetric(self):
        ref = np.zeros((40, 40, 40), dtype=bool)
        ref[2:12, 2:12, 2:12] = True
        pred = ref.copy()
        pred[35, 35, 35] = True
        affine = np.eye(4)
        self.assertEqual(hd95_mm(pred, ref, affine, affine), 0)
        self.assertEqual(hd95_mm(ref, pred, affine, affine), 0)


if __name__ == "__main__":
    unittest.main()
