import unittest
import numpy as np
from audio_codec.quant_utils import quantize_scalar, dequantize_scalar, EPSILON # Assuming EPSILON might be relevant

class TestQuantUtils(unittest.TestCase):

    def test_quantize_scalar_and_dequantize_scalar(self):
        # Define a simple quantization levels table
        # Levels: 0.0, 0.5, 1.0, 1.5, 2.0
        levels_table = np.array([0.0, 0.5, 1.0, 1.5, 2.0])

        # Test cases: (input_value, expected_index, expected_quantized_value)
        test_values = [
            (0.1, 0, 0.0),  # Closest to 0.0
            (0.2, 0, 0.0),  # Still closer to 0.0 than 0.5 (midpoint 0.25)
            (0.26, 1, 0.5), # Closer to 0.5
            (0.45, 1, 0.5), # Closer to 0.5
            (0.70, 1, 0.5), # Closer to 0.5 than 1.0 (midpoint 0.75)
            (0.76, 2, 1.0), # Closer to 1.0
            (1.0, 2, 1.0),  # Exact match
            (1.4, 3, 1.5),
            (1.76, 4, 2.0), # Closer to 2.0
            (2.1, 4, 2.0),  # Beyond max, closest to 2.0
            (-0.1, 0, 0.0)  # Below min, closest to 0.0
        ]

        for value, expected_idx, expected_q_val in test_values:
            # Test quantize_scalar
            q_idx, q_val = quantize_scalar(value, levels_table)
            self.assertEqual(q_idx, expected_idx,
                             f"quantize_scalar failed for value {value}: expected index {expected_idx}, got {q_idx}")
            self.assertAlmostEqual(q_val, expected_q_val, places=6,
                                   msg=f"quantize_scalar failed for value {value}: expected q_value {expected_q_val}, got {q_val}")

            # Test dequantize_scalar using the obtained index
            deq_val = dequantize_scalar(q_idx, levels_table)
            self.assertAlmostEqual(deq_val, expected_q_val, places=6, # Should reconstruct the quantized level
                                   msg=f"dequantize_scalar failed for index {q_idx} (from value {value}): expected {expected_q_val}, got {deq_val}")

        # Test dequantize_scalar with direct indices
        for i in range(len(levels_table)):
            self.assertAlmostEqual(dequantize_scalar(i, levels_table), levels_table[i], places=6,
                                   msg=f"dequantize_scalar failed for direct index {i}")


    def test_quantize_scalar_edge_cases(self):
        levels_table_single = np.array([10.0]) # Single level quantizer
        
        # Test with single level table
        idx, val = quantize_scalar(5.0, levels_table_single)
        self.assertEqual(idx, 0, "Single level quantizer: index should be 0.")
        self.assertEqual(val, 10.0, "Single level quantizer: value should be the single level.")
        
        idx, val = quantize_scalar(15.0, levels_table_single)
        self.assertEqual(idx, 0)
        self.assertEqual(val, 10.0)

        deq_val = dequantize_scalar(0, levels_table_single)
        self.assertEqual(deq_val, 10.0, "Dequantize single level failed.")

        # Test with empty levels table (should raise error)
        empty_table = np.array([])
        with self.assertRaises(ValueError, msg="quantize_scalar should raise ValueError for empty table."):
            quantize_scalar(5.0, empty_table)
        
        # Test dequantize_scalar with invalid index
        levels_table = np.array([0.0, 0.5, 1.0])
        with self.assertRaises(ValueError, msg="dequantize_scalar should raise ValueError for out-of-bounds index."):
            dequantize_scalar(-1, levels_table)
        with self.assertRaises(ValueError, msg="dequantize_scalar should raise ValueError for out-of-bounds index."):
            dequantize_scalar(len(levels_table), levels_table)
        
        # Test input types
        with self.assertRaises(ValueError, msg="quantize_scalar should raise TypeError for non-array table."):
            quantize_scalar(1.0, [0.0, 1.0, 2.0]) # Must be numpy array
        with self.assertRaises(ValueError, msg="dequantize_scalar should raise TypeError for non-array table."):
            dequantize_scalar(0, [0.0, 1.0, 2.0])


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
```
