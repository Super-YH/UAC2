import unittest
import numpy as np
from audio_codec.quantization_medium import (
    quantize_medium_importance_band, 
    dequantize_medium_importance_band,
    NUM_CEPSTRUM_COEFFS_MEDIUM,
    BITS_PER_CEPSTRUM_COEFF_MEDIUM,
    LLOYD_MAX_TABLES_MEDIUM_CEPSTRUM # For checking index ranges
)
from audio_codec.globals import BAND_BOUNDARIES # For sample band data

# calculate_cepstrum is typically in spectral_utils and tested there or via quantization_high.
# from audio_codec.spectral_utils import calculate_cepstrum 

class TestQuantizationMedium(unittest.TestCase):

    def test_quantize_dequantize_medium_band_flow(self):
        # Test the overall flow: quantization and dequantization run without errors
        # and produce data of expected shapes and types.
        # Numerical accuracy is not tested here due to placeholder quantizers.

        # Use a band length from globals for realism, e.g., a middle band.
        # Band 5: BAND_BOUNDARIES[5] to BAND_BOUNDARIES[6]
        if NUM_CEPSTRUM_COEFFS_MEDIUM > 0 and len(BAND_BOUNDARIES) > 6 : # Ensure band 5 exists
            band_length = BAND_BOUNDARIES[6] - BAND_BOUNDARIES[5]
        else:
            band_length = 32 # fallback length
        
        if band_length <=0 : band_length = 16 # ensure positive length

        sample_mdct_band_data = np.random.rand(band_length).astype(np.float32) - 0.5
        sample_mdct_band_data *= 0.05 # Scale down for cepstrum stability

        # Quantize
        quant_data = quantize_medium_importance_band(sample_mdct_band_data)

        # Check output keys and types from quantization
        self.assertIn("quantized_cepstrum_indices", quant_data)
        self.assertIsInstance(quant_data["quantized_cepstrum_indices"], list)
        
        # Check if the number of cepstrum coefficients is as expected.
        # quantize_medium_importance_band pads cepstrum_coeffs if actual calc yields fewer.
        self.assertEqual(len(quant_data["quantized_cepstrum_indices"]), NUM_CEPSTRUM_COEFFS_MEDIUM)

        for i, idx in enumerate(quant_data["quantized_cepstrum_indices"]):
            self.assertIsInstance(idx, int)
            # Check if index is within valid range for the corresponding Lloyd-Max table
            # Max index is 2**BITS_PER_CEPSTRUM_COEFF_MEDIUM - 1
            self.assertTrue(0 <= idx < (2**BITS_PER_CEPSTRUM_COEFF_MEDIUM),
                            f"Cepstrum index {idx} for coeff {i} is out of range.")

        self.assertIn("band_length", quant_data)
        self.assertEqual(quant_data["band_length"], band_length)

        # Dequantize
        # The dequantizer expects the same keys as produced by the quantizer.
        reconstructed_band = dequantize_medium_importance_band(quant_data)

        self.assertIsInstance(reconstructed_band, np.ndarray)
        self.assertEqual(reconstructed_band.shape, (band_length,),
                         "Reconstructed band shape is incorrect.")
        self.assertTrue(np.all(np.isfinite(reconstructed_band)),
                        "Reconstructed band contains non-finite values.")
        
        # Test reconstruction for zero input band
        zero_band_data = np.zeros(band_length)
        quant_data_zero = quantize_medium_importance_band(zero_band_data)
        reconstructed_zero_band = dequantize_medium_importance_band(quant_data_zero)
        
        # For zero input, cepstrum of (EPSILON) will be calculated.
        # Reconstructed envelope might not be exactly zero but should be small.
        # The phase is zero, so it's non-negative.
        self.assertTrue(np.all(reconstructed_zero_band >= 0),
                        "Reconstructed zero band should be non-negative (zero phase).")
        # Max value should be small
        self.assertTrue(np.max(np.abs(reconstructed_zero_band)) < 1e-3, # Heuristic threshold
                        "Reconstructed zero band has unexpectedly large values.")


    def test_quantize_medium_empty_band(self):
        empty_band = np.array([])
        with self.assertRaises(ValueError, 
                               msg="quantize_medium_importance_band should raise ValueError for empty input."):
            quantize_medium_importance_band(empty_band)

    def test_dequantize_medium_invalid_data(self):
        # Missing keys
        with self.assertRaises(KeyError):
            dequantize_medium_importance_band({"band_length": 16}) # Missing cepstrum indices
        
        # Incorrect number of cepstrum indices
        quant_data_bad_cep_count = {
            "quantized_cepstrum_indices": [0] * (NUM_CEPSTRUM_COEFFS_MEDIUM - 1),
            "band_length": 16
        }
        with self.assertRaises(ValueError): # Or TypeError depending on internal checks
            dequantize_medium_importance_band(quant_data_bad_cep_count)

        # Band length zero
        quant_data_zero_len = {
            "quantized_cepstrum_indices": [0] * NUM_CEPSTRUM_COEFFS_MEDIUM,
            "band_length": 0
        }
        # Dequantizer should return empty array or handle gracefully.
        # Current implementation returns np.array([])
        reconstructed = dequantize_medium_importance_band(quant_data_zero_len)
        self.assertEqual(len(reconstructed), 0, "Dequantizing zero length band should result in empty array.")


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
```
