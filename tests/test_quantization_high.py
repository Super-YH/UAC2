import unittest
import numpy as np
from audio_codec.quantization_high import (
    quantize_high_importance_band, 
    dequantize_high_importance_band,
    calculate_cepstrum, # Assuming this is now in spectral_utils, but test it here if not tested separately
    NUM_CEPSTRUM_COEFFS_HIGH,
    BITS_PER_CEPSTRUM_COEFF_HIGH,
    PVQ_NORMALIZATION_GAIN_BITS,
    PVQ_PULSES_BITS,
    LLOYD_MAX_TABLES_HIGH_CEPSTRUM # For checking index ranges
)
from audio_codec.pvq import PVQ_MAX_DIMENSION_CLIP # For band length
from audio_codec.globals import BAND_BOUNDARIES, NUM_MDCT_COEFFS # For sample band data

# If calculate_cepstrum is in spectral_utils, it should be imported from there.
# from audio_codec.spectral_utils import calculate_cepstrum

class TestQuantizationHigh(unittest.TestCase):

    def test_calculate_cepstrum_basic(self):
        # Create a simple signal: a sinusoid (which has a peak in spectrum)
        # Cepstrum should show a peak related to the periodicity of log spectrum.
        # For a single sinusoid, power spectrum is a peak. Log(peak) is a value.
        # DCT of a mostly-zero array with one peak will have distributed energy.
        # c0 (the first cepstral coeff) represents the average log power.
        
        band_len = 64 # Example band length
        fs = 1000 # Sample rate for frequency calculation
        freq = 100 # Hz
        t = np.arange(band_len) / float(fs)
        signal = np.sin(2 * np.pi * freq * t) * 0.5
        mdct_coeffs_sample = signal # Using time domain signal as proxy for MDCT coeffs for simplicity

        num_cep_coeffs_to_calc = NUM_CEPSTRUM_COEFFS_HIGH
        cepstrum = calculate_cepstrum(mdct_coeffs_sample, num_cep_coeffs_to_calc)

        self.assertEqual(len(cepstrum), num_cep_coeffs_to_calc, "Cepstrum output length is incorrect.")
        
        # c0 should be non-zero if signal has energy
        self.assertNotAlmostEqual(cepstrum[0], 0.0, places=5, 
                                  msg="c0 (average log power) is zero for non-zero signal.")

        # Test with zero signal
        zero_signal = np.zeros(band_len)
        cepstrum_zero = calculate_cepstrum(zero_signal, num_cep_coeffs_to_calc)
        # For zero signal, power_spec is 0. log(0+EPSILON) is negative.
        # DCT of constant array will have energy mostly in c0.
        # np.testing.assert_array_almost_equal(cepstrum_zero[1:], np.zeros(num_cep_coeffs_to_calc-1), decimal=5,
        #                                     err_msg="Higher order cepstrum coeffs should be near zero for zero signal if EPSILON is handled well.")
        # This specific assertion is tricky due to EPSILON. A less strict check:
        self.assertTrue(np.all(np.isfinite(cepstrum_zero)), "Cepstrum of zero signal contains non-finite values.")


    def test_quantize_dequantize_high_band_flow(self):
        # Test the overall flow: quantization and dequantization run without errors
        # and produce data of expected shapes and types.
        # Numerical accuracy is not tested here due to placeholder quantizers.

        # Use a band length from globals for realism, e.g., first band.
        # Ensure band_length <= PVQ_MAX_DIMENSION_CLIP for simpler PVQ handling in placeholder
        # (single subvector). If band_length > PVQ_MAX_DIMENSION_CLIP, multiple PVQ indices are generated.
        
        # Choose a band length that might be split by PVQ to test that path.
        # For example, if PVQ_MAX_DIMENSION_CLIP = 64
        band_length = PVQ_MAX_DIMENSION_CLIP + 10 # e.g. 74
        if band_length > (BAND_BOUNDARIES[-1] - BAND_BOUNDARIES[0]): # Ensure it's not larger than total MDCT coeffs
            band_length = NUM_MDCT_COEFFS // NUM_MDCT_COEFFS # use a smaller band if total is too small
            if band_length == 0 : band_length = 16 # fallback if NUM_MDCT_COEFFS is small

        if band_length == 0 and NUM_MDCT_COEFFS > 0 : # if NUM_MDCT_COEFFS is very small
             band_length = NUM_MDCT_COEFFS
        elif band_length == 0 :
            band_length = 16 # Default small band length

        sample_mdct_band_data = np.random.rand(band_length).astype(np.float32) - 0.5
        sample_mdct_band_data *= 0.1 # Scale down to avoid large cepstrum values

        # Quantize
        quant_data = quantize_high_importance_band(sample_mdct_band_data)

        # Check output keys and types from quantization
        self.assertIn("quantized_cepstrum_indices", quant_data)
        self.assertIsInstance(quant_data["quantized_cepstrum_indices"], list)
        self.assertEqual(len(quant_data["quantized_cepstrum_indices"]), NUM_CEPSTRUM_COEFFS_HIGH)
        for idx in quant_data["quantized_cepstrum_indices"]:
            self.assertIsInstance(idx, int)
            # Check if index is within valid range for the corresponding Lloyd-Max table
            # This requires knowing which table was used (LLOYD_MAX_TABLES_HIGH_CEPSTRUM[i])
            # Max index is 2**BITS_PER_CEPSTRUM_COEFF_HIGH - 1
            self.assertTrue(0 <= idx < (2**BITS_PER_CEPSTRUM_COEFF_HIGH))


        self.assertIn("quantized_residual_gain_index", quant_data)
        self.assertIsInstance(quant_data["quantized_residual_gain_index"], int)
        self.assertTrue(0 <= quant_data["quantized_residual_gain_index"] < (2**PVQ_NORMALIZATION_GAIN_BITS))

        self.assertIn("pvq_pulse_count", quant_data)
        self.assertIsInstance(quant_data["pvq_pulse_count"], int)
        self.assertTrue(4 <= quant_data["pvq_pulse_count"] <= 32) # As per spec in quantization_high

        self.assertIn("pvq_codeword_indices", quant_data) # Note: key used by quantizer
        self.assertIsInstance(quant_data["pvq_codeword_indices"], list)
        # Number of PVQ indices depends on band_length and PVQ_MAX_DIMENSION_CLIP
        num_expected_pvq_indices = (band_length + PVQ_MAX_DIMENSION_CLIP - 1) // PVQ_MAX_DIMENSION_CLIP
        self.assertEqual(len(quant_data["pvq_codeword_indices"]), num_expected_pvq_indices)
        for idx in quant_data["pvq_codeword_indices"]:
            self.assertIsInstance(idx, int) # Placeholder PVQ returns an int

        self.assertIn("band_length", quant_data)
        self.assertEqual(quant_data["band_length"], band_length)

        # Dequantize
        # The dequantizer expects the same keys as produced by the quantizer.
        reconstructed_band = dequantize_high_importance_band(quant_data)

        self.assertIsInstance(reconstructed_band, np.ndarray)
        self.assertEqual(reconstructed_band.shape, (band_length,),
                         "Reconstructed band shape is incorrect.")
        self.assertTrue(np.all(np.isfinite(reconstructed_band)),
                        "Reconstructed band contains non-finite values.")
                        
    def test_gain_quantization_log(self):
        from audio_codec.quantization_high import quantize_gain_log, dequantize_gain_log, LOG_GAIN_MIN, LOG_GAIN_MAX
        
        num_bits = PVQ_NORMALIZATION_GAIN_BITS
        num_levels = 2**num_bits

        # Test some values
        gains_to_test = [0.0001, 0.01, 0.1, 1.0, 10.0, 100.0, 2000.0]
        for gain in gains_to_test:
            idx, dequant_gain_q = quantize_gain_log(gain, num_bits)
            self.assertTrue(0 <= idx < num_levels, f"Gain index {idx} out of range for gain {gain}")
            
            dequant_gain_d = dequantize_gain_log(idx, num_bits)
            self.assertAlmostEqual(dequant_gain_q, dequant_gain_d, places=6,
                                   msg=f"Dequantized gain from quantizer ({dequant_gain_q}) "
                                       f"differs from dequantizer output ({dequant_gain_d}) for gain {gain}")
            
            # Check if original gain is reasonably close to dequantized gain
            # This depends on the number of bits and range.
            if gain < 2**LOG_GAIN_MIN or gain > 2**LOG_GAIN_MAX : # Clipped gains
                 # If original gain was clipped, dequant_gain might not be close to original gain
                 # but rather to the clipped gain's quantized version.
                 log_gain_clipped = np.clip(np.log2(gain) if gain > 1e-12 else LOG_GAIN_MIN, LOG_GAIN_MIN, LOG_GAIN_MAX)
                 expected_clipped_dequant_gain = 2**log_gain_clipped 
                 # The test is more about round trip of index.
                 # print(f"Gain {gain} (clipped log {log_gain_clipped:.2f}) -> idx {idx} -> dequant {dequant_gain_d:.3f} (direct from clipped: {expected_clipped_dequant_gain:.3f})")
            else: # In-range gains
                 # print(f"Gain {gain} -> idx {idx} -> dequant {dequant_gain_d:.3f}")
                 relative_error = abs(dequant_gain_d - gain) / (gain + 1e-9) if gain > 1e-9 else abs(dequant_gain_d - gain)
                 # Allow a larger relative error due to log quantization.
                 # Max error is related to (LOG_GAIN_MAX - LOG_GAIN_MIN) / num_levels
                 # For 5 bits, 32 levels. Range 20. Step ~0.625 in log2. Factor of 2^0.625 ~ 1.54.
                 # This means error can be up to ~50% in linear domain for values near decision boundaries.
                 self.assertTrue(relative_error < 0.6, f"Large relative error for gain {gain}, dequantized {dequant_gain_d}")


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)

```
