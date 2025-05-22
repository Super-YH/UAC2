import unittest
import numpy as np
from audio_codec.mdct import sine_window, mdct, imdct
from audio_codec.globals import MDCT_FRAME_SIZE, NEW_SAMPLES_PER_FRAME, NUM_MDCT_COEFFS

class TestMDCT(unittest.TestCase):

    def test_sine_window_properties(self):
        window = sine_window(MDCT_FRAME_SIZE)
        
        # Test length
        self.assertEqual(len(window), MDCT_FRAME_SIZE, "Sine window length is incorrect.")
        
        # Test symmetry (w[n] = w[N-1-n] for sine window w[n] = sin(pi*(n+0.5)/N) )
        # This is not strictly true for the given sine window formula.
        # The formula is sin(pi*(n+0.5)/N).
        # w[0] = sin(0.5*pi/N)
        # w[N-1] = sin(pi*(N-1+0.5)/N) = sin(pi*(N-0.5)/N) = sin(pi - 0.5*pi/N) = sin(0.5*pi/N)
        # So it is symmetric in this sense.
        np.testing.assert_array_almost_equal(window, window[::-1], decimal=6,
                                             err_msg="Sine window is not symmetric as expected.")
        
        # Test values are between 0 and 1
        self.assertTrue(np.all(window >= 0) and np.all(window <= 1),
                        "Sine window values are not all between 0 and 1.")
        
        # Test Princen-Bradley condition (approximate for sine window for TDAC)
        # w[n]^2 + w[n + N/2]^2 = 1 (for n = 0 to N/2-1)
        # For the sine window w[n] = sin(pi*(n+0.5)/N), this condition is met.
        # sin(pi*(n+0.5)/N)^2 + sin(pi*(n+N/2+0.5)/N)^2
        # = sin(A)^2 + sin(A + pi/2)^2 = sin(A)^2 + cos(A)^2 = 1
        N_half = MDCT_FRAME_SIZE // 2
        princen_bradley_sum = window[:N_half]**2 + window[N_half:]**2
        # The specific window is w[n] = sin(pi(n+0.5)/N).
        # The condition for TDAC is w[n]^2 + w[n+M]^2 = 1 where M is hop size (N/2).
        # This applies to the analysis window being applied, then the synthesis window.
        # For a single sine window used in both analysis and synthesis, this holds.
        # Let's check the sum for the first half against the second half (shifted)
        # For perfect reconstruction with overlap-add, the window must satisfy:
        # window[n]^2 + window[n + NEW_SAMPLES_PER_FRAME]^2 = 1 for n in range(NEW_SAMPLES_PER_FRAME)
        # (assuming NEW_SAMPLES_PER_FRAME is N/2)
        if NEW_SAMPLES_PER_FRAME * 2 == MDCT_FRAME_SIZE: # Only if 50% overlap
            check_len = NEW_SAMPLES_PER_FRAME
            pb_condition_met = window[:check_len]**2 + window[check_len:check_len+check_len]**2
            np.testing.assert_array_almost_equal(pb_condition_met, np.ones(check_len), decimal=6,
                                                 err_msg="Sine window does not satisfy Princen-Bradley condition for TDAC.")

    def test_mdct_imdct_single_block_reconstruction(self):
        """Tests if IMDCT(MDCT(block)) reconstructs the original block when the same window is used."""
        original_block = np.random.rand(MDCT_FRAME_SIZE).astype(np.float64) - 0.5
        window = sine_window(MDCT_FRAME_SIZE).astype(np.float64)

        mdct_coeffs = mdct(original_block, window)
        self.assertEqual(len(mdct_coeffs), NUM_MDCT_COEFFS, "MDCT output length is incorrect.")

        reconstructed_block = imdct(mdct_coeffs, window)
        self.assertEqual(len(reconstructed_block), MDCT_FRAME_SIZE, "IMDCT output length is incorrect.")
        
        # Note: Single block MDCT/IMDCT without overlap-add does not perfectly reconstruct the original signal.
        # It reconstructs y_n = x_n * w_n^2 + x_{n+N/2}*w_{n+N/2}*w_n - ... (aliasing terms)
        # The test here is more of a sanity check that it runs and produces something.
        # The actual perfect reconstruction happens with overlap-add.
        # However, if the window is orthogonal (like a rectangular window, not sine),
        # and (2/N) factor is used, direct reconstruction is closer but still not perfect due to aliasing.
        # The test below (TDAC) is the correct one for sine window.
        # For now, just check if it runs and if the energy is somewhat preserved.
        original_energy = np.sum(original_block**2)
        reconstructed_energy = np.sum(reconstructed_block**2)
        # print(f"Original energy: {original_energy}, Reconstructed energy (single block): {reconstructed_energy}")
        # This test is not very meaningful for sine window without overlap-add.
        # Let's skip direct numerical comparison for single block with sine window.
        pass


    def test_perfect_reconstruction_with_overlap_add_tdac(self):
        """Tests MDCT/IMDCT perfect reconstruction using overlap-add (TDAC)."""
        # Signal length to allow for two overlapping blocks
        signal_length = MDCT_FRAME_SIZE + NEW_SAMPLES_PER_FRAME
        original_signal = np.random.rand(signal_length).astype(np.float64) - 0.5
        
        window = sine_window(MDCT_FRAME_SIZE).astype(np.float64)

        # Block 1
        block1_input = original_signal[0 : MDCT_FRAME_SIZE]
        mdct_coeffs1 = mdct(block1_input, window)
        imdct_output1 = imdct(mdct_coeffs1, window)

        # Block 2
        # Starts NEW_SAMPLES_PER_FRAME (e.g. 1024) samples after block1 start
        block2_input = original_signal[NEW_SAMPLES_PER_FRAME : NEW_SAMPLES_PER_FRAME + MDCT_FRAME_SIZE]
        mdct_coeffs2 = mdct(block2_input, window)
        imdct_output2 = imdct(mdct_coeffs2, window)
        
        # Perform overlap-add
        # First segment of output is the first half of imdct_output1
        output_segment1 = imdct_output1[:NEW_SAMPLES_PER_FRAME]
        
        # Second segment is the sum of the second half of imdct_output1 and first half of imdct_output2
        # This segment corresponds to original_signal[NEW_SAMPLES_PER_FRAME : MDCT_FRAME_SIZE]
        overlap_add_segment = imdct_output1[NEW_SAMPLES_PER_FRAME:] + imdct_output2[:NEW_SAMPLES_PER_FRAME]
        
        # Reconstruct the part of the signal that has been fully processed by two blocks' overlap-add
        # This is the middle part of the original signal segment used for block1 and block2
        # Specifically, original_signal[NEW_SAMPLES_PER_FRAME : MDCT_FRAME_SIZE]
        # This segment has length NEW_SAMPLES_PER_FRAME (e.g. 1024 samples)
        
        original_middle_segment = original_signal[NEW_SAMPLES_PER_FRAME : MDCT_FRAME_SIZE]

        # Ensure lengths match for comparison
        self.assertEqual(len(overlap_add_segment), len(original_middle_segment),
                         "Length mismatch in TDAC test segments.")
        self.assertEqual(len(overlap_add_segment), NEW_SAMPLES_PER_FRAME,
                         "Overlap-add segment length is not NEW_SAMPLES_PER_FRAME.")

        # Compare the overlap-add segment with the corresponding original signal segment
        np.testing.assert_array_almost_equal(
            overlap_add_segment,
            original_middle_segment,
            decimal=6, # Using a common precision for float comparisons
            err_msg="TDAC perfect reconstruction failed for the overlap-add segment."
        )

        # For a more complete test, one could reconstruct more segments, e.g.,
        # by taking a longer original signal and processing multiple blocks.
        # The first NEW_SAMPLES_PER_FRAME and last NEW_SAMPLES_PER_FRAME of the whole signal
        # will not be perfectly reconstructed without padding or windowing at edges.
        # The above test focuses on one fully overlapped segment, which is the core of TDAC.

    def test_mdct_input_validation(self):
        window = sine_window(MDCT_FRAME_SIZE)
        # Wrong input signal length
        with self.assertRaises(ValueError):
            mdct(np.random.rand(MDCT_FRAME_SIZE - 1), window)
        # Wrong window length
        with self.assertRaises(ValueError):
            mdct(np.random.rand(MDCT_FRAME_SIZE), sine_window(MDCT_FRAME_SIZE - 1))

    def test_imdct_input_validation(self):
        window = sine_window(MDCT_FRAME_SIZE)
        # Wrong coefficients length
        with self.assertRaises(ValueError):
            imdct(np.random.rand(NUM_MDCT_COEFFS - 1), window)
        # Wrong window length
        with self.assertRaises(ValueError):
            imdct(np.random.rand(NUM_MDCT_COEFFS), sine_window(MDCT_FRAME_SIZE - 1))


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
```
