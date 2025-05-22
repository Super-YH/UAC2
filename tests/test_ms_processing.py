import unittest
import numpy as np
from audio_codec.ms_processing import apply_ms_processing, apply_inverse_ms_processing, MS_ENERGY_THRESHOLD
from audio_codec.globals import MDCT_FRAME_SIZE # For sample data length

class TestMSProcessing(unittest.TestCase):

    def test_apply_ms_processing_and_inverse(self):
        # Test data (can be any length, using MDCT_FRAME_SIZE for relevance)
        length = MDCT_FRAME_SIZE

        # Case 1: L and R are very similar (L approx R) -> MS should ideally NOT be applied
        # S_energy / M_energy should be small, but potentially not less than MS_ENERGY_THRESHOLD if M is also small.
        # If L=R, S=0. S_energy/M_energy = 0 < MS_ENERGY_THRESHOLD -> MS will be applied.
        # This is fine, M/S can represent L/R perfectly.
        l_ch_case1 = np.random.rand(length) * 0.5 + 0.1 # Random data
        r_ch_case1 = l_ch_case1.copy() 
        
        mid1, side1, ms_applied1 = apply_ms_processing(l_ch_case1, r_ch_case1)
        self.assertTrue(ms_applied1, "Case 1 (L=R): MS processing should be applied as S_energy is 0.")
        
        # Verify inverse
        l_reco1, r_reco1 = apply_inverse_ms_processing(mid1, side1, ms_applied1)
        np.testing.assert_array_almost_equal(l_ch_case1, l_reco1, decimal=6, err_msg="Case 1: L channel reconstruction failed.")
        np.testing.assert_array_almost_equal(r_ch_case1, r_reco1, decimal=6, err_msg="Case 1: R channel reconstruction failed.")

        # Case 2: L and R are very different (L = -R) -> MS should be applied if M_energy is not too small
        # If L = -R, M=0. S_energy/M_energy = Inf. This should result in MS NOT being applied by current logic.
        # The current logic: `if m_energy > 1e-9: if (s_energy / m_energy) < MS_ENERGY_THRESHOLD:`
        # If m_energy is ~0, MS is not applied.
        l_ch_case2 = np.random.rand(length) * 0.5 + 0.1
        r_ch_case2 = -l_ch_case2
        
        mid2, side2, ms_applied2 = apply_ms_processing(l_ch_case2, r_ch_case2)
        # For L=-R, M is 0. S is L.
        # M_energy is 0. S_energy is sum(L^2).
        # The condition `if m_energy > 1e-9:` will be false. So MS is NOT applied.
        self.assertFalse(ms_applied2, "Case 2 (L=-R): MS processing should NOT be applied as M_energy is ~0.")
        
        l_reco2, r_reco2 = apply_inverse_ms_processing(mid2, side2, ms_applied2)
        np.testing.assert_array_almost_equal(l_ch_case2, l_reco2, decimal=6, err_msg="Case 2: L channel reconstruction failed.")
        np.testing.assert_array_almost_equal(r_ch_case2, r_reco2, decimal=6, err_msg="Case 2: R channel reconstruction failed.")
        self.assertTrue(mid2 is l_ch_case2, "Case 2: Mid output should be L input when MS not applied.")
        self.assertTrue(side2 is r_ch_case2, "Case 2: Side output should be R input when MS not applied.")


        # Case 3: A mix, where MS processing should be applied based on threshold
        # Create data where S_energy / M_energy < MS_ENERGY_THRESHOLD
        # Example: Mid signal strong, Side signal weak.
        # Let Mid = A, Side = B.  L = A+B, R = A-B.
        # M_energy = sum(A^2), S_energy = sum(B^2).
        # We want sum(B^2)/sum(A^2) < MS_ENERGY_THRESHOLD
        mid_component = np.random.rand(length) # Strong
        side_component = np.random.rand(length) * np.sqrt(MS_ENERGY_THRESHOLD * 0.5) # Weak side
        
        l_ch_case3 = mid_component + side_component
        r_ch_case3 = mid_component - side_component
        
        mid3, side3, ms_applied3 = apply_ms_processing(l_ch_case3, r_ch_case3)
        self.assertTrue(ms_applied3, "Case 3 (Mid strong, Side weak): MS processing should be applied.")
        
        l_reco3, r_reco3 = apply_inverse_ms_processing(mid3, side3, ms_applied3)
        np.testing.assert_array_almost_equal(l_ch_case3, l_reco3, decimal=6, err_msg="Case 3: L channel reconstruction failed.")
        np.testing.assert_array_almost_equal(r_ch_case3, r_reco3, decimal=6, err_msg="Case 3: R channel reconstruction failed.")

        # Case 4: A mix, where MS processing should NOT be applied
        # Create data where S_energy / M_energy > MS_ENERGY_THRESHOLD
        # Example: Side signal strong relative to Mid signal.
        mid_component_weak = np.random.rand(length) * np.sqrt(MS_ENERGY_THRESHOLD * 0.5)
        side_component_strong = np.random.rand(length) 
        
        l_ch_case4 = mid_component_weak + side_component_strong
        r_ch_case4 = mid_component_weak - side_component_strong
        
        mid4, side4, ms_applied4 = apply_ms_processing(l_ch_case4, r_ch_case4)
        # If mid_component_weak is very small, m_energy can be < 1e-9.
        # To ensure m_energy > 1e-9 for the ratio test:
        if np.sum(mid_component_weak**2) < 1e-8: # if M energy is too low, make it slightly higher
             mid_component_weak += 0.01 
             l_ch_case4 = mid_component_weak + side_component_strong
             r_ch_case4 = mid_component_weak - side_component_strong
        
        mid4, side4, ms_applied4 = apply_ms_processing(l_ch_case4, r_ch_case4)
        self.assertFalse(ms_applied4, "Case 4 (Side strong, Mid weak): MS processing should NOT be applied.")

        l_reco4, r_reco4 = apply_inverse_ms_processing(mid4, side4, ms_applied4)
        np.testing.assert_array_almost_equal(l_ch_case4, l_reco4, decimal=6, err_msg="Case 4: L channel reconstruction failed.")
        np.testing.assert_array_almost_equal(r_ch_case4, r_reco4, decimal=6, err_msg="Case 4: R channel reconstruction failed.")

    def test_ms_processing_edge_cases(self):
        length = 16 # Small length for edge cases
        # Zero input
        l_zero = np.zeros(length)
        r_zero = np.zeros(length)
        mid_z, side_z, ms_applied_z = apply_ms_processing(l_zero, r_zero)
        # M_energy=0, S_energy=0. (S/M) is nan. MS not applied by `m_energy > 1e-9` check.
        self.assertFalse(ms_applied_z, "Zero input: MS should not be applied (M_energy is 0).")
        l_reco_z, r_reco_z = apply_inverse_ms_processing(mid_z, side_z, ms_applied_z)
        np.testing.assert_array_equal(l_zero, l_reco_z)
        np.testing.assert_array_equal(r_zero, r_reco_z)

        # Input not numpy arrays
        with self.assertRaises(TypeError):
            apply_ms_processing([1,2,3], np.array([1,2,3]))
        with self.assertRaises(TypeError):
            apply_inverse_ms_processing([1,2,3], np.array([1,2,3]), False)
        
        # Input arrays of different lengths
        with self.assertRaises(ValueError):
            apply_ms_processing(np.array([1,2,3]), np.array([1,2]))


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False) # For running in notebook/environment where argv might be different
```
