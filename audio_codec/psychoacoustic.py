import numpy as np
import math
from audio_codec.globals import NUM_BANDS

# Psychoacoustic constants and parameters

# Placeholder Absolute Threshold of Hearing (ATH) values for each band.
# These are arbitrary and should be tuned/derived from actual psychoacoustic data.
# Higher values mean higher threshold (less sensitivity).
# The values should correspond to the energy scale of MDCT coefficients.
ATH_TABLE = np.array([
    0.5, 0.4, 0.3, 0.2, 0.15, 0.1, 0.08, 0.06, 
    0.05, 0.04, 0.03, 0.02, 0.015, 0.01, 0.008, 0.005
]) * 1e-3 # Scaled down, assuming MDCT coeffs are typically in [-1, 1] range, so energy is smaller.

MASKING_CONSTANT_C = 2.0  # Multiplier for ATH to get masking threshold

# Thresholds for P_band (Signal-to-Mask Ratio proxy)
HIGH_IMPORTANCE_P_BAND_THRESHOLD = 100.0  # Approx 20dB SMR (10*log10(100))
LOW_IMPORTANCE_P_BAND_THRESHOLD = 4.0     # Approx 6dB SMR (10*log10(4))

# Relative energy threshold for high importance: band energy must be in the top 25th percentile.
HIGH_IMPORTANCE_ENERGY_PERCENTILE = 0.25 # Top 25% of band energies

# Energy threshold for low importance: band energy relative to total frame energy.
LOW_IMPORTANCE_ENERGY_THRESHOLD_RATIO = 0.001  # 0.1% of total energy

# Small epsilon to prevent division by zero or issues with very small mask thresholds
MASK_THRESHOLD_EPSILON = 1e-9

def evaluate_band_importance(banded_mdct_coeffs: list[np.ndarray]) -> tuple[list[str], list[float], float]:
    """
    Evaluates the psychoacoustic importance of each MDCT coefficient band.

    Args:
        banded_mdct_coeffs: A list of NumPy arrays, where each array contains
                            the MDCT coefficients for one band.

    Returns:
        A tuple containing:
        - importances: A list of strings ("high", "medium", "low") for each band.
        - band_energies: A list of float values representing the energy of each band.
        - total_frame_energy: The total energy of all bands in the current frame.
    """
    if not isinstance(banded_mdct_coeffs, list) or \
       not all(isinstance(band, np.ndarray) for band in banded_mdct_coeffs):
        raise TypeError("Input banded_mdct_coeffs must be a list of NumPy arrays.")
    
    if len(banded_mdct_coeffs) != NUM_BANDS:
        raise ValueError(f"Expected {NUM_BANDS} bands, got {len(banded_mdct_coeffs)}.")

    if ATH_TABLE.shape[0] != NUM_BANDS:
        raise ValueError(f"ATH_TABLE must have {NUM_BANDS} entries, got {ATH_TABLE.shape[0]}.")

    band_energies = np.array([np.sum(band_coeffs**2) for band_coeffs in banded_mdct_coeffs])
    total_frame_energy = np.sum(band_energies)

    importances = []

    # Determine the energy threshold for the "top X%" criterion for high importance
    if NUM_BANDS == 0: # Should not happen with NUM_BANDS > 0 check above, but good for robustness
        energy_threshold_for_top_percentile = -1.0 # No bands, no threshold
    elif NUM_BANDS > 0 :
        num_top_bands = math.ceil(NUM_BANDS * HIGH_IMPORTANCE_ENERGY_PERCENTILE)
        if num_top_bands == 0 and NUM_BANDS > 0 : # e.g. if HIGH_IMPORTANCE_ENERGY_PERCENTILE is very small
             num_top_bands = 1 # always consider at least the top band if there are bands
        
        if num_top_bands > NUM_BANDS: # Should not happen if percentile <= 1.0
            num_top_bands = NUM_BANDS

        if total_frame_energy < MASK_THRESHOLD_EPSILON : # if total energy is near zero, all bands are unimportant
            energy_threshold_for_top_percentile = MASK_THRESHOLD_EPSILON # any positive energy will be above
        elif num_top_bands == 0 : # Only if NUM_BANDS is 0
             energy_threshold_for_top_percentile = -1.0
        else:
            sorted_energies = np.sort(band_energies)[::-1] # Sort in descending order
            # Ensure index is valid, especially if fewer than num_top_bands exist (e.g. all energies are zero)
            idx = min(num_top_bands - 1, len(sorted_energies) - 1)
            if idx < 0: # If sorted_energies is empty (e.g. NUM_BANDS = 0)
                energy_threshold_for_top_percentile = -1.0 # Effectively no band can be high importance
            else:
                energy_threshold_for_top_percentile = sorted_energies[idx]


    for i in range(NUM_BANDS):
        current_E_band = band_energies[i]
        ATH_band = ATH_TABLE[i]
        
        # Calculate masking threshold from ATH
        # Mask_thresh must be positive for P_band calculation
        Mask_thresh = max(ATH_band * MASKING_CONSTANT_C, MASK_THRESHOLD_EPSILON)

        P_band = current_E_band / Mask_thresh

        # High Importance Condition
        # Band energy must be in the top percentile AND P_band must be high
        # If energy_threshold_for_top_percentile is very low (e.g. frame is silence),
        # current_E_band >= energy_threshold_for_top_percentile might be true for many bands.
        # This is fine, P_band will be low for silent bands.
        is_high = (P_band > HIGH_IMPORTANCE_P_BAND_THRESHOLD) and \
                  (current_E_band >= energy_threshold_for_top_percentile - MASK_THRESHOLD_EPSILON) # Add epsilon for float comparison

        # Low Importance Condition
        # P_band is low OR band energy is very low relative to total frame energy
        # Ensure total_frame_energy is not zero for the relative energy check
        low_energy_abs_threshold = LOW_IMPORTANCE_ENERGY_THRESHOLD_RATIO * total_frame_energy
        is_low = (P_band < LOW_IMPORTANCE_P_BAND_THRESHOLD) or \
                 (current_E_band < low_energy_abs_threshold)

        # Decision Logic
        if is_high:
            importances.append("high")
        elif is_low:
            importances.append("low")
        else:
            importances.append("medium")
            
    return importances, band_energies.tolist(), total_frame_energy

```
