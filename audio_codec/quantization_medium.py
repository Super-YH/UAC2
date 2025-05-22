import numpy as np
import scipy.fft
from audio_codec.quant_utils import quantize_scalar, dequantize_scalar, EPSILON
from audio_codec.spectral_utils import calculate_cepstrum
# No specific globals needed for this module other than potentially NUM_MDCT_COEFFS for consistency checks if added later.

# Constants
NUM_CEPSTRUM_COEFFS_MEDIUM = 4
BITS_PER_CEPSTRUM_COEFF_MEDIUM = 3 # (2^3 = 8 levels)

# Placeholder Lloyd-Max Quantizer Tables for Cepstrum Coefficients (Medium Importance)
# Uniform quantizers. Ranges are heuristic and may need tuning.
# Assuming cepstrum coefficients are roughly in [-8, 8] for c0 and [-4, 4] for others for medium bands.
LLOYD_MAX_TABLES_MEDIUM_CEPSTRUM = []
NUM_LEVELS_CEPSTRUM_MEDIUM = 2**BITS_PER_CEPSTRUM_COEFF_MEDIUM

# c0 (log energy) might have a different range
LLOYD_MAX_TABLES_MEDIUM_CEPSTRUM.append(np.linspace(-8.0, 8.0, NUM_LEVELS_CEPSTRUM_MEDIUM)) 
for _ in range(1, NUM_CEPSTRUM_COEFFS_MEDIUM):
    LLOYD_MAX_TABLES_MEDIUM_CEPSTRUM.append(np.linspace(-4.0, 4.0, NUM_LEVELS_CEPSTRUM_MEDIUM))

def quantize_medium_importance_band(band_mdct_coeffs: np.ndarray) -> dict:
    """
    Quantizes a medium-importance band using Cepstrum coefficients.
    No residual/fine structure is encoded.

    Args:
        band_mdct_coeffs: NumPy array of MDCT coefficients for the band.

    Returns:
        A dictionary containing:
        - "quantized_cepstrum_indices": list[int] (NUM_CEPSTRUM_COEFFS_MEDIUM indices)
        - "band_length": int (original length of the MDCT coefficient band)
    """
    if not isinstance(band_mdct_coeffs, np.ndarray):
        raise TypeError("Input band_mdct_coeffs must be a NumPy array.")
    K_band = len(band_mdct_coeffs)
    if K_band == 0:
        # Handle empty band case: return empty indices or default values
        # For now, let's assume bands are non-empty if they reach quantization stage.
        # Or raise error as it might indicate an issue upstream.
        raise ValueError("Input band_mdct_coeffs cannot be empty for quantization.")

    # a. Cepstrum Calculation
    # calculate_cepstrum handles if K_band < NUM_CEPSTRUM_COEFFS_MEDIUM by returning fewer coeffs.
    cepstrum_coeffs = calculate_cepstrum(band_mdct_coeffs, NUM_CEPSTRUM_COEFFS_MEDIUM)
    
    # Ensure cepstrum_coeffs has NUM_CEPSTRUM_COEFFS_MEDIUM elements, pad with zero if shorter (e.g. K_band is small)
    if len(cepstrum_coeffs) < NUM_CEPSTRUM_COEFFS_MEDIUM:
        cepstrum_coeffs_padded = np.zeros(NUM_CEPSTRUM_COEFFS_MEDIUM)
        cepstrum_coeffs_padded[:len(cepstrum_coeffs)] = cepstrum_coeffs
        cepstrum_coeffs = cepstrum_coeffs_padded


    # b. Cepstrum Quantization
    quantized_cepstrum_indices = []
    for i in range(NUM_CEPSTRUM_COEFFS_MEDIUM):
        # Ensure we have a coefficient to quantize (e.g., if K_band was very small)
        # The padding above should handle this.
        coeff_val = cepstrum_coeffs[i]
        idx, _ = quantize_scalar(coeff_val, LLOYD_MAX_TABLES_MEDIUM_CEPSTRUM[i])
        quantized_cepstrum_indices.append(idx)
            
    return {
        "quantized_cepstrum_indices": quantized_cepstrum_indices,
        "band_length": K_band 
    }

def dequantize_medium_importance_band(quant_data: dict) -> np.ndarray:
    """
    Dequantizes a medium-importance band from its quantized cepstrum representation.
    Reconstructs only the spectral envelope; fine structure is not preserved.

    Args:
        quant_data: A dictionary containing:
            - "quantized_cepstrum_indices": list[int]
            - "band_length": int (original length of the MDCT coefficient band)

    Returns:
        Reconstructed band MDCT coefficients (np.ndarray).
    """
    quantized_cepstrum_indices = quant_data["quantized_cepstrum_indices"]
    K_band = quant_data["band_length"]

    if K_band == 0:
        return np.array([])
    
    if len(quantized_cepstrum_indices) != NUM_CEPSTRUM_COEFFS_MEDIUM:
        raise ValueError(f"Expected {NUM_CEPSTRUM_COEFFS_MEDIUM} cepstrum indices, "
                         f"got {len(quantized_cepstrum_indices)}")

    # a. Dequantize Cepstrum
    dequantized_cepstrum_list = []
    for i, idx in enumerate(quantized_cepstrum_indices):
        val = dequantize_scalar(idx, LLOYD_MAX_TABLES_MEDIUM_CEPSTRUM[i])
        dequantized_cepstrum_list.append(val)
    dequantized_cepstrum = np.array(dequantized_cepstrum_list)

    # b. Reconstruct Spectral Envelope
    # Pad dequantized_cepstrum to original band length K_band for IDCT
    # Only use up to K_band coefficients if K_band < NUM_CEPSTRUM_COEFFS_MEDIUM
    # (though dequantized_cepstrum should always have NUM_CEPSTRUM_COEFFS_MEDIUM elements based on encoder)
    cepstrum_padded = np.zeros(K_band)
    num_coeffs_to_use = min(NUM_CEPSTRUM_COEFFS_MEDIUM, K_band)
    cepstrum_padded[:num_coeffs_to_use] = dequantized_cepstrum[:num_coeffs_to_use]
    
    # Apply IDCT (Type-II, ortho-normalized)
    reconstructed_log_spectral_envelope = scipy.fft.idct(cepstrum_padded, type=2, norm='ortho')
    
    # Convert from log-power to power, then to amplitude
    reconstructed_spectral_envelope_power = np.exp(reconstructed_log_spectral_envelope)
    
    # c. Reconstruct MDCT Coefficients (Amplitude only, zero phase)
    # "位相は0またはランダム。" -> Using zero phase means positive square root.
    # Add EPSILON before sqrt to prevent issues with negative (due to numerical errors) or zero values.
    reconstructed_amplitude_envelope = np.sqrt(np.maximum(reconstructed_spectral_envelope_power, EPSILON))
    
    reconstructed_band_mdct_coeffs = reconstructed_amplitude_envelope
    
    # Ensure output has the correct length K_band. IDCT output should match.
    # If K_band was smaller than num_coeffs_to_use in cepstrum_padded, IDCT result might be shorter.
    # However, cepstrum_padded is of length K_band, so IDCT output will be K_band.
    
    return reconstructed_band_mdct_coeffs
```
