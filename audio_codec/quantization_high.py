import numpy as np
import scipy.fft
from audio_codec.globals import NUM_MDCT_COEFFS # For potential use in padding, etc.
from audio_codec.pvq import pvq_encode, pvq_decode # Placeholder PVQ
from audio_codec.quant_utils import quantize_scalar, dequantize_scalar, EPSILON
from audio_codec.spectral_utils import calculate_cepstrum

# Constants
NUM_CEPSTRUM_COEFFS_HIGH = 8
BITS_PER_CEPSTRUM_COEFF_HIGH = 4
PVQ_NORMALIZATION_GAIN_BITS = 5
PVQ_PULSES_BITS = 5 # For signaling the number of pulses
PVQ_MAX_DIMENSION_CLIP = 64

# EPSILON is now imported from quant_utils

# Placeholder Lloyd-Max Quantizer Tables for Cepstrum Coefficients
# For now, uniform quantizers. Range_val might be different per coefficient in a real scenario.
# Assuming cepstrum coefficients are roughly in [-10, 10] for c0 and [-5, 5] for others.
# These ranges are heuristic and need tuning.
LLOYD_MAX_TABLES_HIGH_CEPSTRUM = []
NUM_LEVELS_CEPSTRUM = 2**BITS_PER_CEPSTRUM_COEFF_HIGH

# c0 (log energy) might have a different range
LLOYD_MAX_TABLES_HIGH_CEPSTRUM.append(np.linspace(-10.0, 10.0, NUM_LEVELS_CEPSTRUM))
for _ in range(1, NUM_CEPSTRUM_COEFFS_HIGH):
    LLOYD_MAX_TABLES_HIGH_CEPSTRUM.append(np.linspace(-5.0, 5.0, NUM_LEVELS_CEPSTRUM))

# quantize_scalar and dequantize_scalar are now imported from quant_utils

# Gain Quantization Helpers
# Logarithmic quantization for gain values
# Gain is always positive. Log gain can be positive or negative.
# Typical log2(gain) might range from -5 (gain ~0.03) to +5 (gain ~32)

LOG_GAIN_MIN = -10.0 # Approx gain of 0.001
LOG_GAIN_MAX = 10.0  # Approx gain of 1024

def quantize_gain_log(gain: float, num_bits: int) -> tuple[int, float]:
    """
    Quantizes a gain value logarithmically.
    Maps gain to log2 domain, then uniformly quantizes.
    """
    if gain < EPSILON: # Avoid log of zero or negative
        log_gain = LOG_GAIN_MIN # Map very small gains to min log gain
    else:
        log_gain = np.log2(gain)
    
    # Clip log_gain to the expected range before quantization
    log_gain_clipped = np.clip(log_gain, LOG_GAIN_MIN, LOG_GAIN_MAX)
    
    num_levels = 2**num_bits
    # Quantizer step size in log domain
    q_step = (LOG_GAIN_MAX - LOG_GAIN_MIN) / (num_levels -1) if num_levels > 1 else 0
    
    # Quantize: find nearest level index
    # index = round((log_gain_clipped - LOG_GAIN_MIN) / q_step)
    # Or, define levels and use quantize_scalar logic (more robust for edge cases like num_levels=1)
    
    if num_levels == 1:
        index = 0
        dequantized_log_gain = (LOG_GAIN_MIN + LOG_GAIN_MAX) / 2.0
    else:
        # Calculate quantization levels in log domain
        log_levels = np.linspace(LOG_GAIN_MIN, LOG_GAIN_MAX, num_levels)
        index, dequantized_log_gain = quantize_scalar(log_gain_clipped, log_levels)
        index = int(index) # Ensure it's an int

    dequantized_gain = 2**dequantized_log_gain
    return index, dequantized_gain

def dequantize_gain_log(index: int, num_bits: int) -> float:
    """
    Dequantizes a gain index that was logarithmically quantized.
    """
    num_levels = 2**num_bits
    if not (0 <= index < num_levels):
        raise ValueError(f"Index {index} out of bounds for {num_bits} bits ({num_levels} levels)")

    if num_levels == 1:
        dequantized_log_gain = (LOG_GAIN_MIN + LOG_GAIN_MAX) / 2.0
    else:
        # Calculate quantization levels in log domain (same as in quantize_gain_log)
        log_levels = np.linspace(LOG_GAIN_MIN, LOG_GAIN_MAX, num_levels)
        dequantized_log_gain = dequantize_scalar(index, log_levels)
        
    dequantized_gain = 2**dequantized_log_gain
    return dequantized_gain

def calculate_cepstrum(band_mdct_coeffs: np.ndarray, num_cep_coeffs: int) -> np.ndarray:
    """
    Calculates the real cepstrum of a band of MDCT coefficients.
    """
    if band_mdct_coeffs.ndim != 1 or band_mdct_coeffs.shape[0] == 0:
        raise ValueError("Input MDCT coefficients must be a 1D non-empty array.")

    # a. Calculate Power Spectrum
    power_spec = band_mdct_coeffs**2
    
    # b. Convert to Log Scale
    log_power_spec = np.log(power_spec + EPSILON)
    
    # c. Apply DCT Type-II
    # The length of log_power_spec is the dimension of the band K_band
    # scipy.fft.dct is applied along the last axis by default.
    cepstrum_full = scipy.fft.dct(log_power_spec, type=2, norm='ortho') # norm='ortho' added for consistency

    # d. Return the first num_cep_coeffs
    return cepstrum_full[:num_cep_coeffs]

# calculate_cepstrum is now imported from spectral_utils

def quantize_high_importance_band(band_mdct_coeffs: np.ndarray) -> dict:
    """
    Quantizes a high-importance band using Cepstrum + PVQ.
    """
    K_band = len(band_mdct_coeffs)
    if K_band == 0:
        raise ValueError("Input band_mdct_coeffs cannot be empty.")

    # a. Cepstrum Calculation
    cepstrum_coeffs = calculate_cepstrum(band_mdct_coeffs, NUM_CEPSTRUM_COEFFS_HIGH)

    # b. Cepstrum Quantization
    quantized_cepstrum_indices = []
    dequantized_cepstrum_list = [] # Store for direct use
    for i in range(NUM_CEPSTRUM_COEFFS_HIGH):
        idx, val = quantize_scalar(cepstrum_coeffs[i], LLOYD_MAX_TABLES_HIGH_CEPSTRUM[i])
        quantized_cepstrum_indices.append(idx)
        dequantized_cepstrum_list.append(val)
    
    # c. Reconstruct Quantized Cepstrum (already have dequantized_cepstrum_list)
    dequantized_cepstrum = np.array(dequantized_cepstrum_list)

    # d. Reconstruct Spectral Envelope from Quantized Cepstrum
    # Pad dequantized_cepstrum to original band length K_band for IDCT
    cepstrum_padded = np.zeros(K_band)
    num_coeffs_to_use = min(NUM_CEPSTRUM_COEFFS_HIGH, K_band) # Handle K_band < NUM_CEPSTRUM_COEFFS_HIGH
    cepstrum_padded[:num_coeffs_to_use] = dequantized_cepstrum[:num_coeffs_to_use]
    
    reconstructed_log_spectral_envelope = scipy.fft.idct(cepstrum_padded, type=2, norm='ortho')
    # reconstructed_spectral_envelope_power = np.exp(reconstructed_log_spectral_envelope)
    # Amplitude envelope:
    amp_envelope = np.sqrt(np.exp(reconstructed_log_spectral_envelope) + EPSILON) # Add EPSILON before sqrt too

    # e. Residual Calculation
    # normalized_residual_vector = band_mdct_coeffs / (amp_envelope + EPSILON)
    # Ensure signs are preserved and consistent with how amp_envelope is derived.
    # If cepstrum is from MDCT^2, then amp_envelope is |MDCT_reconstructed_from_cepstrum|.
    # The residual should be MDCT_original / MDCT_reconstructed_from_cepstrum (with signs).
    # A simpler way: residual = original_coeffs - reconstructed_coeffs_from_envelope, then normalize.
    # However, the spec implies a multiplicative residual.
    # Let's stick to:
    normalized_residual_vector = band_mdct_coeffs / (amp_envelope + EPSILON)


    # f. Residual Normalization (L2 norm)
    gain = np.linalg.norm(normalized_residual_vector, 2)
    # If gain is very small, pvq_target_vector might become unstable or all zeros.
    if gain < EPSILON:
        pvq_target_vector = np.zeros_like(normalized_residual_vector)
        # If K_band > 0, set first component to 1 to make it a valid unit vector for PVQ
        if K_band > 0 : pvq_target_vector[0] = 1.0 
    else:
        pvq_target_vector = normalized_residual_vector / gain
        
    # g. Quantize Gain
    quantized_residual_gain_index, _ = quantize_gain_log(gain, PVQ_NORMALIZATION_GAIN_BITS)

    # h. Determine PVQ Pulse Count (L_high)
    # For now, fixed value. Range 4-32. Signaling uses PVQ_PULSES_BITS (5 bits).
    # A value like 10 is a placeholder.
    pvq_pulse_count = 10 # Placeholder
    # Ensure it's within representable range of PVQ_PULSES_BITS (0-31 index for 4-35 pulses)
    # Or, if pvq_pulse_count is the value itself, it should be e.g. 4 to 32.
    # The spec says "4-32の範囲。パルス数は5ビットで符号化。"
    # This means the value 4 is encoded as 0, 32 as 28.
    # Let's assume pvq_pulse_count IS the actual number of pulses.
    # The signaling of this count (e.g. (pvq_pulse_count - 4)) will be handled by entropy coder.
    # Here we just need the value.
    pvq_pulse_count = np.clip(pvq_pulse_count, 4, 32)


    # i. PVQ Encoding of pvq_target_vector
    pvq_codeword_indices = []
    
    num_subvectors = (K_band + PVQ_MAX_DIMENSION_CLIP - 1) // PVQ_MAX_DIMENSION_CLIP
    
    for i_sub in range(num_subvectors):
        start = i_sub * PVQ_MAX_DIMENSION_CLIP
        end = min((i_sub + 1) * PVQ_MAX_DIMENSION_CLIP, K_band)
        subvector = pvq_target_vector[start:end]
        
        # PVQ expects L2 normalized vectors. Our subvectors might not be perfectly L2 normalized
        # if the original pvq_target_vector was split.
        # Re-normalize subvectors before PVQ encoding.
        subvector_norm = np.linalg.norm(subvector, 2)
        if subvector_norm < EPSILON:
            # If subvector is zero, create a dummy unit vector for PVQ.
            # This can happen if parts of the gain-normalized residual are zero.
            normalized_subvector = np.zeros_like(subvector)
            if len(subvector)>0: normalized_subvector[0] = 1.0
        else:
            normalized_subvector = subvector / subvector_norm
            
        # Edge case: if subvector dimension is 0 (e.g. K_band = 0 was not caught, or bad split)
        if normalized_subvector.shape[0] == 0:
            # This should not happen if K_band > 0 and splitting logic is correct
            # pvq_idx = 0 # Or handle error appropriately
            continue # Skip if subvector is empty

        pvq_idx = pvq_encode(normalized_subvector, pvq_pulse_count)
        pvq_codeword_indices.append(pvq_idx)

    return {
        "quantized_cepstrum_indices": quantized_cepstrum_indices,
        "quantized_residual_gain_index": quantized_residual_gain_index,
        "pvq_pulse_count": pvq_pulse_count, # Actual number of pulses
        "pvq_codeword_indices": pvq_codeword_indices,
        "band_length": K_band # Needed for dequantization
    }

def dequantize_high_importance_band(quant_data: dict) -> np.ndarray:
    """
    Dequantizes a high-importance band from its quantized representation.
    """
    quantized_cepstrum_indices = quant_data["quantized_cepstrum_indices"]
    quantized_residual_gain_index = quant_data["quantized_residual_gain_index"]
    pvq_pulse_count = quant_data["pvq_pulse_count"]
    pvq_codeword_indices = quant_data["pvq_codeword_indices"]
    K_band = quant_data["band_length"]

    if K_band == 0:
        return np.array([])

    # a. Dequantize Cepstrum
    dequantized_cepstrum_list = []
    for i, idx in enumerate(quantized_cepstrum_indices):
        val = dequantize_scalar(idx, LLOYD_MAX_TABLES_HIGH_CEPSTRUM[i])
        dequantized_cepstrum_list.append(val)
    dequantized_cepstrum = np.array(dequantized_cepstrum_list)

    # b. Reconstruct Spectral Envelope
    cepstrum_padded = np.zeros(K_band)
    num_coeffs_to_use = min(NUM_CEPSTRUM_COEFFS_HIGH, K_band)
    cepstrum_padded[:num_coeffs_to_use] = dequantized_cepstrum[:num_coeffs_to_use]
    
    reconstructed_log_spectral_envelope = scipy.fft.idct(cepstrum_padded, type=2, norm='ortho')
    # amp_envelope = np.sqrt(np.exp(reconstructed_log_spectral_envelope) + EPSILON)
    # Make sure this matches the exact formulation in encoder.
    # If there was an EPSILON before np.exp in encoder, it should be here too.
    # Based on encoder: np.sqrt(np.exp(reconstructed_log_spectral_envelope) + EPSILON)
    amp_envelope = np.sqrt(np.exp(reconstructed_log_spectral_envelope) + EPSILON)


    # c. Dequantize Residual Gain
    dequantized_gain = dequantize_gain_log(quantized_residual_gain_index, PVQ_NORMALIZATION_GAIN_BITS)

    # d. PVQ Decoding
    reconstructed_normalized_residual_parts = []
    num_subvectors = (K_band + PVQ_MAX_DIMENSION_CLIP - 1) // PVQ_MAX_DIMENSION_CLIP

    current_dim_offset = 0
    for i_sub in range(num_subvectors):
        start = i_sub * PVQ_MAX_DIMENSION_CLIP
        end = min((i_sub + 1) * PVQ_MAX_DIMENSION_CLIP, K_band)
        subvector_dim = end - start
        
        if subvector_dim == 0:
            continue

        pvq_idx = pvq_codeword_indices[i_sub]
        # Decoded subvector is L2 normalized
        decoded_subvector = pvq_decode(pvq_idx, subvector_dim, pvq_pulse_count)
        reconstructed_normalized_residual_parts.append(decoded_subvector)
        current_dim_offset += subvector_dim
        
    # e. Concatenate decoded subvectors
    if not reconstructed_normalized_residual_parts: # If K_band was 0 or all subvectors were empty
        reconstructed_normalized_residual_vector = np.zeros(K_band)
        if K_band > 0: # Avoid issues if K_band is 0
            # This case might imply an error or an all-zero signal that was encoded.
            # If the vector was meant to be zero, gain would be zero.
            # PVQ vectors are unit norm, so this implies a placeholder for zero-energy parts.
            pass # Remains zeros
    else:
        reconstructed_normalized_residual_vector = np.concatenate(reconstructed_normalized_residual_parts)

    # The subvectors were individually L2 normalized before PVQ encoding in the quantizer if split.
    # The PVQ decoder returns L2 normalized vectors.
    # To reconstruct the original `pvq_target_vector` (which was globally L2 normalized),
    # we need to rescale the concatenated subvectors so that the whole vector is L2 normalized.
    # This is tricky: the original `pvq_target_vector` was L2 normalized *before* splitting.
    # The current `reconstructed_normalized_residual_vector` is a concatenation of unit-norm subvectors.
    # This means its overall norm is sqrt(num_subvectors_with_energy).
    # This needs to be re-normalized to unit norm to match the `pvq_target_vector` that `gain` applies to.
    # However, if the subvector re-normalization in the encoder was `subvector / subvector_norm`
    # and PVQ reconstructs that `normalized_subvector`, then concatenating them does NOT yield
    # the original `pvq_target_vector`. This is a common issue in PVQ with subvector splitting.
    # For now, assume the concatenation is a good enough approximation of the shape.
    # A more correct approach would involve distributing the global norm or encoding sub-gains.
    # Let's re-normalize the concatenated vector to have unit norm, to be consistent with `pvq_target_vector`
    # before scaling by `dequantized_gain`.
    
    current_norm = np.linalg.norm(reconstructed_normalized_residual_vector, 2)
    if current_norm > EPSILON:
        reconstructed_normalized_residual_vector /= current_norm
    else:
        # If norm is zero (e.g. all subvectors were zero, or K_band is small and PVQ returns zero for some reason)
        # Set to a default unit vector, e.g. [1, 0, 0 ...]
        # This matches the encoder's handling of zero gain.
        if K_band > 0 : 
            reconstructed_normalized_residual_vector = np.zeros(K_band)
            reconstructed_normalized_residual_vector[0] = 1.0


    # f. Rescale Residual
    reconstructed_residual_vector = reconstructed_normalized_residual_vector * dequantized_gain
    
    # g. Combine with Envelope
    reconstructed_band_mdct_coeffs = reconstructed_residual_vector * amp_envelope
    
    return reconstructed_band_mdct_coeffs

```
