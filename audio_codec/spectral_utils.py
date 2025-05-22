import numpy as np
import scipy.fft
from audio_codec.quant_utils import EPSILON # Import EPSILON from the new quant_utils

def calculate_cepstrum(band_mdct_coeffs: np.ndarray, num_cep_coeffs: int) -> np.ndarray:
    """
    Calculates the real cepstrum of a band of MDCT coefficients.
    """
    if not isinstance(band_mdct_coeffs, np.ndarray):
        raise TypeError("Input MDCT coefficients must be a NumPy array.")
    if band_mdct_coeffs.ndim != 1 or band_mdct_coeffs.shape[0] == 0:
        raise ValueError("Input MDCT coefficients must be a 1D non-empty array.")
    if not isinstance(num_cep_coeffs, int) or num_cep_coeffs <= 0:
        raise ValueError("Number of cepstrum coefficients must be a positive integer.")
    if num_cep_coeffs > band_mdct_coeffs.shape[0]:
        # This can happen if band is very small.
        # Cepstrum calculation will still work, but result will be shorter than num_cep_coeffs
        # or DCT might behave unexpectedly if num_cep_coeffs is used to slice padded versions later.
        # For now, let's allow it, the slicing will handle it.
        pass

    # a. Calculate Power Spectrum
    power_spec = band_mdct_coeffs**2
    
    # b. Convert to Log Scale
    # Adding EPSILON to avoid log(0)
    log_power_spec = np.log(power_spec + EPSILON)
    
    # c. Apply DCT Type-II
    # The length of log_power_spec is the dimension of the band K_band
    # scipy.fft.dct is applied along the last axis by default.
    # norm='ortho' ensures the transform is orthogonal
    cepstrum_full = scipy.fft.dct(log_power_spec, type=2, norm='ortho')
    
    # d. Return the first num_cep_coeffs
    # If K_band < num_cep_coeffs, this will return all available coefficients.
    return cepstrum_full[:num_cep_coeffs]
