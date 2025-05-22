import numpy as np
from audio_codec.globals import BAND_BOUNDARIES, NUM_BANDS, NUM_MDCT_COEFFS

def split_mdct_coeffs_into_bands(mdct_coeffs: np.ndarray) -> list[np.ndarray]:
    """
    Splits MDCT coefficients into multiple bands based on predefined boundaries.

    Args:
        mdct_coeffs: A NumPy array of MDCT coefficients (length NUM_MDCT_COEFFS).

    Returns:
        A list of NumPy arrays, where each array contains the MDCT coefficients
        for one band. The list will contain NUM_BANDS arrays.
    """
    if not isinstance(mdct_coeffs, np.ndarray):
        raise TypeError("Input mdct_coeffs must be a NumPy array.")
    
    if mdct_coeffs.ndim != 1:
        raise ValueError("Input mdct_coeffs must be a 1-dimensional array.")

    if mdct_coeffs.shape[0] != NUM_MDCT_COEFFS:
        raise ValueError(f"Input mdct_coeffs must have length {NUM_MDCT_COEFFS}, "
                         f"but got {mdct_coeffs.shape[0]}.")

    if len(BAND_BOUNDARIES) != NUM_BANDS + 1:
        raise ValueError("BAND_BOUNDARIES length must be NUM_BANDS + 1.")

    banded_coeffs = []
    for i in range(NUM_BANDS):
        start_index = BAND_BOUNDARIES[i]
        end_index = BAND_BOUNDARIES[i+1]
        
        if start_index < 0 or end_index > mdct_coeffs.shape[0] or start_index >= end_index:
            raise ValueError(f"Invalid band boundary for band {i}: [{start_index}, {end_index})")
            
        banded_coeffs.append(mdct_coeffs[start_index:end_index])
        
    return banded_coeffs

def combine_bands_into_mdct_coeffs(banded_coeffs: list[np.ndarray]) -> np.ndarray:
    """
    Combines a list of banded MDCT coefficients back into a single MDCT coefficient array.

    Args:
        banded_coeffs: A list of NumPy arrays, where each array contains the
                       MDCT coefficients for one band.

    Returns:
        A NumPy array of combined MDCT coefficients (length NUM_MDCT_COEFFS).
    """
    if not isinstance(banded_coeffs, list) or \
       not all(isinstance(band, np.ndarray) for band in banded_coeffs):
        raise TypeError("Input banded_coeffs must be a list of NumPy arrays.")

    if len(banded_coeffs) != NUM_BANDS:
        raise ValueError(f"Input banded_coeffs must contain {NUM_BANDS} arrays, "
                         f"but got {len(banded_coeffs)}.")

    # It's good practice to also check if the total number of coefficients matches NUM_MDCT_COEFFS
    # total_coeffs_count = sum(band.shape[0] for band in banded_coeffs)
    # if total_coeffs_count != NUM_MDCT_COEFFS:
    #     raise ValueError(f"Total number of coefficients in bands ({total_coeffs_count}) "
    #                      f"does not match NUM_MDCT_COEFFS ({NUM_MDCT_COEFFS}).")
    # For now, we assume BAND_BOUNDARIES is consistent with NUM_MDCT_COEFFS

    # Concatenate all band arrays directly
    # This relies on the bands being in the correct order and covering the entire spectrum
    # as defined by BAND_BOUNDARIES.
    return np.concatenate(banded_coeffs)
