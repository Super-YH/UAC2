import numpy as np
from audio_codec.globals import MDCT_FRAME_SIZE, NUM_MDCT_COEFFS # Assuming N = MDCT_FRAME_SIZE, M = NUM_MDCT_COEFFS = N/2

def sine_window(length: int) -> np.ndarray:
    """
    Generates a Sine window.

    Args:
        length: The length of the window (N).

    Returns:
        A NumPy array representing the Sine window of the specified length.
        w[n] = sin(pi * (n + 0.5) / length) for n = 0, ..., length-1.
    """
    if not isinstance(length, int) or length <= 0:
        raise ValueError("Window length must be a positive integer.")
    return np.sin(np.pi * (np.arange(length) + 0.5) / length)

def mdct(input_signal: np.ndarray, window: np.ndarray) -> np.ndarray:
    """
    Computes the Modified Discrete Cosine Transform (MDCT) of an input signal.

    Args:
        input_signal: A NumPy array of audio samples (length N).
        window: A NumPy array representing the window to be applied (length N).

    Returns:
        A NumPy array of MDCT coefficients (length N/2).
        Formula: X_k = sum_{n=0}^{N-1} input_signal[n] * window[n] * cos(pi/N * (n + 0.5 + N/2) * (k + 0.5))
                 for k = 0, ..., N/2 - 1.
    """
    N = input_signal.shape[0]
    if N != MDCT_FRAME_SIZE:
        raise ValueError(f"Input signal length must be {MDCT_FRAME_SIZE}, but got {N}")
    if window.shape[0] != N:
        raise ValueError(f"Window length must be {N}, but got {window.shape[0]}")

    M = N // 2 # NUM_MDCT_COEFFS

    windowed_signal = input_signal * window
    mdct_coeffs = np.zeros(M)

    # Precompute (n + 0.5 + N/2) terms
    n_factor = np.arange(N) + 0.5 + M # M is N/2

    for k in range(M):
        k_factor = k + 0.5
        cos_term = np.cos(np.pi / N * n_factor * k_factor)
        mdct_coeffs[k] = np.sum(windowed_signal * cos_term)
    
    return mdct_coeffs

def imdct(mdct_coeffs: np.ndarray, window: np.ndarray) -> np.ndarray:
    """
    Computes the Inverse Modified Discrete Cosine Transform (IMDCT).

    Args:
        mdct_coeffs: A NumPy array of MDCT coefficients (length N/2).
        window: A NumPy array representing the window to be applied (length N).

    Returns:
        A NumPy array of reconstructed audio samples (length N).
        Formula: y_n = (2/N) * window[n] * sum_{k=0}^{N/2-1} mdct_coeffs[k] * cos(pi/N * (n + 0.5 + N/2) * (k + 0.5))
                 for n = 0, ..., N-1.
    """
    M = mdct_coeffs.shape[0] # N/2
    if M != NUM_MDCT_COEFFS:
        raise ValueError(f"Number of MDCT coefficients must be {NUM_MDCT_COEFFS}, but got {M}")
    
    N = M * 2 # MDCT_FRAME_SIZE
    if window.shape[0] != N:
        raise ValueError(f"Window length must be {N}, but got {window.shape[0]}")

    output_signal = np.zeros(N)
    
    # Precompute (k + 0.5) terms
    k_factor = np.arange(M) + 0.5

    for n in range(N):
        n_factor = n + 0.5 + M # M is N/2
        cos_term = np.cos(np.pi / N * n_factor * k_factor)
        sum_coeffs = np.sum(mdct_coeffs * cos_term)
        output_signal[n] = sum_coeffs
        
    # Apply window and scaling factor
    # The window must be applied again in IMDCT for perfect reconstruction (TDAC).
    output_signal = (2.0 / N) * window * output_signal 
    
    return output_signal
