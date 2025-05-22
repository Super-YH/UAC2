import numpy as np

MS_ENERGY_THRESHOLD = 0.25

def apply_ms_processing(left_channel_frame: np.ndarray, right_channel_frame: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
    """
    Applies Mid-Side (MS) processing to a stereo audio frame if beneficial.

    Args:
        left_channel_frame: NumPy array of the left channel audio data for the frame.
        right_channel_frame: NumPy array of the right channel audio data for the frame.

    Returns:
        A tuple containing:
        - channel1_frame: Mid signal if MS processing was applied, else Left channel.
        - channel2_frame: Side signal if MS processing was applied, else Right channel.
        - ms_applied: Boolean flag indicating if MS processing was applied.
    """
    if not isinstance(left_channel_frame, np.ndarray) or not isinstance(right_channel_frame, np.ndarray):
        raise TypeError("Input channels must be NumPy arrays.")

    # Ensure input frames are of the same length
    if left_channel_frame.shape != right_channel_frame.shape:
        raise ValueError("Left and Right channel frames must have the same dimensions.")

    # Calculate Mid and Side signals for energy calculation
    mid_signal_for_energy = (left_channel_frame + right_channel_frame) / 2.0
    side_signal_for_energy = (left_channel_frame - right_channel_frame) / 2.0

    # Calculate energies
    s_energy = np.sum(side_signal_for_energy**2)
    m_energy = np.sum(mid_signal_for_energy**2)

    apply_ms = False
    if m_energy > 1e-9: # Avoid division by zero or near-zero
        if (s_energy / m_energy) < MS_ENERGY_THRESHOLD:
            apply_ms = True
    # If m_energy is very small or zero, it implies silence or near silence in the mid channel.
    # In such cases, MS processing might not be beneficial or could even be detrimental.
    # For example, if L = -R (perfectly out of phase), M = 0. S_energy / M_energy would be Inf.
    # If L = R = 0 (silence), M = 0, S = 0. S_energy / M_energy would be NaN.
    # If M_energy is zero, and S_energy is non-zero, it means L = -R.
    # In this case, L/R representation is likely better.
    # If M_energy and S_energy are both zero (silence), it doesn't matter.

    if apply_ms:
        # Actual Mid and Side signals to be returned
        mid_output = (left_channel_frame + right_channel_frame) / 2.0
        side_output = (left_channel_frame - right_channel_frame) / 2.0
        return mid_output, side_output, True
    else:
        return left_channel_frame, right_channel_frame, False

def apply_inverse_ms_processing(channel1_frame: np.ndarray, channel2_frame: np.ndarray, is_ms_applied: bool) -> tuple[np.ndarray, np.ndarray]:
    """
    Applies inverse Mid-Side (MS) processing to reconstruct Left/Right channels.

    Args:
        channel1_frame: Mid signal if is_ms_applied is True, else Left channel.
        channel2_frame: Side signal if is_ms_applied is True, else Right channel.
        is_ms_applied: Boolean flag indicating if MS processing was originally applied.

    Returns:
        A tuple containing:
        - left_channel_frame: Reconstructed Left channel.
        - right_channel_frame: Reconstructed Right channel.
    """
    if not isinstance(channel1_frame, np.ndarray) or not isinstance(channel2_frame, np.ndarray):
        raise TypeError("Input channels must be NumPy arrays.")

    if is_ms_applied:
        # channel1 is Mid, channel2 is Side
        left_channel = channel1_frame + channel2_frame  # L = M + S
        right_channel = channel1_frame - channel2_frame # R = M - S
        return left_channel, right_channel
    else:
        # channel1 is Left, channel2 is Right
        return channel1_frame, channel2_frame
