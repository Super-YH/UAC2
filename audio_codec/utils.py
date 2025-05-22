import numpy as np
import scipy.io.wavfile as wavfile

def read_wav(filepath: str) -> tuple[np.ndarray, int]:
    """Reads a WAV file and returns the audio data as a NumPy array and the sample rate.

    Handles mono/stereo. Normalizes audio data to be between -1.0 and 1.0.
    """
    sample_rate, data = wavfile.read(filepath)

    # Normalize to -1.0 to 1.0
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.uint8: # 8-bit WAV files are usually unsigned
        data = (data.astype(np.float32) - 128.0) / 128.0
    
    # If stereo, average the channels to make it mono for now
    # TODO: Decide on proper stereo handling
    if data.ndim > 1 and data.shape[1] == 2:
        data = np.mean(data, axis=1)
        
    return data, sample_rate

def write_wav(filepath: str, data: np.ndarray, sample_rate: int):
    """Writes a NumPy array to a WAV file. 
    
    Data is assumed to be between -1.0 and 1.0 and will be converted to 16-bit PCM for saving.
    """
    # Convert to 16-bit PCM
    data_int16 = (data * 32767.0).astype(np.int16)
    wavfile.write(filepath, sample_rate, data_int16)
