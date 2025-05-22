import unittest
import numpy as np
import os
import scipy.io.wavfile as wavfile

# Allow access to the main scripts, assuming tests are run from project root
# or PYTHONPATH is set up.
# For simplicity, if these scripts are not directly importable,
# this test might need to use subprocess to call them.
# Assuming they can be imported for now:
from audio_codec.encoder_main import encode_wav_file
from audio_codec.decoder_main import decode_bitstream_file
from audio_codec.utils import read_wav # To read back the decoded WAV for comparison
from audio_codec.globals import SAMPLE_RATE as CODEC_SAMPLE_RATE

import tempfile

# Define a temporary directory for test files if it doesn't exist
# TEST_OUTPUT_DIR = "test_outputs" 
# if not os.path.exists(TEST_OUTPUT_DIR):
#     os.makedirs(TEST_OUTPUT_DIR)

class TestFullPipeline(unittest.TestCase):

    def _create_dummy_wav(self, filepath, duration_sec=0.1, sample_rate=CODEC_SAMPLE_RATE, num_channels=1):
        """Creates a simple WAV file for testing."""
        num_samples = int(duration_sec * sample_rate)
        amplitude = 0.5 
        frequency = 440
        t = np.linspace(0, duration_sec, num_samples, endpoint=False)
        signal_ch = (amplitude * np.sin(2 * np.pi * frequency * t)).astype(np.float32)

        if num_channels == 1:
            wav_data_float = signal_ch
        elif num_channels == 2:
            signal_ch2 = (amplitude * 0.8 * np.sin(2 * np.pi * frequency * 1.5 * t)).astype(np.float32)
            wav_data_float = np.stack((signal_ch, signal_ch2), axis=-1)
        else:
            raise ValueError("Unsupported number of channels for dummy WAV.")

        wav_data_int16 = (wav_data_float * 32767.0).astype(np.int16)
        wavfile.write(filepath, sample_rate, wav_data_int16)
        return wav_data_float # Return float version for comparison

    def run_encode_decode_test(self, test_name_prefix, num_channels):
        """Helper to run a standard encode/decode cycle using temp files."""
        
        orig_wav_fd, orig_wav_path = tempfile.mkstemp(suffix='.wav', prefix=f"{test_name_prefix}_orig_")
        os.close(orig_wav_fd) # Close file descriptor, path is what we need
        
        bitstream_fd, bitstream_path = tempfile.mkstemp(suffix='.customaudio', prefix=f"{test_name_prefix}_bits_")
        os.close(bitstream_fd)

        decoded_wav_fd, decoded_wav_path = tempfile.mkstemp(suffix='.wav', prefix=f"{test_name_prefix}_deco_")
        os.close(decoded_wav_fd)

        files_to_clean = [orig_wav_path, bitstream_path, decoded_wav_path]
        
        try:
            print(f"\nRunning integration test: {test_name_prefix} ({num_channels}-channel)")
            print(f"  Original WAV: {orig_wav_path}")
            print(f"  Bitstream: {bitstream_path}")
            print(f"  Decoded WAV: {decoded_wav_path}")

            # 1. Create original WAV
            duration = 1.0 # Use 1 second duration
            original_float_data = self._create_dummy_wav(orig_wav_path, duration_sec=duration, num_channels=num_channels)
            self.assertTrue(os.path.exists(orig_wav_path))

            # 2. Encode
            encode_wav_file(orig_wav_path, bitstream_path)
            self.assertTrue(os.path.exists(bitstream_path))
            self.assertTrue(os.path.getsize(bitstream_path) > 0)

            # 3. Decode
            decode_bitstream_file(bitstream_path, decoded_wav_path)
            self.assertTrue(os.path.exists(decoded_wav_path))
            self.assertTrue(os.path.getsize(decoded_wav_path) > 0)

            # 4. Compare
            # Read original WAV again (as float, which read_wav should provide if utils.read_wav is used)
            # Or use the returned original_float_data if its scaling matches read_wav.
            # For simplicity, re-read the saved original to ensure it's what encoder saw.
            data_orig_read, rate_orig = read_wav(orig_wav_path) 
            data_decoded, rate_decoded = read_wav(decoded_wav_path)

            self.assertEqual(rate_orig, CODEC_SAMPLE_RATE)
            self.assertEqual(rate_decoded, CODEC_SAMPLE_RATE)
            
            # Align lengths
            len_orig = data_orig_read.shape[0]
            len_decoded = data_decoded.shape[0]
            
            from audio_codec.globals import NEW_SAMPLES_PER_FRAME
            # Expected length after decoding is multiple of NEW_SAMPLES_PER_FRAME
            # due to how encoder pads and decoder outputs segments.
            expected_len_after_framing = int(np.ceil(len_orig / NEW_SAMPLES_PER_FRAME) * NEW_SAMPLES_PER_FRAME)
            self.assertAlmostEqual(len_decoded, expected_len_after_framing, delta=NEW_SAMPLES_PER_FRAME // 2, # Allow some slack
                                   msg=f"Decoded length {len_decoded} vs expected framed {expected_len_after_framing}")

            min_len = min(len_orig, len_decoded) # Compare up to the shorter length after framing considerations
            
            data_orig_aligned = data_orig_read[:min_len]
            data_decoded_aligned = data_decoded[:min_len]

            if num_channels > 1:
                 self.assertEqual(data_orig_aligned.shape[1], num_channels, "Original data channel mismatch after read.")
                 self.assertEqual(data_decoded_aligned.shape[1], num_channels, "Decoded data channel mismatch.")

            # Basic check: not all zeros (if input wasn't zero)
            if np.any(data_orig_aligned != 0):
                 self.assertFalse(np.all(data_decoded_aligned == 0), "Decoded audio is all zeros but original was not.")

            # SNR Calculation
            signal_power = np.sum(data_orig_aligned**2)
            noise = data_orig_aligned - data_decoded_aligned
            noise_power = np.sum(noise**2)
            
            snr = 10 * np.log10(signal_power / (noise_power + 1e-9)) # Epsilon for stability
            print(f"  Signal Power: {signal_power:.2f}, Noise Power: {noise_power:.2f}, SNR: {snr:.2f} dB")
            
            # Due to placeholder quantizers and pass-through PPMd, SNR might be low.
            # A very basic check could be that noise_power is not excessively larger than signal_power.
            # Or a very low SNR threshold.
            self.assertTrue(snr > -10, f"SNR is too low ({snr:.2f} dB). Check pipeline integrity.") # Very lenient

            # Correlation
            if num_channels == 1:
                correlation = np.corrcoef(data_orig_aligned.flatten(), data_decoded_aligned.flatten())[0, 1]
            else: # Stereo: correlate each channel or average
                corr_ch1 = np.corrcoef(data_orig_aligned[:,0].flatten(), data_decoded_aligned[:,0].flatten())[0,1]
                corr_ch2 = np.corrcoef(data_orig_aligned[:,1].flatten(), data_decoded_aligned[:,1].flatten())[0,1]
                correlation = (corr_ch1 + corr_ch2) / 2.0
            print(f"  Correlation: {correlation:.2f}")
            self.assertTrue(correlation > 0.1, f"Correlation is too low ({correlation:.2f}).") # Very lenient

        finally:
            # Cleanup
            for f_path in files_to_clean:
                if os.path.exists(f_path):
                    try:
                        os.remove(f_path)
                    except Exception as e:
                        print(f"Warning: Could not delete temp file {f_path}: {e}")
    
    def test_encode_decode_mono(self):
        self.run_encode_decode_test("mono_sine", num_channels=1)

    def test_encode_decode_stereo(self):
        self.run_encode_decode_test("stereo_sine", num_channels=2)

# @classmethod
# def tearDownClass(cls):
#     print(f"\nIntegration tests finished.")
    # No need to clean TEST_OUTPUT_DIR as tempfile is used.

if __name__ == '__main__':
    # This allows running the tests directly.
    # Ensure that the script can find the audio_codec package.
    # (e.g., by running from the project root or having PYTHONPATH set)
    
    # Example of how to adjust path if needed when running directly:
    # import sys
    # script_dir = os.path.dirname(__file__)
    # project_root = os.path.abspath(os.path.join(script_dir, '..')) # Assuming tests is one level down
    # if project_root not in sys.path:
    #    sys.path.insert(0, project_root)
    
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
```
