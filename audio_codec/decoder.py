import numpy as np
from audio_codec.ppmd_coder import PPMdCoder
from audio_codec.serialization import deserialize_frame_data
from audio_codec.quantization_high import dequantize_high_importance_band
from audio_codec.quantization_medium import dequantize_medium_importance_band
from audio_codec.quantization_low import reconstruct_low_importance_band
from audio_codec.banding import combine_bands_into_mdct_coeffs
# from audio_codec.ms_processing import apply_inverse_ms_processing # Not used in this function
from audio_codec.globals import NUM_BANDS, NUM_MDCT_COEFFS, BAND_BOUNDARIES

def get_band_length(band_idx: int) -> int:
    """Helper function to get length of a band by its index."""
    if not (0 <= band_idx < NUM_BANDS):
        raise ValueError(f"Invalid band_idx: {band_idx}")
    return BAND_BOUNDARIES[band_idx+1] - BAND_BOUNDARIES[band_idx]

def decode_frame_payload(
    ppmd_compressed_payload: bytes,
    num_channels: int,
    ppmd_coder: PPMdCoder
) -> tuple[np.ndarray | None, np.ndarray | None, bool | None]:
    """
    Decodes a single frame's payload to produce MDCT coefficients for each channel
    and the MS processing flag.

    Args:
        ppmd_compressed_payload: The byte payload for one frame (PPMd compressed).
        num_channels: Number of channels (1 or 2).
        ppmd_coder: An instance of PPMdCoder.

    Returns:
        A tuple (ch1_mdct_coeffs, ch2_mdct_coeffs, ms_applied_flag).
        ch2_mdct_coeffs is None if mono. ms_applied_flag is None if mono.
        Returns (None, None, None) if any critical error occurs.
    """
    ch1_mdct_coeffs: np.ndarray | None = None
    ch2_mdct_coeffs: np.ndarray | None = None
    ms_applied_flag: bool | None = None

    try:
        # a. PPMd Decoding
        serialized_data = ppmd_coder.decode(ppmd_compressed_payload)

        # b. Deserialization
        # Assuming NUM_BANDS is correctly imported from globals for deserialize_frame_data
        frame_data = deserialize_frame_data(serialized_data, num_channels, NUM_BANDS)
        
        if not frame_data or not frame_data.get("channels"):
            print("Decoder Error: Deserialization failed or produced no channel data.")
            return None, None, None
        
        ms_applied_flag = frame_data.get("ms_applied") # Get ms_applied from frame_data

        processed_channels_coeffs = []

        for channel_idx in range(num_channels):
            if channel_idx >= len(frame_data["channels"]):
                print(f"Decoder Error: Not enough channel data in frame_data for channel {channel_idx}.")
                # Fill with zeros if a channel is missing, or return error
                # For robustness, create zero MDCT coeffs for this channel
                processed_channels_coeffs.append(np.zeros(NUM_MDCT_COEFFS))
                continue

            current_channel_data = frame_data['channels'][channel_idx]
            current_channel_importances = current_channel_data['band_importances']
            
            # c. Initialize list to store reconstructed bands for this channel
            reconstructed_bands_for_this_channel: list[np.ndarray | None] = [None] * NUM_BANDS

            # d.i. Iterate through bands
            for band_idx in range(NUM_BANDS):
                band_info = current_channel_data['band_data'][band_idx]
                importance = current_channel_importances[band_idx] # Or band_info['importance']
                
                # Determine band_length
                # For High/Medium, band_length should be in band_info from deserialization.
                # For Low, reconstruct_low_importance_band calculates it or takes it as arg.
                # Use helper for consistency.
                current_band_length = get_band_length(band_idx)

                if importance == "high":
                    # Ensure band_info has 'band_length' if dequantizer expects it.
                    # dequantize_high_importance_band takes quant_data which includes 'band_length'.
                    # deserialize_frame_data should populate this.
                    # Let's check what `deserialize_frame_data` puts in band_info for 'high'.
                    # It does not explicitly add 'band_length' for 'high' bands in serialization.py.
                    # This needs to be added to band_info during deserialization or passed.
                    # For now, `dequantize_high_importance_band` uses K_band = quant_data["band_length"]
                    # So, `band_info` must contain `band_length`.
                    if "band_length" not in band_info: # Add it if missing from deserializer for high.
                        band_info["band_length"] = current_band_length
                    reconstructed_coeffs = dequantize_high_importance_band(band_info)
                elif importance == "medium":
                    # dequantize_medium_importance_band also expects 'band_length' in quant_data.
                    # `deserialize_frame_data` in serialization.py should populate this.
                    # It does: quantize_medium_importance_band returns it, and it's stored.
                    if "band_length" not in band_info: # Should be there for medium
                         print(f"Warning: band_length missing for medium band {band_idx}. Using calculated.")
                         band_info["band_length"] = current_band_length
                    reconstructed_coeffs = dequantize_medium_importance_band(band_info)
                elif importance == "low":
                    reconstructed_coeffs = reconstruct_low_importance_band(
                        band_idx, 
                        current_band_length, # Pass calculated band length
                        reconstructed_bands_for_this_channel, # Pass list of prior bands in this channel
                        current_channel_importances
                    )
                else:
                    print(f"Decoder Error: Unknown importance '{importance}' for band {band_idx}. Nullifying band.")
                    reconstructed_coeffs = np.zeros(current_band_length)
                
                if reconstructed_coeffs is None or reconstructed_coeffs.shape[0] != current_band_length :
                    print(f"Decoder Error: Band {band_idx} (importance {importance}) recon failed or wrong length. Nullifying.")
                    reconstructed_coeffs = np.zeros(current_band_length)

                reconstructed_bands_for_this_channel[band_idx] = reconstructed_coeffs
            
            # d.ii. Combine Bands for the Channel
            # Ensure all bands are ndarrays before combining
            final_bands_for_channel = []
            for i, band_coeffs in enumerate(reconstructed_bands_for_this_channel):
                if band_coeffs is None:
                    # This should not happen if logic above is correct (nullifies on error)
                    print(f"Decoder Warning: Band {i} is None before combining. Nullifying.")
                    final_bands_for_channel.append(np.zeros(get_band_length(i)))
                else:
                    final_bands_for_channel.append(band_coeffs)
            
            channel_mdct_coeffs = combine_bands_into_mdct_coeffs(final_bands_for_channel)
            processed_channels_coeffs.append(channel_mdct_coeffs)

        # Assign to ch1_mdct_coeffs, ch2_mdct_coeffs
        if len(processed_channels_coeffs) > 0:
            ch1_mdct_coeffs = processed_channels_coeffs[0]
        if len(processed_channels_coeffs) > 1:
            ch2_mdct_coeffs = processed_channels_coeffs[1]

        # e. Return reconstructed MDCT coefficients and ms_applied_flag
        return ch1_mdct_coeffs, ch2_mdct_coeffs, ms_applied_flag

    except Exception as e:
        print(f"Decoder Error: An exception occurred during decode_frame_payload: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

if __name__ == '__main__':
    print("Running basic tests for decoder.py...")
    # Setup mock objects and data for testing
    # This requires a significant setup:
    # - Mock PPMdCoder
    # - Sample serialized_data (output of serialize_frame_data)
    # - Expected MDCT coefficient structure

    # For now, this section will be minimal as it's complex to mock the entire chain here.
    # A more complete test would involve encoding a known signal and verifying decoded output.

    mock_ppmd_coder = PPMdCoder() # Using pass-through version

    # Example: Construct a very simple mono frame payload (highly simplified)
    # This would normally come from `serialize_frame_data`
    # Let's assume one band, medium importance, length 16.
    # Cepstrum indices: [0,0,0,0] for NUM_CEPSTRUM_COEFFS_MEDIUM=4, BITS_PER_CEPSTRUM_COEFF_MEDIUM=3
    # Importance: "medium" (0b01)
    
    # This test is non-trivial because `deserialize_frame_data` is complex.
    # A true test requires valid bitstream data.
    
    # Test case 1: Mono, 1 band (band 0, length from BAND_BOUNDARIES[1]-BAND_BOUNDARIES[0])
    # Let's assume band 0 is medium importance, length is BAND_BOUNDARIES[1] - BAND_BOUNDARIES[0]
    # And it's the only band for simplicity in this direct test (NUM_BANDS=1 for test)
    
    # Reconsider: Direct testing of decode_frame_payload here is too complex without
    # a full, correct serialization chain or extensive mocking.
    # The main functionality will be tested via the main encoder/decoder script.

    print("Decoder structure implemented. Further testing requires full pipeline.")

    # Test get_band_length
    try:
        length_band_0 = get_band_length(0)
        print(f"Length of band 0: {length_band_0} (expected from globals.BAND_BOUNDARIES)")
        assert length_band_0 == (BAND_BOUNDARIES[1] - BAND_BOUNDARIES[0])
        length_last_band = get_band_length(NUM_BANDS-1)
        assert length_last_band == (BAND_BOUNDARIES[NUM_BANDS] - BAND_BOUNDARIES[NUM_BANDS-1])
        print(f"Length of band {NUM_BANDS-1}: {length_last_band}")
    except Exception as e:
        print(f"Error in get_band_length test: {e}")


    print("Basic decoder.py tests finished.")
```
