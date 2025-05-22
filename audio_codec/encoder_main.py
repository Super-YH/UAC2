import argparse
import numpy as np
import os
import time # For simple progress or timing

from audio_codec.utils import read_wav
from audio_codec.ms_processing import apply_ms_processing
from audio_codec.mdct import sine_window, mdct
from audio_codec.banding import split_mdct_coeffs_into_bands
from audio_codec.psychoacoustic import evaluate_band_importance
from audio_codec.quantization_high import quantize_high_importance_band
from audio_codec.quantization_medium import quantize_medium_importance_band
from audio_codec.quantization_low import handle_low_importance_band_encoder
from audio_codec.serialization import serialize_frame_data
from audio_codec.ppmd_coder import PPMdCoder
from audio_codec.bitstream import write_frame_to_bitstream, SAMPLE_RATE_MAP_TO_IDX # Import map for SR index
from audio_codec.globals import (
    SAMPLE_RATE as CODEC_SAMPLE_RATE, # Use specific name to avoid confusion
    MDCT_FRAME_SIZE, 
    NEW_SAMPLES_PER_FRAME, 
    NUM_BANDS,
    # BAND_BOUNDARIES # Not directly used in encoder_main, but by imported functions
)

def encode_wav_file(input_filepath: str, output_filepath: str):
    """
    Encodes a WAV file using the custom audio codec.
    """
    print(f"Starting encoding for: {input_filepath}")
    
    # 1. Initialization
    try:
        ppmd_coder = PPMdCoder() # Initialize PPMd Coder
        
        # Load WAV file
        audio_data, input_sample_rate = read_wav(input_filepath)
        
        if input_sample_rate != CODEC_SAMPLE_RATE:
            raise ValueError(
                f"Unsupported sample rate: {input_sample_rate} Hz. "
                f"Codec currently supports {CODEC_SAMPLE_RATE} Hz."
            )

        # Determine number of channels and handle mono/stereo
        if audio_data.ndim == 1:
            num_channels = 1
            # For mono, create a second dimension for consistency in processing loop
            # audio_data_ch1 = audio_data.reshape(-1, 1) # This is not right, should be (samples,)
            audio_data_ch1 = audio_data
            audio_data_ch2 = None # No second channel
            is_stereo = False
            print("Input is Mono.")
        elif audio_data.ndim == 2 and audio_data.shape[1] == 2:
            num_channels = 2
            audio_data_ch1 = audio_data[:, 0]
            audio_data_ch2 = audio_data[:, 1]
            is_stereo = True
            print("Input is Stereo.")
        else:
            raise ValueError(f"Unsupported audio format. Expected mono or stereo, got shape {audio_data.shape}")

        # MDCT window
        mdct_win = sine_window(MDCT_FRAME_SIZE)

        # Overlap buffers for MDCT (store previous *processed* 1024-sample segments)
        # These store the latter half of the previous MDCT block's *source* samples,
        # which have already undergone MS processing if applicable for that frame.
        prev_processed_samples_ch1 = np.zeros(NEW_SAMPLES_PER_FRAME)
        prev_processed_samples_ch2 = np.zeros(NEW_SAMPLES_PER_FRAME) if is_stereo else None

        total_samples = len(audio_data_ch1)
        num_frames = (total_samples + NEW_SAMPLES_PER_FRAME - 1) // NEW_SAMPLES_PER_FRAME
        
        print(f"Total samples: {total_samples}, Frames: {num_frames}")

    except FileNotFoundError:
        print(f"Error: Input file not found: {input_filepath}")
        return
    except Exception as e:
        print(f"Error during initialization: {e}")
        return

    # Open output file stream
    with open(output_filepath, 'wb') as out_f:
        start_time = time.time()
        for frame_idx in range(num_frames):
            # print(f"Processing frame {frame_idx + 1}/{num_frames}...")

            # a. Get current new samples (1024 per channel)
            start_sample = frame_idx * NEW_SAMPLES_PER_FRAME
            end_sample = start_sample + NEW_SAMPLES_PER_FRAME
            
            current_new_samples_ch1 = audio_data_ch1[start_sample:end_sample]
            
            # Pad with zeros if at the end of the file
            if len(current_new_samples_ch1) < NEW_SAMPLES_PER_FRAME:
                padding_ch1 = np.zeros(NEW_SAMPLES_PER_FRAME - len(current_new_samples_ch1))
                current_new_samples_ch1 = np.concatenate((current_new_samples_ch1, padding_ch1))

            if is_stereo:
                current_new_samples_ch2 = audio_data_ch2[start_sample:end_sample]
                if len(current_new_samples_ch2) < NEW_SAMPLES_PER_FRAME:
                    padding_ch2 = np.zeros(NEW_SAMPLES_PER_FRAME - len(current_new_samples_ch2))
                    current_new_samples_ch2 = np.concatenate((current_new_samples_ch2, padding_ch2))
            
            # d. Stereo Processing (MS Processing on new 1024-sample segments)
            ms_applied_flag = False # Default for mono or if MS not applied
            if is_stereo:
                # apply_ms_processing expects full frames, but spec says "1024 samples" for input.
                # This means we are applying MS to the *new* 1024 samples.
                processed_new_ch1_segment, processed_new_ch2_segment, ms_applied_flag = \
                    apply_ms_processing(current_new_samples_ch1, current_new_samples_ch2)
            else: # Mono
                processed_new_ch1_segment = current_new_samples_ch1
                # processed_new_ch2_segment is not used for mono

            # b. Create 2048-sample blocks for MDCT using processed segments
            mdct_input_block_ch1 = np.concatenate((prev_processed_samples_ch1, processed_new_ch1_segment))
            if is_stereo:
                mdct_input_block_ch2 = np.concatenate((prev_processed_samples_ch2, processed_new_ch2_segment))
            
            # c. Store current processed new samples for next frame's overlap
            prev_processed_samples_ch1 = processed_new_ch1_segment
            if is_stereo:
                prev_processed_samples_ch2 = processed_new_ch2_segment

            # e. Initialize frame_output_data dictionary
            frame_output_data = {
                "is_stereo": is_stereo, 
                "ms_applied": ms_applied_flag if is_stereo else None, 
                "channels": []
            }

            # f. Process each channel
            for ch_idx in range(num_channels):
                channel_pcm_data_for_mdct = mdct_input_block_ch1 if ch_idx == 0 else mdct_input_block_ch2
                
                channel_data_for_serialization = {"band_importances": [], "band_data": []}

                # i. MDCT
                mdct_coeffs = mdct(channel_pcm_data_for_mdct, mdct_win)
                
                # ii. Band Splitting
                banded_coeffs = split_mdct_coeffs_into_bands(mdct_coeffs)
                
                # iii. Band Importance Evaluation
                importances, band_energies, total_frame_energy = evaluate_band_importance(banded_coeffs)
                channel_data_for_serialization["band_importances"] = importances

                # iv. Quantization
                for band_k in range(NUM_BANDS):
                    current_band_coeffs_segment = banded_coeffs[band_k]
                    importance_k = importances[band_k]
                    
                    quant_band_data_k = {} # Holds data from quantization function
                    
                    if importance_k == "high":
                        quant_band_data_k = quantize_high_importance_band(current_band_coeffs_segment)
                    elif importance_k == "medium":
                        quant_band_data_k = quantize_medium_importance_band(current_band_coeffs_segment)
                    elif importance_k == "low":
                        # handle_low_importance_band_encoder returns None or an empty dict.
                        # The serializer needs to handle this (e.g. by writing nothing for 'low' bands
                        # beyond their importance bits).
                        _ = handle_low_importance_band_encoder(current_band_coeffs_segment)
                        # quant_band_data_k remains empty or as needed by serializer.
                        # serializer.py expects 'importance' key in band_specific_data if it processes it.
                        # For low, it only writes importance bits.
                        # So, no data beyond importance is needed for low bands in `band_data`.
                    
                    # Store the actual importance string for clarity if deserializer needs it,
                    # though it primarily uses the band_importances list.
                    # The band_data dictionary in serialize_frame_data is indexed by band_idx,
                    # and it uses band_specific_data = channel_content["band_data"][band_idx].
                    # This band_specific_data is the dict returned by quantize_X_importance_band.
                    # Let's ensure it contains 'importance' for consistency, though serialize_frame_data
                    # currently gets importance from the separate `band_importances` list.
                    # The quantizer functions do not return 'importance'.
                    # `serialize_frame_data` uses `band_specific_data = channel_content["band_data"][band_idx]`
                    # and `importance_str = band_importances[band_idx]`.
                    # So, `quant_band_data_k` just needs the quantization products (indices, etc.).
                    
                    # The `quantize_X` functions return dicts like:
                    # high: {"quantized_cepstrum_indices", "quantized_residual_gain_index", "pvq_pulse_count", "pvq_codeword_indices", "band_length"}
                    # medium: {"quantized_cepstrum_indices", "band_length"}
                    # low: returns None.
                    # `serialize_frame_data` uses `band_specific_data["pvq_indices"]` etc.
                    # The keys from quantization functions need to match.
                    # `quantize_high_importance_band` returns "pvq_codeword_indices", serializer expects "pvq_indices".
                    # This needs to be harmonized. Let's assume serializer uses "pvq_codeword_indices".
                    # (Checked serialization.py: it expects "pvq_indices". This is a mismatch.)
                    # Let's fix this by standardizing on "pvq_codeword_indices" or by renaming in serializer.
                    # For now, I'll adjust here to match what serializer expects if it's simpler.
                    # serializer.py expects: band_specific_data["pvq_indices"]
                    # quantization_high.py returns: "pvq_codeword_indices"
                    # Renaming here:
                    if importance_k == "high" and "pvq_codeword_indices" in quant_band_data_k:
                        quant_band_data_k["pvq_indices"] = quant_band_data_k.pop("pvq_codeword_indices")
                    
                    # `serialize_frame_data` also expects "cepstrum_indices" from high/medium.
                    # `quantize_high/medium_importance_band` return "quantized_cepstrum_indices".
                    # Renaming here:
                    if (importance_k == "high" or importance_k == "medium") and \
                       "quantized_cepstrum_indices" in quant_band_data_k:
                        quant_band_data_k["cepstrum_indices"] = quant_band_data_k.pop("quantized_cepstrum_indices")

                    # `quantize_high_importance_band` returns "quantized_residual_gain_index".
                    # `serialize_frame_data` expects "gain_index".
                    if importance_k == "high" and "quantized_residual_gain_index" in quant_band_data_k:
                        quant_band_data_k["gain_index"] = quant_band_data_k.pop("quantized_residual_gain_index")


                    channel_data_for_serialization["band_data"].append(quant_band_data_k)
                
                frame_output_data["channels"].append(channel_data_for_serialization)

            # g. Serialize frame data
            serialized_payload = serialize_frame_data(frame_output_data)
            
            # h. PPMd Encode
            compressed_payload = ppmd_coder.encode(serialized_payload)
            
            # i. Write to bitstream file
            # `write_frame_to_bitstream` expects actual sample rate and num_channels, not indices.
            # (Checked bitstream.py: it takes actual sample_rate and maps to index internally)
            write_frame_to_bitstream(out_f, compressed_payload, CODEC_SAMPLE_RATE, num_channels)
            
            if (frame_idx + 1) % 100 == 0 : # Simple progress update
                print(f"Processed frame {frame_idx + 1}/{num_frames}...")


        end_time = time.time()
        print(f"Encoding finished for {input_filepath}.")
        print(f"Output written to: {output_filepath}")
        print(f"Total processing time: {end_time - start_time:.2f} seconds.")

def main():
    parser = argparse.ArgumentParser(description="Encodes a WAV file using a custom audio codec.")
    parser.add_argument("input", help="Path to the input WAV file (must be 48kHz mono/stereo).")
    parser.add_argument("output", help="Path to the output bitstream file.")
    
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found.")
        return

    encode_wav_file(args.input, args.output)

if __name__ == "__main__":
    main()
```
