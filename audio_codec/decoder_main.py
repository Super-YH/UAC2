import argparse
import numpy as np
import os
import time # For simple progress or timing

from audio_codec.utils import write_wav
from audio_codec.ms_processing import apply_inverse_ms_processing
from audio_codec.mdct import sine_window, imdct
from audio_codec.decoder import decode_frame_payload # Core decoding logic
from audio_codec.ppmd_coder import PPMdCoder
from audio_codec.bitstream import read_frame_from_bitstream
from audio_codec.globals import (
    SAMPLE_RATE as CODEC_SAMPLE_RATE, # Use specific name
    MDCT_FRAME_SIZE, 
    NEW_SAMPLES_PER_FRAME,
    NUM_MDCT_COEFFS # For error handling (creating zero coeffs)
)

def decode_bitstream_file(input_filepath: str, output_filepath: str):
    """
    Decodes a bitstream file using the custom audio codec and saves as WAV.
    """
    print(f"Starting decoding for: {input_filepath}")

    try:
        # Initialization
        ppmd_coder = PPMdCoder()
        mdct_win = sine_window(MDCT_FRAME_SIZE)

        # Overlap buffers for IMDCT (store second half of previous IMDCT output)
        overlap_ch1 = np.zeros(NEW_SAMPLES_PER_FRAME)
        overlap_ch2 = np.zeros(NEW_SAMPLES_PER_FRAME) # Used only if stereo

        # Lists to store final 1024-sample audio segments for each channel
        output_audio_segments_ch1 = []
        output_audio_segments_ch2 = [] # Used only if stereo
        
        num_frames_processed = 0
        first_frame_header_info = None

    except Exception as e:
        print(f"Error during initialization: {e}")
        return

    with open(input_filepath, 'rb') as in_f:
        start_time = time.time()
        while True:
            # a. Read frame header and payload
            header_info = read_frame_from_bitstream(in_f)
            if header_info is None: # EOF or error
                break
            
            if first_frame_header_info is None:
                first_frame_header_info = header_info
                # Basic check against codec's fixed sample rate
                if header_info['sample_rate'] != CODEC_SAMPLE_RATE:
                    print(f"Warning: Bitstream sample rate {header_info['sample_rate']}Hz "
                          f"differs from codec's configured {CODEC_SAMPLE_RATE}Hz. "
                          "Output WAV will use bitstream's rate for this frame.")
            
            # Ensure consistency if multiple rates were allowed by bitstream.
            # For now, we assume all frames have the same rate as the first.
            current_frame_sample_rate = header_info['sample_rate']


            # b. Extract payload
            ppmd_compressed_payload = header_info['payload']
            
            # c. Determine number of channels
            num_channels = header_info['num_channels']

            # e. Decode Frame Payload to MDCT Coefficients
            ch1_mdct_coeffs, ch2_mdct_coeffs, ms_applied_flag = decode_frame_payload(
                ppmd_compressed_payload, num_channels, ppmd_coder
            )

            # Handle potential decoding errors for MDCT coefficients
            if ch1_mdct_coeffs is None:
                print(f"Warning: Failed to decode Ch1 MDCT for frame {num_frames_processed}. Using zeros.")
                ch1_mdct_coeffs = np.zeros(NUM_MDCT_COEFFS)
            if num_channels == 2 and ch2_mdct_coeffs is None:
                print(f"Warning: Failed to decode Ch2 MDCT for frame {num_frames_processed}. Using zeros.")
                ch2_mdct_coeffs = np.zeros(NUM_MDCT_COEFFS)

            # f. Per-Channel IMDCT and Overlap-Add
            current_frame_output_segments = [] # Temp list for [ch1_segment, ch2_segment (if stereo)]

            for ch_idx in range(num_channels):
                current_channel_mdct_coeffs = ch1_mdct_coeffs if ch_idx == 0 else ch2_mdct_coeffs
                overlap_buffer_current_ch = overlap_ch1 if ch_idx == 0 else overlap_ch2

                # i. IMDCT
                time_domain_block_2048 = imdct(current_channel_mdct_coeffs, mdct_win)

                # ii. Overlap-Add
                output_segment_1024 = time_domain_block_2048[:NEW_SAMPLES_PER_FRAME] + overlap_buffer_current_ch
                
                # Store new overlap for next frame
                if ch_idx == 0:
                    overlap_ch1 = time_domain_block_2048[NEW_SAMPLES_PER_FRAME:]
                else: # ch_idx == 1
                    overlap_ch2 = time_domain_block_2048[NEW_SAMPLES_PER_FRAME:]
                
                current_frame_output_segments.append(output_segment_1024)

            # g. Inverse MS Processing (applied to the 1024-sample output segments)
            final_ch1_segment_to_store = current_frame_output_segments[0]
            final_ch2_segment_to_store = None
            if num_channels == 2:
                final_ch2_segment_to_store = current_frame_output_segments[1] # Default for L/R or MS not applied
                if ms_applied_flag: # This flag comes from decode_frame_payload
                    # apply_inverse_ms_processing expects (Mid, Side, True) or (L, R, False)
                    # If ms_applied_flag is True, current_frame_output_segments[0] is Mid, [1] is Side.
                    final_ch1_segment_to_store, final_ch2_segment_to_store = apply_inverse_ms_processing(
                        current_frame_output_segments[0], # Mid if ms_applied
                        current_frame_output_segments[1], # Side if ms_applied
                        is_ms_applied=True # ms_applied_flag is the direct indicator
                    )
            
            output_audio_segments_ch1.append(final_ch1_segment_to_store)
            if num_channels == 2 and final_ch2_segment_to_store is not None:
                output_audio_segments_ch2.append(final_ch2_segment_to_store)
            
            num_frames_processed += 1
            if num_frames_processed % 100 == 0:
                print(f"Processed frame {num_frames_processed}...")
        
        end_time = time.time()
        print(f"Decoding finished. Processed {num_frames_processed} frames.")
        print(f"Total processing time: {end_time - start_time:.2f} seconds.")

    if not output_audio_segments_ch1:
        print("No audio data was decoded. Output file will not be written.")
        return

    # Concatenate all segments and write WAV
    final_audio_ch1 = np.concatenate(output_audio_segments_ch1)
    output_wav_data = final_audio_ch1 # Default for mono

    if first_frame_header_info and first_frame_header_info['num_channels'] == 2:
        if output_audio_segments_ch2: # Ensure there's data for CH2
            final_audio_ch2 = np.concatenate(output_audio_segments_ch2)
            # Ensure both channels have same length (can differ slightly due to error handling)
            min_len = min(len(final_audio_ch1), len(final_audio_ch2))
            output_wav_data = np.stack((final_audio_ch1[:min_len], final_audio_ch2[:min_len]), axis=-1)
        else: # Should not happen if num_channels was 2 consistently
            print("Warning: Stereo indicated but CH2 data is missing. Writing mono.")
            # output_wav_data remains final_audio_ch1

    # Determine sample rate for writing WAV: use from the first frame or default.
    wav_sample_rate = first_frame_header_info['sample_rate'] if first_frame_header_info else CODEC_SAMPLE_RATE
    
    try:
        write_wav(output_filepath, output_wav_data, wav_sample_rate)
        print(f"Output WAV file written to: {output_filepath}")
    except Exception as e:
        print(f"Error writing WAV file: {e}")


def main():
    parser = argparse.ArgumentParser(description="Decodes a custom audio bitstream file to WAV.")
    parser.add_argument("input", help="Path to the input bitstream file.")
    parser.add_argument("output", help="Path to the output WAV file.")
    
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found.")
        return

    decode_bitstream_file(args.input, args.output)

if __name__ == "__main__":
    main()
```
