import struct

# Frame Header Constants
SYNC_WORD = 0xACDCCODE  # Example Sync Word
SYNC_WORD_BYTES = SYNC_WORD.to_bytes(4, 'big')

# Sampling Rate Index (as per problem description, assuming these map to actual rates)
# 00: 48kHz, 01: 44.1kHz, 10: 32kHz, 11: Reserved
SAMPLE_RATE_MAP_TO_IDX = {
    48000: 0b00,
    44100: 0b01,
    32000: 0b10,
    # Add other rates if needed
}
SAMPLE_RATE_IDX_TO_MAP = {v: k for k, v in SAMPLE_RATE_MAP_TO_IDX.items()}
SAMPLE_RATE_BITS = 2

# Channel Mode (as per problem description)
# 0: Mono, 1: Stereo (which could be L/R or M/S based on further flags in payload)
CHANNEL_MODE_MONO = 0
CHANNEL_MODE_STEREO = 1
CHANNEL_MODE_BITS = 1


def write_frame_to_bitstream(output_stream, frame_payload: bytes, sample_rate: int, num_channels: int):
    """
    Writes a complete frame (header + payload) to the output bitstream.

    Args:
        output_stream: A file-like object opened in binary write mode.
        frame_payload: The (PPMd compressed) payload bytes.
        sample_rate: The actual sample rate (e.g., 48000).
        num_channels: Number of channels (1 for mono, 2 for stereo).
    """
    if not hasattr(output_stream, 'write'):
        raise TypeError("output_stream must be a file-like object with a write method.")
    if not isinstance(frame_payload, bytes):
        raise TypeError("frame_payload must be bytes.")
    if sample_rate not in SAMPLE_RATE_MAP_TO_IDX:
        raise ValueError(f"Unsupported sample rate: {sample_rate}")
    if num_channels not in [1, 2]:
        raise ValueError(f"Unsupported number of channels: {num_channels}")

    # 1. Sync Word (32 bits)
    output_stream.write(SYNC_WORD_BYTES)

    # 2. Frame Payload Length (16 bits)
    # Length of the frame_payload in bytes. Max 65535 bytes.
    payload_length = len(frame_payload)
    if payload_length > 65535:
        raise ValueError(f"Frame payload too large ({payload_length} bytes), max is 65535.")
    # ">H" means big-endian, unsigned short (2 bytes)
    output_stream.write(struct.pack(">H", payload_length))

    # 3. Metadata byte (8 bits total)
    #   - Sampling frequency index (2 bits)
    #   - Channel mode (1 bit)
    #   - Reserved (5 bits)
    sample_rate_idx = SAMPLE_RATE_MAP_TO_IDX[sample_rate]
    channel_mode_bit = CHANNEL_MODE_MONO if num_channels == 1 else CHANNEL_MODE_STEREO
    
    # Construct metadata byte:
    # Bits: [SR1 SR0 CM R R R R R] (SR = Sample Rate, CM = Channel Mode, R = Reserved)
    # Example: SR_idx = 0b00 (48kHz), CM = 0b1 (Stereo)
    # metadata_byte = (0b00 << 6) | (0b1 << 5) | 0b00000 = 0b00100000 = 0x20
    metadata_byte = (sample_rate_idx << (8 - SAMPLE_RATE_BITS))  # Shift SR to MSB side
    metadata_byte |= (channel_mode_bit << (8 - SAMPLE_RATE_BITS - CHANNEL_MODE_BITS))
    # Reserved bits are 0 by default if metadata_byte started at 0.

    output_stream.write(metadata_byte.to_bytes(1, 'big'))

    # 4. Frame Payload
    output_stream.write(frame_payload)

    # 5. CRC (16 bits) - Skipped for now as per instruction


def read_frame_from_bitstream(input_stream) -> dict | None:
    """
    Reads a complete frame (header + payload) from the input bitstream.

    Args:
        input_stream: A file-like object opened in binary read mode.

    Returns:
        A dictionary `{"payload": bytes, "sample_rate": int, "num_channels": int, "payload_length": int}`
        or `None` if EOF is encountered cleanly (especially at the start of a frame) or sync word mismatch.
    """
    if not hasattr(input_stream, 'read'):
        raise TypeError("input_stream must be a file-like object with a read method.")

    # 1. Read and Verify Sync Word (32 bits)
    sync_bytes_read = input_stream.read(4)
    if not sync_bytes_read: # EOF
        return None
    if len(sync_bytes_read) < 4: # Partial read, could be EOF or error
        # print("Warning: Partial read for sync word, assuming EOF or corrupted stream.")
        return None 
    if sync_bytes_read != SYNC_WORD_BYTES:
        # This could be a loss of sync, or end of stream with padding.
        # For robust streaming, one might search for next sync word.
        # For file-based, this is likely an error or end.
        # print(f"Error: Sync word mismatch. Expected {SYNC_WORD_BYTES.hex()}, got {sync_bytes_read.hex()}")
        # To allow testing with concatenated files, maybe don't raise, just return None.
        raise ValueError(f"Sync word mismatch. Expected {SYNC_WORD_BYTES.hex()}, got {sync_bytes_read.hex()}")


    # 2. Read Frame Payload Length (16 bits)
    len_bytes = input_stream.read(2)
    if not len_bytes or len(len_bytes) < 2:
        # print("Warning: Could not read payload length, assuming EOF or corrupted stream.")
        return None 
    payload_length = struct.unpack(">H", len_bytes)[0]

    # 3. Read Metadata byte (8 bits)
    metadata_byte_read = input_stream.read(1)
    if not metadata_byte_read:
        # print("Warning: Could not read metadata byte, assuming EOF or corrupted stream.")
        return None
    metadata_byte = int.from_bytes(metadata_byte_read, 'big')

    # Extract info from metadata_byte:
    # SR_idx is the first 2 bits
    sample_rate_idx = (metadata_byte >> (8 - SAMPLE_RATE_BITS)) & ((1 << SAMPLE_RATE_BITS) - 1)
    # CM_bit is the next 1 bit
    channel_mode_bit = (metadata_byte >> (8 - SAMPLE_RATE_BITS - CHANNEL_MODE_BITS)) & ((1 << CHANNEL_MODE_BITS) -1)

    sample_rate = SAMPLE_RATE_IDX_TO_MAP.get(sample_rate_idx)
    if sample_rate is None:
        raise ValueError(f"Invalid sample rate index: {sample_rate_idx}")

    num_channels = 1 if channel_mode_bit == CHANNEL_MODE_MONO else 2

    # 4. Read Frame Payload
    frame_payload = input_stream.read(payload_length)
    if len(frame_payload) < payload_length:
        # print(f"Warning: Could not read full payload. Expected {payload_length}, got {len(frame_payload)}.")
        # This is an error or truncated stream.
        raise EOFError(f"Truncated payload. Expected {payload_length} bytes, got {len(frame_payload)}.")

    # 5. CRC - Skipped

    return {
        "payload": frame_payload,
        "sample_rate": sample_rate,
        "num_channels": num_channels,
        "payload_length": payload_length,
        # For convenience, also pass back raw indices if needed by other parts of system
        "sample_rate_idx": sample_rate_idx, 
        "channel_mode_bit": channel_mode_bit 
    }

if __name__ == '__main__':
    import io

    # Test write_frame_to_bitstream and read_frame_from_bitstream
    print("Starting bitstream tests...")
    dummy_payload = b"This is a test payload."
    sr = 48000
    n_ch = 2

    # Write
    buffer = io.BytesIO()
    write_frame_to_bitstream(buffer, dummy_payload, sr, n_ch)
    buffer.seek(0) # Rewind to read
    
    # Read
    read_data = read_frame_from_bitstream(buffer)

    assert read_data is not None, "Failed to read data."
    assert read_data["payload"] == dummy_payload, "Payload mismatch."
    assert read_data["sample_rate"] == sr, "Sample rate mismatch."
    assert read_data["num_channels"] == n_ch, "Num channels mismatch."
    assert read_data["payload_length"] == len(dummy_payload), "Payload length mismatch."
    print("Bitstream read/write test successful!")

    # Test with multiple frames
    buffer_multi = io.BytesIO()
    payload1 = b"Payload 1"
    payload2 = b"Payload 2, slightly longer"
    write_frame_to_bitstream(buffer_multi, payload1, 48000, 1)
    write_frame_to_bitstream(buffer_multi, payload2, 44100, 2)
    buffer_multi.seek(0)

    frame1_data = read_frame_from_bitstream(buffer_multi)
    assert frame1_data is not None and frame1_data["payload"] == payload1
    print(f"Frame 1: SR={frame1_data['sample_rate']}, Ch={frame1_data['num_channels']}, Len={frame1_data['payload_length']}")

    frame2_data = read_frame_from_bitstream(buffer_multi)
    assert frame2_data is not None and frame2_data["payload"] == payload2
    print(f"Frame 2: SR={frame2_data['sample_rate']}, Ch={frame2_data['num_channels']}, Len={frame2_data['payload_length']}")
    
    # Test EOF
    eof_data = read_frame_from_bitstream(buffer_multi)
    assert eof_data is None, "Expected None at EOF."
    print("Multi-frame and EOF test successful.")

    # Test invalid sync word (by corrupting the stream)
    buffer.seek(0)
    corrupted_stream_list = list(buffer.read())
    corrupted_stream_list[0] = 0x00 # Corrupt sync word
    corrupted_buffer = io.BytesIO(bytes(corrupted_stream_list))
    try:
        read_frame_from_bitstream(corrupted_buffer)
        print("Error: Sync word mismatch was not detected.")
    except ValueError as e:
        if "Sync word mismatch" in str(e):
            print(f"Sync word mismatch correctly detected: {e}")
        else:
            raise
    print("All bitstream tests complete.")
```
