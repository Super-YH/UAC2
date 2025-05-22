import math

# Constants for serialization (matching quantization module definitions)
# For High Importance Bands
NUM_CEPSTRUM_COEFFS_HIGH = 8
BITS_PER_CEPSTRUM_COEFF_HIGH = 4 # 2^4 = 16 levels
PVQ_NORMALIZATION_GAIN_BITS = 5  # 2^5 = 32 levels
PVQ_PULSES_BITS = 5              # For signaling the number of pulses (4-32 range -> 28 values, needs 5 bits)
                                 # Actual pulse count values are 4 to 32. Encode (value - 4).
PVQ_INDEX_BITS = 16 # Placeholder: Assuming PVQ index from placeholder fits in 16 bits.
PVQ_NUM_SUBVECTORS_BITS = 6 # Max number of subvectors (e.g. 1024 / min_clip_size like 16 = 64, needs 6 bits)


# For Medium Importance Bands
NUM_CEPSTRUM_COEFFS_MEDIUM = 4
BITS_PER_CEPSTRUM_COEFF_MEDIUM = 3 # 2^3 = 8 levels

# Importance mapping
IMPORTANCE_MAP_TO_BITS = {"low": 0b00, "medium": 0b01, "high": 0b10}
IMPORTANCE_BITS_TO_MAP = {v: k for k, v in IMPORTANCE_MAP_TO_BITS.items()}
IMPORTANCE_NUM_BITS = 2


class BitWriter:
    def __init__(self):
        self._buffer = bytearray()
        self._current_byte = 0
        self._bit_position = 0  # Next bit to write (0-7, from MSB to LSB)

    def write_bits(self, value: int, num_bits: int):
        if num_bits < 0:
            raise ValueError("Number of bits cannot be negative.")
        if num_bits == 0:
            return
        if value < 0 or value >= (1 << num_bits):
            # This check is important to ensure value fits in num_bits
            raise ValueError(f"Value {value} does not fit in {num_bits} bits.")

        for i in range(num_bits - 1, -1, -1):  # Iterate from MSB of value to LSB
            bit = (value >> i) & 1
            self._current_byte |= (bit << (7 - self._bit_position))
            self._bit_position += 1
            if self._bit_position == 8:
                self._buffer.append(self._current_byte)
                self._current_byte = 0
                self._bit_position = 0
    
    def write_variable_length_integer(self, value: int):
        """
        Simple variable length encoding for non-negative integers.
        Writes 7 bits of the number and a continuation bit (MSB).
        If MSB is 1, more bytes follow. If 0, this is the last byte.
        (Adapted from Google Protocol Buffers Varint)
        """
        if value < 0:
            raise ValueError("Variable length integer encoding supports only non-negative values.")
        
        while True:
            byte_val = value & 0x7F  # Get the last 7 bits
            value >>= 7
            if value == 0:
                self.write_bits(byte_val, 8) # Write the 7 bits (MSB=0 implicitly by write_bits if num_bits=7, but here we ensure full byte)
                                             # Or, more directly: self._buffer.append(byte_val) and handle padding
                break
            else:
                self.write_bits(byte_val | 0x80, 8) # Set MSB to 1 and write 8 bits

    def write_bytes(self, byte_data: bytes):
        """Appends raw bytes. Must be byte-aligned."""
        if self._bit_position != 0:
            raise ValueError("Cannot write raw bytes when not byte-aligned. Call flush_byte() first.")
        self._buffer.extend(byte_data)

    def flush_byte(self):
        """Flushes the current byte to the buffer if it contains unwritten bits."""
        if self._bit_position > 0:
            self._buffer.append(self._current_byte)
            self._current_byte = 0
            self._bit_position = 0
            
    def get_bytes(self) -> bytes:
        self.flush_byte() # Ensure any pending bits are written
        return bytes(self._buffer)


class BitReader:
    def __init__(self, byte_data: bytes):
        self._buffer = byte_data
        self._byte_index = 0
        self._bit_position = 0  # Next bit to read (0-7, from MSB to LSB)

    def read_bits(self, num_bits: int) -> int:
        if num_bits < 0:
            raise ValueError("Number of bits cannot be negative.")
        if num_bits == 0:
            return 0
        
        value = 0
        for _ in range(num_bits):
            if self._byte_index >= len(self._buffer) and self._bit_position ==0 : # Allow reading last byte fully
                 # Check if trying to read beyond buffer AFTER processing current byte
                if self._byte_index >= len(self._buffer) :
                     raise EOFError("Attempt to read beyond buffer_writer length.")


            current_byte_val = self._buffer[self._byte_index]
            bit = (current_byte_val >> (7 - self._bit_position)) & 1
            value = (value << 1) | bit
            self._bit_position += 1
            if self._bit_position == 8:
                self._byte_index += 1
                self._bit_position = 0
        return value

    def read_variable_length_integer(self) -> int:
        """Reads a variable length encoded non-negative integer."""
        value = 0
        shift = 0
        while True:
            byte_val = self.read_bits(8) # Read a full byte
            value |= (byte_val & 0x7F) << shift
            shift += 7
            if not (byte_val & 0x80): # If MSB is 0, this is the last byte
                break
            if shift > 64: # Protect against malformed data (too many bytes for int64)
                raise ValueError("Variable length integer too long.")
        return value

    def read_bytes(self, num_bytes: int) -> bytes:
        """Reads specified number of bytes. Must be byte-aligned."""
        if self._bit_position != 0:
            raise ValueError("Cannot read raw bytes when not byte-aligned.")
        if self._byte_index + num_bytes > len(self._buffer):
            raise EOFError("Attempt to read beyond buffer length.")
        
        data = self._buffer[self._byte_index : self._byte_index + num_bytes]
        self._byte_index += num_bytes
        return data

    def is_aligned(self) -> bool:
        return self._bit_position == 0

    def has_more_data(self) -> bool:
        return self._byte_index < len(self._buffer) or \
               (self._byte_index == len(self._buffer) -1 and self._bit_position < 8 and self._bit_position !=0) if len(self._buffer) > 0 else False


def serialize_frame_data(frame_data: dict) -> bytes:
    writer = BitWriter()

    is_stereo = frame_data.get("is_stereo", False) # Default to mono if not specified
    # No explicit bit for is_stereo, it's implicit from channel count in bitstream header

    if is_stereo:
        ms_applied = frame_data.get("ms_applied", False) # Default to False if not specified for stereo
        writer.write_bits(1 if ms_applied else 0, 1)

    num_channels = len(frame_data["channels"])

    for channel_idx in range(num_channels):
        channel_content = frame_data["channels"][channel_idx]
        band_importances = channel_content["band_importances"]
        
        for band_idx, importance_str in enumerate(band_importances):
            # Write importance classification
            importance_bits = IMPORTANCE_MAP_TO_BITS[importance_str]
            writer.write_bits(importance_bits, IMPORTANCE_NUM_BITS)

            band_specific_data = channel_content["band_data"][band_idx]
            # Verify consistency, if needed:
            # assert band_specific_data["importance"] == importance_str
            
            if importance_str == "high":
                cep_indices = band_specific_data["cepstrum_indices"]
                for idx in cep_indices: # Should be NUM_CEPSTRUM_COEFFS_HIGH (8)
                    writer.write_bits(idx, BITS_PER_CEPSTRUM_COEFF_HIGH)
                
                writer.write_bits(band_specific_data["gain_index"], PVQ_NORMALIZATION_GAIN_BITS)
                
                # Pulse count (4-32 range) is encoded as 0-28
                pulse_count_val = band_specific_data["pvq_pulse_count"]
                encoded_pulse_count = pulse_count_val - 4 
                writer.write_bits(encoded_pulse_count, PVQ_PULSES_BITS)

                # PVQ indices (list of ints from placeholder)
                num_pvq_indices = len(band_specific_data["pvq_indices"])
                writer.write_bits(num_pvq_indices, PVQ_NUM_SUBVECTORS_BITS) # Write the count of indices
                for pvq_idx in band_specific_data["pvq_indices"]:
                     writer.write_bits(pvq_idx, PVQ_INDEX_BITS) # Fixed size for now
                    # writer.write_variable_length_integer(pvq_idx) # Alternative

            elif importance_str == "medium":
                cep_indices = band_specific_data["cepstrum_indices"]
                for idx in cep_indices: # Should be NUM_CEPSTRUM_COEFFS_MEDIUM (4)
                    writer.write_bits(idx, BITS_PER_CEPSTRUM_COEFF_MEDIUM)
            
            # "low" importance bands write nothing further.
            
    return writer.get_bytes()


def deserialize_frame_data(byte_stream: bytes, num_channels: int, num_bands: int = 16) -> dict:
    reader = BitReader(byte_stream)
    frame_data = {"channels": []}

    if num_channels == 2:
        frame_data["is_stereo"] = True
        frame_data["ms_applied"] = bool(reader.read_bits(1))
    elif num_channels == 1:
        frame_data["is_stereo"] = False
        frame_data["ms_applied"] = None # Not applicable for mono
    else:
        raise ValueError(f"Unsupported number of channels: {num_channels}")

    for _ in range(num_channels):
        channel_data = {"band_importances": [], "band_data": []}
        for band_idx in range(num_bands): # Assuming fixed number of bands (e.g., 16 from globals)
            importance_bits = reader.read_bits(IMPORTANCE_NUM_BITS)
            importance_str = IMPORTANCE_BITS_TO_MAP[importance_bits]
            channel_data["band_importances"].append(importance_str)

            current_band_data = {"importance": importance_str}

            if importance_str == "high":
                current_band_data["cepstrum_indices"] = [
                    reader.read_bits(BITS_PER_CEPSTRUM_COEFF_HIGH) for _ in range(NUM_CEPSTRUM_COEFFS_HIGH)
                ]
                current_band_data["gain_index"] = reader.read_bits(PVQ_NORMALIZATION_GAIN_BITS)
                
                encoded_pulse_count = reader.read_bits(PVQ_PULSES_BITS)
                current_band_data["pvq_pulse_count"] = encoded_pulse_count + 4 # Decode from 0-28 to 4-32

                # Assuming 1 PVQ index per band for placeholder (as PVQ splits are internal to quantizer)
                # If PVQ returns a list of indices per band, this needs adjustment
                # The quantizer returns "pvq_codeword_indices" which is a list.
                # For simplicity, let's assume a fixed number of PVQ subvectors for now, e.g. 1
                # This needs to align with how quantize_high_importance_band structures its output.
                # Let's assume quantize_high_importance_band's "pvq_codeword_indices" is what we store.
                # The placeholder pvq_encode in quantization_high.py currently does:
                # pvq_codeword_indices.append(pvq_idx)
                # This means it's a list. How many elements?
                # num_subvectors = (K_band + PVQ_MAX_DIMENSION_CLIP - 1) // PVQ_MAX_DIMENSION_CLIP
                # This is variable. This is a problem for fixed bit-rate deserialization here.
                # Simplification: Assume only ONE PVQ index is written/read for now for high bands
                # This would mean PVQ_MAX_DIMENSION_CLIP >= max band length.
                # To properly handle this, we would need to write num_subvectors first, then loop.
                # For now, let's assume only one subvector, so one index.
                # TODO: Revisit PVQ index serialization if bands can be larger than PVQ_MAX_DIMENSION_CLIP.
                # If `quantize_high_importance_band` always returns a list for `pvq_indices`,
                # even if it's `[single_index]`:
                
                # Hack for now: read a fixed number of PVQ indices, e.g., 1.
                # This needs to be made dynamic or defined by number of subvectors.
                # For a placeholder, we'll assume 1 index.
                num_pvq_indices_to_read = 1 # Placeholder!
                current_band_data["pvq_indices"] = [
                    reader.read_bits(PVQ_INDEX_BITS) for _ in range(num_pvq_indices_to_read)
                ]
                # current_band_data["pvq_indices"] = [
                #    reader.read_variable_length_integer() for _ in range(num_pvq_indices_to_read)
                # ]

            elif importance_str == "medium":
                current_band_data["cepstrum_indices"] = [
                    reader.read_bits(BITS_PER_CEPSTRUM_COEFF_MEDIUM) for _ in range(NUM_CEPSTRUM_COEFFS_MEDIUM)
                ]
            
            # "low" importance bands have no further data.
            channel_data["band_data"].append(current_band_data)
        
        frame_data["channels"].append(channel_data)
            
    return frame_data
```
