import unittest
import numpy as np # Not strictly needed for all tests here, but common.
from audio_codec.serialization import (
    BitWriter, BitReader, 
    serialize_frame_data, deserialize_frame_data,
    IMPORTANCE_MAP_TO_BITS, IMPORTANCE_BITS_TO_MAP, IMPORTANCE_NUM_BITS,
    NUM_CEPSTRUM_COEFFS_HIGH, BITS_PER_CEPSTRUM_COEFF_HIGH,
    PVQ_NORMALIZATION_GAIN_BITS, PVQ_PULSES_BITS, PVQ_INDEX_BITS, PVQ_NUM_SUBVECTORS_BITS,
    NUM_CEPSTRUM_COEFFS_MEDIUM, BITS_PER_CEPSTRUM_COEFF_MEDIUM
)
from audio_codec.globals import NUM_BANDS # Default number of bands

class TestBitWriterBitReader(unittest.TestCase):

    def test_write_read_bits_simple(self):
        writer = BitWriter()
        # Value 5 (0101), 4 bits
        writer.write_bits(0b0101, 4)
        # Value 10 (1010), 4 bits
        writer.write_bits(0b1010, 4)
        # Total 1 byte: 01011010 (0x5A)
        
        data = writer.get_bytes()
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0], 0x5A)

        reader = BitReader(data)
        val1 = reader.read_bits(4)
        self.assertEqual(val1, 0b0101)
        val2 = reader.read_bits(4)
        self.assertEqual(val2, 0b1010)

    def test_write_read_bits_multi_byte(self):
        writer = BitWriter()
        # Value 0xABCD (1010101111001101), 16 bits
        writer.write_bits(0xABCD, 16)
        # Value 0xF (1111), 4 bits
        writer.write_bits(0xF, 4)
        # Value 0x1 (001), 3 bits (will cause next write to start mid-byte)
        writer.write_bits(0x1, 3)
        # Value 0x2 (10), 2 bits
        writer.write_bits(0x2, 2) # Total 16+4+3+2 = 25 bits. 3 full bytes, 1 bit in last.

        data = writer.get_bytes()
        self.assertEqual(len(data), 4) # 3 full bytes, 1 bit in the 4th byte
        # Expected: AB CD F1 (from 1111 001) then (10)00000
        # 0xAB, 0xCD
        # Byte 3: 11110011 (0xF3) from 0xF (4 bits) and 0x1 (3 bits) + 1 bit from 0x2
        # Byte 4: 00000000 (0x00) -- wait, (1111)(001)(10) -> 11110011 0...
        # F (1111) | 1 (001) | 2 (10)
        # 1111 | 001 | 10 -> 11110011 | 00000000
        # 0xF3, 0x00
        # Expected: data[0]=0xAB, data[1]=0xCD
        # data[2] = 0xF0 | (0x1 << 1) | (0x2 >> 1) ... this is tricky to manually verify.
        # Let's trust read_bits for verification if write_bits is correct.

        reader = BitReader(data)
        val_abcd = reader.read_bits(16)
        self.assertEqual(val_abcd, 0xABCD)
        val_f = reader.read_bits(4)
        self.assertEqual(val_f, 0xF)
        val_1 = reader.read_bits(3)
        self.assertEqual(val_1, 0x1)
        val_2 = reader.read_bits(2)
        self.assertEqual(val_2, 0x2)

    def test_variable_length_integer(self):
        writer = BitWriter()
        test_numbers = [0, 1, 127, 128, 255, 256, 16383, 16384, 2097151, 2097152]
        for num in test_numbers:
            writer.write_variable_length_integer(num)
        
        data = writer.get_bytes()
        reader = BitReader(data)
        for num in test_numbers:
            decoded_num = reader.read_variable_length_integer()
            self.assertEqual(decoded_num, num, f"Varint failed for {num}")

    def test_bit_reader_eof(self):
        writer = BitWriter()
        writer.write_bits(0xFF, 8) # 1 byte
        data = writer.get_bytes()
        
        reader = BitReader(data)
        reader.read_bits(8)
        with self.assertRaises(EOFError):
            reader.read_bits(1) # Try to read past end

        reader_empty = BitReader(b"")
        with self.assertRaises(EOFError):
            reader_empty.read_bits(1)


class TestFrameSerialization(unittest.TestCase):

    def _create_sample_band_data(self, importance: str, band_length: int):
        """Helper to create sample data for a single band."""
        band_data = {"importance": importance, "band_length": band_length} # Store for verification
        if importance == "high":
            band_data["cepstrum_indices"] = [i % (2**BITS_PER_CEPSTRUM_COEFF_HIGH) for i in range(NUM_CEPSTRUM_COEFFS_HIGH)]
            band_data["gain_index"] = 15 # Example value
            band_data["pvq_pulse_count"] = 10 # Example value (4-32 range)
            
            # Determine num_subvectors based on a sample PVQ_MAX_DIMENSION_CLIP (e.g. 64)
            # This should match how quantizer would determine it.
            # For test simplicity, assume 1 or 2 subvectors.
            # Let's make it dependent on band_length to test this path in serializer.
            pvq_max_dim_clip_test = 64 # Example from quantization_high
            num_pvq_indices = (band_length + pvq_max_dim_clip_test - 1) // pvq_max_dim_clip_test
            if num_pvq_indices == 0 and band_length > 0 : num_pvq_indices = 1 # at least one if band has length
            if band_length == 0 : num_pvq_indices = 0

            band_data["pvq_indices"] = [j % (2**PVQ_INDEX_BITS) for j in range(num_pvq_indices)]
        elif importance == "medium":
            band_data["cepstrum_indices"] = [i % (2**BITS_PER_CEPSTRUM_COEFF_MEDIUM) for i in range(NUM_CEPSTRUM_COEFFS_MEDIUM)]
        # "low" bands have no specific data beyond importance
        return band_data

    def _compare_frame_data(self, original, deserialized):
        self.assertEqual(original.get("is_stereo"), deserialized.get("is_stereo"))
        self.assertEqual(original.get("ms_applied"), deserialized.get("ms_applied"))
        self.assertEqual(len(original["channels"]), len(deserialized["channels"]))

        for i, orig_ch in enumerate(original["channels"]):
            deser_ch = deserialized["channels"][i]
            self.assertListEqual(orig_ch["band_importances"], deser_ch["band_importances"])
            self.assertEqual(len(orig_ch["band_data"]), len(deser_ch["band_data"]))

            for j, orig_band_data_item in enumerate(orig_ch["band_data"]):
                deser_band_data_item = deser_ch["band_data"][j]
                # Compare contents of band_data_item
                # Note: deserialized band_data_item also includes 'importance' string.
                # Original sample helper also includes 'importance' and 'band_length' for comparison convenience.
                self.assertEqual(orig_band_data_item["importance"], deser_band_data_item["importance"])
                
                if orig_band_data_item["importance"] == "high":
                    self.assertListEqual(orig_band_data_item["cepstrum_indices"], deser_band_data_item["cepstrum_indices"])
                    self.assertEqual(orig_band_data_item["gain_index"], deser_band_data_item["gain_index"])
                    self.assertEqual(orig_band_data_item["pvq_pulse_count"], deser_band_data_item["pvq_pulse_count"])
                    self.assertListEqual(orig_band_data_item["pvq_indices"], deser_band_data_item["pvq_indices"])
                elif orig_band_data_item["importance"] == "medium":
                    self.assertListEqual(orig_band_data_item["cepstrum_indices"], deser_band_data_item["cepstrum_indices"])
                # No specific data for "low" to compare beyond importance.


    def test_frame_serialization_deserialization_mono(self):
        sample_frame_data_mono = {
            "is_stereo": False,
            "ms_applied": None, # Not applicable for mono
            "channels": [
                {
                    "band_importances": ["high", "medium"] + ["low"] * (NUM_BANDS - 2) if NUM_BANDS >=2 else ["high"]*NUM_BANDS,
                    "band_data": [] 
                }
            ]
        }
        # Populate band_data based on importances
        # Assume simple band lengths for testing, e.g., all 32 or use BAND_BOUNDARIES if available
        # For now, let's use a fixed band_length for simplicity in test data generation.
        test_band_length = 70 # To test PVQ subvector count > 1
        
        importances_ch0 = sample_frame_data_mono["channels"][0]["band_importances"]
        for k in range(NUM_BANDS):
            # Use a simple band length for testing, or derive from BAND_BOUNDARIES
            # current_band_len = BAND_BOUNDARIES[k+1] - BAND_BOUNDARIES[k] if k < len(BAND_BOUNDARIES)-1 else 32
            current_band_len = test_band_length # Fixed for this test
            sample_frame_data_mono["channels"][0]["band_data"].append(
                self._create_sample_band_data(importances_ch0[k], current_band_len)
            )
        
        num_channels_original = 1

        serialized = serialize_frame_data(sample_frame_data_mono)
        deserialized = deserialize_frame_data(serialized, num_channels_original, NUM_BANDS)
        
        self._compare_frame_data(sample_frame_data_mono, deserialized)


    def test_frame_serialization_deserialization_stereo_ms(self):
        sample_frame_data_stereo_ms = {
            "is_stereo": True,
            "ms_applied": True, # MS processing was applied
            "channels": [
                { # Channel 1 (Mid)
                    "band_importances": ["medium"] * NUM_BANDS,
                    "band_data": []
                },
                { # Channel 2 (Side)
                    "band_importances": ["low"] * NUM_BANDS,
                    "band_data": []
                }
            ]
        }
        test_band_length = 20 # All bands same length for this test
        for ch_idx in range(2):
            importances_ch = sample_frame_data_stereo_ms["channels"][ch_idx]["band_importances"]
            for k in range(NUM_BANDS):
                sample_frame_data_stereo_ms["channels"][ch_idx]["band_data"].append(
                    self._create_sample_band_data(importances_ch[k], test_band_length)
                )
        
        num_channels_original = 2
        serialized = serialize_frame_data(sample_frame_data_stereo_ms)
        deserialized = deserialize_frame_data(serialized, num_channels_original, NUM_BANDS)

        self._compare_frame_data(sample_frame_data_stereo_ms, deserialized)


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
```
