#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <cmath>
#include <cstdint>
#include <algorithm> // For std::min, std::max, std::sort, etc.
#include <numeric>   // For std::iota
#include <stdexcept> // For std::runtime_error, std::out_of_range
#include <cstring>   // For memcpy, strncmp
#include <map>       // For ATH table (example)

// Conditional M_PI definition for MSVC or other compilers if not in cmath
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// --- codec_params.h ---
namespace CodecParams {
    constexpr int SAMPLING_FREQUENCY = 48000;
    constexpr int MDCT_FRAME_SIZE_N = 2048;
    constexpr int NEW_SAMPLES_PER_FRAME = 1024; // N / 2
    constexpr int MDCT_COEFFS_PER_CHANNEL = MDCT_FRAME_SIZE_N / 2; // 1024

    constexpr int PPMD_MAX_CONTEXT_ORDER = 5;
    constexpr size_t PPMD_MEMORY_LIMIT = 16 * 1024 * 1024; // 16 MB

    constexpr double MS_ENERGY_RATIO_THRESHOLD = 0.25;

    const std::vector<int> BAND_OFFSETS = {
        0, 16, 32, 48, 64, 96, 128, 160, 192, 256, 320, 384, 512, 640, 768, 896, MDCT_COEFFS_PER_CHANNEL
    };
    constexpr int NUM_BANDS = 16;

    constexpr double HIGH_IMPORTANCE_P_BAND_THRESHOLD = 100.0;
    constexpr double HIGH_IMPORTANCE_ENERGY_PERCENTILE_THRESHOLD = 0.25;
    constexpr double LOW_IMPORTANCE_P_BAND_THRESHOLD = 4.0;
    constexpr double LOW_IMPORTANCE_ENERGY_PERCENTILE_THRESHOLD = 0.001;
    constexpr double PSYCHOACOUSTIC_C_FACTOR = 2.0;

    constexpr int N_CEP_HIGH = 8;
    constexpr int BITS_PER_CEP_HIGH_COEFF = 4;
    constexpr int BITS_PVQ_GAIN_HIGH = 5;
    constexpr int BITS_PVQ_PULSES_HIGH = 5;
    constexpr int PVQ_MAX_K_HIGH = 64;

    constexpr int N_CEP_MID = 4;
    constexpr int BITS_PER_CEP_MID_COEFF = 3;

    constexpr uint32_t SYNC_WORD = 0xACDCCODE;

    enum class ChannelMode : uint8_t {
        MONO = 0,
        STEREO_MS = 1
    };

    enum class SamplingRateIndex : uint8_t {
        SR_48000 = 0,
        SR_44100 = 1,
        SR_32000 = 2,
        SR_RESERVED = 3
    };

    enum class BandImportance : uint8_t {
        LOW = 0,
        MEDIUM = 1,
        HIGH = 2,
        RESERVED = 3
    };
} // namespace CodecParams

// --- audio_types.h ---
namespace AudioTypes {
    using Sample = float;
    using Frame = std::vector<Sample>;
    using MdctCoeffs = std::vector<double>;

    struct PcmFrame {
        Frame channel_left;
        Frame channel_right;
        bool is_stereo = false;
    };

    struct MdctFrameData { // Renamed from MdctFrame to avoid conflict with a class name
        MdctCoeffs channel1_coeffs;
        MdctCoeffs channel2_coeffs;
    };

    struct BandData {
        CodecParams::BandImportance importance;
        std::vector<int> cepstrum_indices;
        int gain_index;
        int pulse_count;
        std::vector<uint8_t> pvq_indices_payload; // Serialized PVQ indices

        BandData() : importance(CodecParams::BandImportance::LOW), gain_index(0), pulse_count(0) {}
    };

    struct EncodedChannelData {
        std::vector<BandData> bands_data;
        EncodedChannelData() : bands_data(CodecParams::NUM_BANDS) {}
    };

    struct EncodedFrameData {
        bool ms_applied = false;
        EncodedChannelData ch1_data;
        EncodedChannelData ch2_data;
        CodecParams::ChannelMode channel_mode;
    };
} // namespace AudioTypes

// --- wav_io.h/cpp ---
namespace WavIO {
    struct WavHeader {
        uint32_t riff_id;        // "RIFF"
        uint32_t riff_size;      // Size of file - 8
        uint32_t wave_id;        // "WAVE"
        uint32_t fmt_id;         // "fmt "
        uint32_t fmt_size;       // Size of fmt chunk (usually 16)
        uint16_t audio_format;   // 1 for PCM
        uint16_t num_channels;
        uint32_t sample_rate;
        uint32_t byte_rate;      // sample_rate * num_channels * bits_per_sample / 8
        uint16_t block_align;    // num_channels * bits_per_sample / 8
        uint16_t bits_per_sample;
        // Possible 'data' chunk header
        uint32_t data_id;        // "data"
        uint32_t data_size;      // Size of audio data
    };

    template<typename T>
    AudioTypes::Sample pcmToSample(T pcm_val, uint16_t bits_per_sample) {
        if (bits_per_sample == 16) {
            return static_cast<AudioTypes::Sample>(pcm_val) / 32768.0f;
        } else if (bits_per_sample == 24) {
            // Assuming pcm_val is int32_t holding 24-bit data in lower 3 bytes, sign extended
            if (pcm_val & 0x00800000) { // if negative
                 return static_cast<AudioTypes::Sample>(static_cast<int32_t>(pcm_val | 0xFF000000)) / 8388608.0f;
            }
            return static_cast<AudioTypes::Sample>(pcm_val) / 8388607.0f;
        }
        // Add other types (e.g. 8-bit, 32-bit float) as needed
        return 0.0f;
    }

    template<typename T>
    T sampleToPcm(AudioTypes::Sample sample_val, uint16_t bits_per_sample) {
        if (bits_per_sample == 16) {
            AudioTypes::Sample clipped = std::max(-1.0f, std::min(1.0f, sample_val));
            return static_cast<int16_t>(clipped * 32767.0f);
        } else if (bits_per_sample == 24) {
            AudioTypes::Sample clipped = std::max(-1.0f, std::min(1.0f, sample_val));
             if (clipped >= 1.0f) return 0x007FFFFF; // Max positive 24-bit
             if (clipped <= -1.0f) return static_cast<int32_t>(0xFF800000); // Min negative 24-bit (as int32)
             return static_cast<int32_t>(clipped * 8388607.0f) & 0x00FFFFFF;
        }
        return 0;
    }


    class WavReader {
    public:
        WavReader(const std::string& filename) {
            file_ = fopen(filename.c_str(), "rb");
            if (!file_ || !readHeader()) {
                if (file_) fclose(file_);
                file_ = nullptr;
                throw std::runtime_error("WAV: Failed to open or parse WAV file: " + filename);
            }
        }
        ~WavReader() { if (file_) fclose(file_); }
        bool isOpen() const { return file_ != nullptr; }
        const WavHeader& getHeader() const { return header_; }
        uint32_t getTotalSamplesPerChannel() const {
            if (header_.bits_per_sample == 0 || header_.num_channels == 0) return 0;
            return header_.data_size / (header_.bits_per_sample / 8) / header_.num_channels;
        }

        bool readFrame(AudioTypes::PcmFrame& pcm_frame, int num_samples_per_channel) {
            if (!file_ || (samples_read_ + num_samples_per_channel > getTotalSamplesPerChannel() && getTotalSamplesPerChannel() > 0) ) {
                 if (samples_read_ >= getTotalSamplesPerChannel() && getTotalSamplesPerChannel() > 0) return false; // Already at EOF
                 num_samples_per_channel = getTotalSamplesPerChannel() - samples_read_; // Read remaining
                 if (num_samples_per_channel <= 0) return false;
            }
            if (num_samples_per_channel == 0) return false;


            pcm_frame.is_stereo = (header_.num_channels == 2);
            pcm_frame.channel_left.assign(num_samples_per_channel, 0.0f);
            if (pcm_frame.is_stereo) {
                pcm_frame.channel_right.assign(num_samples_per_channel, 0.0f);
            }

            int bytes_per_sample = header_.bits_per_sample / 8;
            std::vector<char> buffer(num_samples_per_channel * header_.num_channels * bytes_per_sample);
            size_t bytes_to_read = buffer.size();
            size_t bytes_actually_read = fread(buffer.data(), 1, bytes_to_read, file_);

            if (bytes_actually_read == 0 && feof(file_)) return false;
            if (bytes_actually_read < bytes_to_read && ferror(file_)) {
                throw std::runtime_error("WAV: File read error.");
            }
            // If fewer bytes read than expected (but not error), it means EOF. Adjust num_samples_per_channel.
            int actual_samples_in_buffer_total = bytes_actually_read / bytes_per_sample;
            int actual_samples_per_channel_read = actual_samples_in_buffer_total / header_.num_channels;

            if (actual_samples_per_channel_read < num_samples_per_channel) {
                 pcm_frame.channel_left.resize(actual_samples_per_channel_read);
                 if(pcm_frame.is_stereo) pcm_frame.channel_right.resize(actual_samples_per_channel_read);
                 num_samples_per_channel = actual_samples_per_channel_read;
                 if (num_samples_per_channel == 0) return false; // No full samples read
            }


            for (int i = 0; i < num_samples_per_channel; ++i) {
                if (header_.bits_per_sample == 16) {
                    int16_t val_l = *reinterpret_cast<int16_t*>(&buffer[(i * header_.num_channels) * bytes_per_sample]);
                    pcm_frame.channel_left[i] = pcmToSample(val_l, 16);
                    if (pcm_frame.is_stereo) {
                        int16_t val_r = *reinterpret_cast<int16_t*>(&buffer[(i * header_.num_channels + 1) * bytes_per_sample]);
                        pcm_frame.channel_right[i] = pcmToSample(val_r, 16);
                    }
                } else if (header_.bits_per_sample == 24) {
                    int32_t val_l = 0;
                    memcpy(&val_l, &buffer[(i * header_.num_channels) * bytes_per_sample], 3);
                    if (val_l & 0x00800000) val_l |= 0xFF000000; // Sign extend
                    pcm_frame.channel_left[i] = pcmToSample(val_l, 24);

                    if (pcm_frame.is_stereo) {
                        int32_t val_r = 0;
                        memcpy(&val_r, &buffer[(i * header_.num_channels + 1) * bytes_per_sample], 3);
                        if (val_r & 0x00800000) val_r |= 0xFF000000; // Sign extend
                        pcm_frame.channel_right[i] = pcmToSample(val_r, 24);
                    }
                } else {
                    throw std::runtime_error("WAV: Unsupported bit depth for reading: " + std::to_string(header_.bits_per_sample));
                }
            }
            samples_read_ += num_samples_per_channel;
            return num_samples_per_channel > 0;
        }

    private:
        FILE* file_ = nullptr;
        WavHeader header_;
        uint32_t samples_read_ = 0;

        bool readHeader() {
            // Simplified: assumes canonical WAV structure. Robust parsing is more complex.
            if (fread(&header_.riff_id, 4, 1, file_) != 1 || strncmp((char*)&header_.riff_id, "RIFF", 4) != 0) return false;
            if (fread(&header_.riff_size, 4, 1, file_) != 1) return false;
            if (fread(&header_.wave_id, 4, 1, file_) != 1 || strncmp((char*)&header_.wave_id, "WAVE", 4) != 0) return false;

            char chunk_id[5] = {0};
            uint32_t chunk_size;
            bool fmt_found = false;
            while (fread(chunk_id, 4, 1, file_) == 1) {
                if (fread(&chunk_size, 4, 1, file_) != 1) return false;
                if (strncmp(chunk_id, "fmt ", 4) == 0) {
                    header_.fmt_id = *(uint32_t*)chunk_id;
                    header_.fmt_size = chunk_size;
                    if (fread(&header_.audio_format, 2, 1, file_) != 1) return false;
                    if (fread(&header_.num_channels, 2, 1, file_) != 1) return false;
                    if (fread(&header_.sample_rate, 4, 1, file_) != 1) return false;
                    if (fread(&header_.byte_rate, 4, 1, file_) != 1) return false;
                    if (fread(&header_.block_align, 2, 1, file_) != 1) return false;
                    if (fread(&header_.bits_per_sample, 2, 1, file_) != 1) return false;
                    if (chunk_size > 16) fseek(file_, chunk_size - 16, SEEK_CUR);
                    fmt_found = true;
                } else if (strncmp(chunk_id, "data", 4) == 0) {
                    header_.data_id = *(uint32_t*)chunk_id;
                    header_.data_size = chunk_size;
                    if (!fmt_found) return false; // 'data' before 'fmt '
                    return true; // Found data chunk, header parsing mostly done
                } else {
                    if (fseek(file_, chunk_size, SEEK_CUR) != 0) return false; // Skip unknown chunk
                }
                 // Check for EOF or error after fseek/fread attempts inside loop
                if (feof(file_) || ferror(file_)) break;
            }
            return false; // Reached EOF without finding 'data' or error
        }
    };

    class WavWriter {
    public:
        WavWriter(const std::string& filename, uint16_t num_channels, uint32_t sample_rate, uint16_t bits_per_sample) {
            header_.num_channels = num_channels;
            header_.sample_rate = sample_rate;
            header_.bits_per_sample = bits_per_sample;
            header_.audio_format = 1; // PCM
            header_.block_align = (num_channels * bits_per_sample) / 8;
            header_.byte_rate = sample_rate * header_.block_align;
            
            header_.riff_id = 0x46464952; // "RIFF"
            header_.wave_id = 0x45564157; // "WAVE"
            header_.fmt_id  = 0x20746D66; // "fmt "
            header_.fmt_size = 16;
            header_.data_id = 0x61746164; // "data"
            header_.data_size = 0; 
            header_.riff_size = 36; // Base size (44 - 8) + data_size

            file_ = fopen(filename.c_str(), "wb");
            if (!file_) throw std::runtime_error("WAV: Failed to open WAV file for writing: " + filename);
            writePreliminaryHeader();
        }
        ~WavWriter() {
            if (file_) {
                finalize();
                fclose(file_);
            }
        }
        bool isOpen() const { return file_ != nullptr; }

        bool writeFrame(const AudioTypes::PcmFrame& pcm_frame) {
            if (!file_ || pcm_frame.channel_left.empty()) return false;
            
            int num_samples = pcm_frame.channel_left.size();
            int bytes_per_sample = header_.bits_per_sample / 8;
            std::vector<char> buffer(num_samples * header_.num_channels * bytes_per_sample);

            for (int i = 0; i < num_samples; ++i) {
                if (header_.bits_per_sample == 16) {
                    int16_t val_l = sampleToPcm<int16_t>(pcm_frame.channel_left[i], 16);
                    memcpy(&buffer[(i * header_.num_channels) * bytes_per_sample], &val_l, bytes_per_sample);
                    if (header_.num_channels == 2) {
                        int16_t val_r = pcm_frame.channel_right.empty() ? 0 : sampleToPcm<int16_t>(pcm_frame.channel_right[i], 16);
                        memcpy(&buffer[(i * header_.num_channels + 1) * bytes_per_sample], &val_r, bytes_per_sample);
                    }
                } else if (header_.bits_per_sample == 24) {
                    int32_t val_l_32 = sampleToPcm<int32_t>(pcm_frame.channel_left[i], 24);
                    memcpy(&buffer[(i * header_.num_channels) * bytes_per_sample], &val_l_32, 3); // Write 3 bytes
                     if (header_.num_channels == 2) {
                        int32_t val_r_32 = pcm_frame.channel_right.empty() ? 0 : sampleToPcm<int32_t>(pcm_frame.channel_right[i], 24);
                        memcpy(&buffer[(i * header_.num_channels + 1) * bytes_per_sample], &val_r_32, 3);
                    }
                } else {
                     throw std::runtime_error("WAV: Unsupported bit depth for writing: " + std::to_string(header_.bits_per_sample));
                }
            }
            size_t items_written = fwrite(buffer.data(), 1, buffer.size(), file_);
            if (items_written != buffer.size()) return false;
            data_written_bytes_ += items_written;
            return true;
        }

        void finalize() {
            if (!file_ || finalized_) return;
            
            header_.data_size = data_written_bytes_;
            header_.riff_size = 36 + data_written_bytes_;
            
            fseek(file_, 4, SEEK_SET); 
            fwrite(&header_.riff_size, 4, 1, file_);
            
            fseek(file_, 40, SEEK_SET); 
            fwrite(&header_.data_size, 4, 1, file_);
            
            fflush(file_);
            finalized_ = true;
        }

    private:
        FILE* file_ = nullptr;
        WavHeader header_;
        uint32_t data_written_bytes_ = 0;
        bool finalized_ = false;

        void writePreliminaryHeader() {
            fwrite(&header_.riff_id, 4, 1, file_);
            fwrite(&header_.riff_size, 4, 1, file_); // Placeholder
            fwrite(&header_.wave_id, 4, 1, file_);
            fwrite(&header_.fmt_id, 4, 1, file_);
            fwrite(&header_.fmt_size, 4, 1, file_);
            fwrite(&header_.audio_format, 2, 1, file_);
            fwrite(&header_.num_channels, 2, 1, file_);
            fwrite(&header_.sample_rate, 4, 1, file_);
            fwrite(&header_.byte_rate, 4, 1, file_);
            fwrite(&header_.block_align, 2, 1, file_);
            fwrite(&header_.bits_per_sample, 2, 1, file_);
            fwrite(&header_.data_id, 4, 1, file_);
            fwrite(&header_.data_size, 4, 1, file_); // Placeholder
        }
    };
} // namespace WavIO

// --- bitstream.h/cpp ---
namespace Bitstream {
    class BitstreamWriter {
    public:
        BitstreamWriter() = default;
        void writeBit(bool bit) {
            current_byte_ |= (bit << (7 - bit_count_));
            bit_count_++;
            if (bit_count_ == 8) {
                buffer_.push_back(current_byte_);
                current_byte_ = 0;
                bit_count_ = 0;
            }
        }
        void writeBits(uint64_t value, int num_bits) {
            if (num_bits < 0 || num_bits > 64) throw std::out_of_range("BitstreamWriter: num_bits out of range.");
            for (int i = num_bits - 1; i >= 0; --i) {
                writeBit((value >> i) & 1);
            }
        }
        void writeBytes(const uint8_t* data, size_t num_bytes) {
            flushByte(); 
            for (size_t i = 0; i < num_bytes; ++i) {
                buffer_.push_back(data[i]);
            }
        }
        void flushByte() {
            if (bit_count_ > 0) {
                buffer_.push_back(current_byte_);
                current_byte_ = 0;
                bit_count_ = 0;
            }
        }
        const std::vector<uint8_t>& getBuffer() const { return buffer_; }
        void clear() { buffer_.clear(); current_byte_ = 0; bit_count_ = 0; }
        size_t getByteCount() const { return buffer_.size() + (bit_count_ > 0 ? 1 : 0) ;}

    private:
        std::vector<uint8_t> buffer_;
        uint8_t current_byte_ = 0;
        int bit_count_ = 0;
    };

    class BitstreamReader {
    public:
        BitstreamReader(const std::vector<uint8_t>& buffer) : data_(buffer.data()), size_(buffer.size()) {}
        BitstreamReader(const uint8_t* data, size_t size) : data_(data), size_(size) {}

        bool readBit(bool& bit) {
            if (byte_pos_ >= size_ && (byte_pos_ == size_ -1 && bit_pos_ == 8 )) return false; // EOF
            if (byte_pos_ == size_) return false; // trying to read past buffer end

            bit = (data_[byte_pos_] >> (7 - bit_pos_)) & 1;
            bit_pos_++;
            if (bit_pos_ == 8) {
                byte_pos_++;
                bit_pos_ = 0;
            }
            return true;
        }
        bool readBits(uint64_t& value, int num_bits) {
            if (num_bits < 0 || num_bits > 64) throw std::out_of_range("BitstreamReader: num_bits out of range.");
            if (getRemainingBits() < static_cast<size_t>(num_bits)) return false;
            value = 0;
            for (int i = 0; i < num_bits; ++i) {
                bool b_val;
                if (!readBit(b_val)) return false;
                value = (value << 1) | b_val;
            }
            return true;
        }
        bool readBytes(uint8_t* data, size_t num_bytes) {
            if (bit_pos_ != 0) return false; // Must be byte-aligned
            if (byte_pos_ + num_bytes > size_) return false;
            memcpy(data, data_ + byte_pos_, num_bytes);
            byte_pos_ += num_bytes;
            return true;
        }
        bool isEof() const { return byte_pos_ >= size_; }
        size_t getRemainingBits() const {
            if (byte_pos_ >= size_) return 0;
            return (size_ - byte_pos_ -1) * 8 + (8 - bit_pos_);
        }
         void alignToByte() {
            if (bit_pos_ != 0) {
                byte_pos_++;
                bit_pos_ = 0;
            }
        }

    private:
        const uint8_t* data_ = nullptr;
        size_t size_ = 0;
        size_t byte_pos_ = 0;
        int bit_pos_ = 0;
    };
} // namespace Bitstream

// --- ppmd.h/cpp (Stub) ---
namespace Ppmd {
    class PPMdEncoder {
    public:
        PPMdEncoder(int /*max_order*/, size_t /*memory_limit*/) {}
        void encodeData(const std::vector<uint8_t>& data_in, std::vector<uint8_t>& data_out) {
            data_out = data_in; // STUB: No compression
        }
        std::vector<uint8_t> finalize() { return {}; } // STUB
    };

    class PPMdDecoder {
    public:
        PPMdDecoder(int /*max_order*/, size_t /*memory_limit*/) {}
        void decodeData(const std::vector<uint8_t>& data_in, std::vector<uint8_t>& data_out, size_t /*expected_decoded_size*/) {
            data_out = data_in; // STUB: No decompression
        }
        void setInput(const std::vector<uint8_t>& /*data_in*/) {} // STUB
    };
} // namespace Ppmd

// --- math_utils.h/cpp ---
namespace MathUtils {
    std::vector<double> dct_ii(const std::vector<double>& input) {
        int N = input.size();
        if (N == 0) return {};
        std::vector<double> output(N);
        double C_k_norm = std::sqrt(2.0 / N);
        double C_0_norm = std::sqrt(1.0 / N);

        for (int k = 0; k < N; ++k) {
            double sum = 0.0;
            for (int n = 0; n < N; ++n) {
                sum += input[n] * std::cos(M_PI * k * (2.0 * n + 1.0) / (2.0 * N));
            }
            output[k] = sum * (k == 0 ? C_0_norm : C_k_norm);
        }
        return output;
    }

    std::vector<double> idct_ii(const std::vector<double>& input_dct_coeffs) {
        // For an orthonormal DCT-II, its inverse is also DCT-II (or scaled DCT-III)
        // Here we apply DCT-II again, assuming the DCT-II above is orthonormal
        int N = input_dct_coeffs.size();
        if (N == 0) return {};
        std::vector<double> output(N);
        for (int n = 0; n < N; ++n) {
            double sum = 0.0;
            for (int k = 0; k < N; ++k) {
                double factor = (k == 0) ? std::sqrt(1.0 / N) : std::sqrt(2.0 / N);
                sum += factor * input_dct_coeffs[k] * std::cos(M_PI * k * (2.0 * n + 1.0) / (2.0 * N));
            }
            output[n] = sum;
        }
        return output;
    }

    std::vector<double> calcSineWindow(int size) {
        std::vector<double> window(size);
        for (int i = 0; i < size; ++i) {
            window[i] = std::sin(M_PI * (static_cast<double>(i) + 0.5) / static_cast<double>(size));
        }
        return window;
    }
    
    // Placeholder Lloyd-Max Quantizer (uses uniform quantization)
    int quantizeScalar(double value, int bits, double min_val, double max_val) {
        if (bits <= 0) return 0;
        int num_levels = 1 << bits;
        double step = (max_val - min_val) / num_levels;
        if (value <= min_val) return 0;
        if (value >= max_val) return num_levels - 1;
        int index = static_cast<int>(std::floor((value - min_val) / step)); // floor for consistency
        return std::max(0, std::min(num_levels - 1, index));
    }

    double dequantizeScalar(int index, int bits, double min_val, double max_val) {
        if (bits <= 0) return (min_val + max_val) / 2.0;
        int num_levels = 1 << bits;
        double step = (max_val - min_val) / num_levels;
        return min_val + (static_cast<double>(index) + 0.5) * step;
    }
    
    // These would be pre-calculated tables for actual Lloyd-Max
    // For Cepstrum (High Importance, 8 coeffs, 4 bits each)
    const std::vector<double> K_CEP_HIGH_MIN_VALS(CodecParams::N_CEP_HIGH, -5.0); // Example
    const std::vector<double> K_CEP_HIGH_MAX_VALS(CodecParams::N_CEP_HIGH, 5.0);  // Example
    // For Cepstrum (Mid Importance, 4 coeffs, 3 bits each)
    const std::vector<double> K_CEP_MID_MIN_VALS(CodecParams::N_CEP_MID, -5.0);   // Example
    const std::vector<double> K_CEP_MID_MAX_VALS(CodecParams::N_CEP_MID, 5.0);    // Example

    // PVQ Gain Quantizer (Logarithmic)
    const double PVQ_MIN_LOG_GAIN = -5.0; // log10(0.00001)
    const double PVQ_MAX_LOG_GAIN = 5.0;  // log10(100000)

    int quantizeGainLog(double gain, int bits) {
        if (gain <= 1e-9) gain = 1e-9; // Avoid log(0) or log(negative)
        double log_gain = std::log10(gain);
        return quantizeScalar(log_gain, bits, PVQ_MIN_LOG_GAIN, PVQ_MAX_LOG_GAIN);
    }

    double dequantizeGainLog(int index, int bits) {
        double log_gain = dequantizeScalar(index, bits, PVQ_MIN_LOG_GAIN, PVQ_MAX_LOG_GAIN);
        return std::pow(10.0, log_gain);
    }
} // namespace MathUtils

// --- mdct_handler.h/cpp ---
namespace MdctHandler {
    class MDCT {
    public:
        MDCT(int N) : N_(N), N_half_(N / 2) {
            if (N_ <= 0 || (N_ % 2 != 0)) {
                throw std::invalid_argument("MDCT size N must be positive and even.");
            }
            // Precompute twiddle factors for MDCT
            // X_k = sum_{n=0}^{N-1} x_n * cos( (pi/(2N)) * (2n+1+N/2) * (2k+1) )
            // This is one form. Another common form:
            // X_k = sum_{n=0}^{N-1} x_n * cos( (pi/N) * (n + 0.5 + N/4) * (k + 0.5) ) * 2
            // Or using DCT-IV: MDCT_coeff[k] = DCT_IV(windowed_signal)[k]
            // For simplicity, using a direct (but not optimized) implementation.
            // The formula used here is:
            // Y_k = sum_{n=0}^{N-1} x_n * cos( (PI/N) * (n + 0.5 + N/(2*N)) * (k+0.5) ) for k=0..N/2-1
            // No, a more standard one is:
            // X_k = sum_{n=0}^{N-1} x[n] * cos( (PI/N) * (n + 1/2 + N/2) * (k + 1/2) ) for k = 0...N/2-1
            // For this implementation, let's assume the use of a type-IV DCT relationship if possible or a direct sum.
            // The one from spec: "MDCT (Modified Discrete Cosine Transform): 時間領域エイリアシングキャンセル特性を持つ直交変換。"
            // A common MDCT formula:
            // F[k] = sum_{n=0}^{N-1} f[n] * cos( (pi/N) * (n + 0.5 + N/2.0) * (k + 0.5) ), k=0..N/2-1
            // The IMDCT formula:
            // f[n] = (2/N) * sum_{k=0}^{N/2-1} F[k] * cos( (pi/N) * (n + 0.5 + N/2.0) * (k + 0.5) )
            // Note: The IMDCT sum is identical to MDCT sum, just with a scaling factor. So one function can do both.
            precompute_cos_table_();
        }

        void transform(const std::vector<double>& input, std::vector<double>& output) {
            if (input.size() != static_cast<size_t>(N_)) {
                throw std::invalid_argument("MDCT input size must be N.");
            }
            output.assign(N_half_, 0.0);

            for (int k = 0; k < N_half_; ++k) {
                double sum = 0.0;
                for (int n = 0; n < N_; ++n) {
                    //sum += input[n] * std::cos((M_PI / N_) * (n + 0.5 + N_half_ / 2.0) * (k + 0.5)); // Error in N_half_/2.0. Should be N_/2.0
                    // sum += input[n] * std::cos((M_PI / N_) * (n + 0.5 + N_ / 2.0) * (k + 0.5)); // This seems correct for common definition
                    // sum += input[n] * cos_table_[n * N_half_ + k]; // If precomputed this way
                    // Corrected from common definition:
                    // X_k = sum_{n=0}^{N-1} x_n cos[ (pi/N) * (n + 0.5 + N/2) * (k + 0.5) ]
                    // For IMDCT: x_n = (2/N) * sum_{k=0}^{N/2-1} X_k cos[ (pi/N) * (n + 0.5 + N/2) * (k + 0.5) ]
                    // The precomputed table helps:
                    sum += input[n] * cos_table_mdct_[(n * N_half_) + k];
                }
                output[k] = sum; // For MDCT, some definitions include a factor of 2 or sqrt(2/N)
                                 // For perfect reconstruction with the IMDCT below, this is fine.
            }
        }

        void inverse_transform(const std::vector<double>& input_coeffs, std::vector<double>& output_signal) {
            if (input_coeffs.size() != static_cast<size_t>(N_half_)) {
                throw std::invalid_argument("IMDCT input size must be N/2.");
            }
            output_signal.assign(N_, 0.0);
            double scale = 2.0 / N_;

            for (int n = 0; n < N_; ++n) {
                double sum = 0.0;
                for (int k = 0; k < N_half_; ++k) {
                    // sum += input_coeffs[k] * std::cos((M_PI / N_) * (n + 0.5 + N_ / 2.0) * (k + 0.5));
                    sum += input_coeffs[k] * cos_table_mdct_[(n * N_half_) + k]; // Same table
                }
                output_signal[n] = sum * scale;
            }
        }

    private:
        int N_;
        int N_half_;
        std::vector<double> cos_table_mdct_; // For F[k] = sum x[n] cos_term[n,k]

        void precompute_cos_table_() {
            cos_table_mdct_.resize(N_ * N_half_);
            for (int n = 0; n < N_; ++n) {
                for (int k = 0; k < N_half_; ++k) {
                    cos_table_mdct_[(n * N_half_) + k] = std::cos((M_PI / N_) * (n + 0.5 + N_ / 2.0) * (k + 0.5));
                }
            }
        }
    };

    std::vector<AudioTypes::Sample> sine_window_cache; // Global for simplicity
    const std::vector<AudioTypes::Sample>& getSineWindow(int size) {
        if (sine_window_cache.size() != static_cast<size_t>(size)) {
            sine_window_cache.resize(size);
            for (int i = 0; i < size; ++i) {
                sine_window_cache[i] = static_cast<AudioTypes::Sample>(std::sin(M_PI * (static_cast<double>(i) + 0.5) / static_cast<double>(size)));
            }
        }
        return sine_window_cache;
    }

    void applyWindow(std::vector<AudioTypes::Sample>& block) {
        const auto& window = getSineWindow(block.size());
        for (size_t i = 0; i < block.size(); ++i) {
            block[i] *= window[i];
        }
    }
     void applyWindow(std::vector<double>& block) { // Overload for double
        const auto& window_float = getSineWindow(block.size()); // get float window
        std::vector<double> window_double(window_float.begin(), window_float.end()); // convert
        for (size_t i = 0; i < block.size(); ++i) {
            block[i] *= window_double[i];
        }
    }


} // namespace MdctHandler

// --- ms_processor.h/cpp ---
namespace MsProcessor {
    void applyMSProcessing(const AudioTypes::Frame& L, const AudioTypes::Frame& R,
                           AudioTypes::Frame& M, AudioTypes::Frame& S, bool& ms_applied) {
        size_t frame_size = L.size();
        M.resize(frame_size);
        S.resize(frame_size);

        double s_energy = 0.0;
        double m_energy = 0.0;

        for (size_t i = 0; i < frame_size; ++i) {
            AudioTypes::Sample mid_val = (L[i] + R[i]) / 2.0f;
            AudioTypes::Sample side_val = (L[i] - R[i]) / 2.0f;
            M[i] = mid_val; // Store temporarily for energy calculation
            S[i] = side_val;
            m_energy += mid_val * mid_val;
            s_energy += side_val * side_val;
        }

        if (m_energy > 1e-9 && (s_energy / m_energy) < CodecParams::MS_ENERGY_RATIO_THRESHOLD) {
            ms_applied = true;
            // M and S are already calculated
        } else {
            ms_applied = false;
            M = L; // Output L as M (channel 1)
            S = R; // Output R as S (channel 2)
        }
    }

    void inverseMSProcessing(AudioTypes::Frame& M_or_L, AudioTypes::Frame& S_or_R, bool ms_applied) {
        if (!ms_applied) return; // L/R are already in M_or_L, S_or_R

        size_t frame_size = M_or_L.size();
        AudioTypes::Frame L_temp(frame_size);
        AudioTypes::Frame R_temp(frame_size);

        for (size_t i = 0; i < frame_size; ++i) {
            L_temp[i] = M_or_L[i] + S_or_R[i];
            R_temp[i] = M_or_L[i] - S_or_R[i];
        }
        M_or_L = L_temp;
        S_or_R = R_temp;
    }
} // namespace MsProcessor

// --- psychoacoustic.h/cpp ---
namespace Psychoacoustic {
    // Simplified ATH table (dB SPL for 48kHz)
    // Freq (Hz) | ATH (dB SPL)
    // Needs to be mapped to MDCT bin indices for each band.
    // This is a very coarse approximation. A real ATH is a continuous curve.
    std::map<int, double> ath_dB_table_48kHz = {
        { 63, 26.0}, { 125, 15.0}, { 250, 7.0}, { 500, 3.0},
        {1000, 0.0}, {2000, -2.0}, {4000, -7.0}, {8000, 5.0},
        {12000, 10.0},{16000, 20.0}
    };

    double getATHValueForBand(int band_index, int mdct_size, int sample_rate) {
        // Determine center frequency of the band
        int start_coeff = CodecParams::BAND_OFFSETS[band_index];
        int end_coeff = CodecParams::BAND_OFFSETS[band_index + 1] - 1;
        double center_coeff = (start_coeff + end_coeff) / 2.0;
        double center_freq = center_coeff * (static_cast<double>(sample_rate) / mdct_size);

        // Find closest ATH entry (simple linear interpolation or nearest for this stub)
        auto it_upper = ath_dB_table_48kHz.lower_bound(static_cast<int>(center_freq));
        if (it_upper == ath_dB_table_48kHz.begin()) return it_upper->second;
        if (it_upper == ath_dB_table_48kHz.end()) {
            it_upper--;
            return it_upper->second;
        }
        auto it_lower = it_upper;
        --it_lower;

        // Linear interpolation in dB domain
        double freq_low = it_lower->first;
        double ath_low = it_lower->second;
        double freq_high = it_upper->first;
        double ath_high = it_upper->second;

        if (freq_high == freq_low) return ath_low; // Avoid division by zero
        double ath_db = ath_low + (ath_high - ath_low) * (center_freq - freq_low) / (freq_high - freq_low);
        
        // Convert dB SPL to power (relative, assuming 0dB SPL is some reference power unit)
        // Power is proportional to 10^(dB/10)
        return std::pow(10.0, ath_db / 10.0);
    }

    // Placeholder ATH values per band for 48kHz, N=2048 (MDCT coeffs 1024)
    // These should be precomputed based on a proper ATH curve.
    std::vector<double> precomputed_ath_power_per_band;

    void initializePsychoacousticModel(int mdct_size, int sample_rate) {
        if (!precomputed_ath_power_per_band.empty() && precomputed_ath_power_per_band.size() == CodecParams::NUM_BANDS) return;

        precomputed_ath_power_per_band.resize(CodecParams::NUM_BANDS);
        for (int i = 0; i < CodecParams::NUM_BANDS; ++i) {
            precomputed_ath_power_per_band[i] = getATHValueForBand(i, mdct_size, sample_rate);
             // Apply a floor, e.g. if ATH is very low, can cause issues
            if (precomputed_ath_power_per_band[i] < 1e-6) precomputed_ath_power_per_band[i] = 1e-6;
        }
    }


    CodecParams::BandImportance evaluateBandImportance(
        const AudioTypes::MdctCoeffs& band_coeffs,
        int band_idx,
        double total_frame_energy,
        const std::vector<double>& all_band_energies) {
        
        if (precomputed_ath_power_per_band.empty()) {
             // Should be initialized by encoder/decoder setup
            initializePsychoacousticModel(CodecParams::MDCT_FRAME_SIZE_N, CodecParams::SAMPLING_FREQUENCY);
        }

        double e_band = 0.0;
        for (double coeff : band_coeffs) {
            e_band += coeff * coeff;
        }

        double ath_band = precomputed_ath_power_per_band[band_idx];
        double mask_thresh = ath_band * CodecParams::PSYCHOACOUSTIC_C_FACTOR; // Simplified: no spreading
        
        double p_band = (mask_thresh > 1e-9) ? (e_band / mask_thresh) : (e_band / 1e-9); // Avoid division by zero

        // Determine energy rank (simplified)
        // For a more accurate percentile, sort all_band_energies
        // This simplified check: is e_band among the highest?
        // A proper percentile requires sorting all band energies for the frame.
        bool is_high_energy_band = false;
        if (total_frame_energy > 1e-9) {
            std::vector<double> sorted_energies = all_band_energies;
            std::sort(sorted_energies.rbegin(), sorted_energies.rend()); // Sort descending
            int top_25_percent_count = std::max(1, static_cast<int>(sorted_energies.size() * CodecParams::HIGH_IMPORTANCE_ENERGY_PERCENTILE_THRESHOLD));
            if (e_band >= sorted_energies[std::min(static_cast<int>(sorted_energies.size())-1, top_25_percent_count-1)]) {
                 is_high_energy_band = true;
            }
        }


        if (p_band > CodecParams::HIGH_IMPORTANCE_P_BAND_THRESHOLD && is_high_energy_band) {
            return CodecParams::BandImportance::HIGH;
        }
        if (p_band < CodecParams::LOW_IMPORTANCE_P_BAND_THRESHOLD || 
            (total_frame_energy > 1e-9 && (e_band / total_frame_energy) < CodecParams::LOW_IMPORTANCE_ENERGY_PERCENTILE_THRESHOLD)) {
            return CodecParams::BandImportance::LOW;
        }
        return CodecParams::BandImportance::MEDIUM;
    }
} // namespace Psychoacoustic

// --- cepstrum.h/cpp ---
namespace Cepstrum {
    AudioTypes::MdctCoeffs calculateCepstrum(const AudioTypes::MdctCoeffs& band_mdct_coeffs, int num_cep_coeffs) {
        if (band_mdct_coeffs.empty() || num_cep_coeffs <= 0) return {};
        
        std::vector<double> power_spectrum(band_mdct_coeffs.size());
        for (size_t i = 0; i < band_mdct_coeffs.size(); ++i) {
            power_spectrum[i] = band_mdct_coeffs[i] * band_mdct_coeffs[i];
        }

        std::vector<double> log_power_spectrum(power_spectrum.size());
        for (size_t i = 0; i < power_spectrum.size(); ++i) {
            log_power_spectrum[i] = (power_spectrum[i] > 1e-12) ? std::log(power_spectrum[i]) : std::log(1e-12); // Avoid log(0)
        }

        std::vector<double> cepstrum_full = MathUtils::dct_ii(log_power_spectrum);
        
        AudioTypes::MdctCoeffs result_cepstrum(num_cep_coeffs);
        for (int i = 0; i < num_cep_coeffs && i < static_cast<int>(cepstrum_full.size()); ++i) {
            result_cepstrum[i] = cepstrum_full[i];
        }
        return result_cepstrum;
    }

    AudioTypes::MdctCoeffs reconstructEnvelopeFromCepstrum(const AudioTypes::MdctCoeffs& cepstrum_coeffs, int original_spectrum_size) {
        if (cepstrum_coeffs.empty() || original_spectrum_size <= 0) return {};

        std::vector<double> padded_cepstrum = cepstrum_coeffs;
        if (static_cast<int>(padded_cepstrum.size()) < original_spectrum_size) {
            padded_cepstrum.resize(original_spectrum_size, 0.0); // Pad with zeros for IDCT
        } else if (static_cast<int>(padded_cepstrum.size()) > original_spectrum_size) {
            padded_cepstrum.resize(original_spectrum_size); // Truncate if cepstrum has more coeffs than needed (should not happen)
        }
        
        std::vector<double> log_power_spectrum_reconst = MathUtils::idct_ii(padded_cepstrum);
        
        AudioTypes::MdctCoeffs envelope_amplitude_spectrum(original_spectrum_size);
        for (int i = 0; i < original_spectrum_size; ++i) {
            // log_power_spectrum gives log(|X|^2) = 2 * log(|X|)
            // So, |X| = exp( (1/2) * log_power_spectrum )
            // However, if input to DCT for cepstrum was log(|MDCTcoeff|^2), then IDCT gives back log(|MDCTcoeff|^2).
            // Then envelope magnitude is sqrt(exp(log_power_spectrum_reconst))
            envelope_amplitude_spectrum[i] = std::sqrt(std::exp(log_power_spectrum_reconst[i]));
             if (std::isnan(envelope_amplitude_spectrum[i]) || std::isinf(envelope_amplitude_spectrum[i])) {
                envelope_amplitude_spectrum[i] = 0.0; // Handle potential numerical issues
            }
        }
        return envelope_amplitude_spectrum;
    }
} // namespace Cepstrum

// --- pvq.h/cpp (Stub) ---
namespace Pvq {
    // STUB PVQ implementation
    // A real PVQ implementation is mathematically complex involving combinatorics (counting pulses on a pyramid).

    // Encodes a normalized vector into a PVQ index and number of pulses.
    // For stub, it just serializes the vector itself after some dummy operations.
    std::vector<uint8_t> pvqEncode(const AudioTypes::MdctCoeffs& normalized_residual, int& K_dim, int& L_pulses) {
        K_dim = std::min(static_cast<int>(normalized_residual.size()), CodecParams::PVQ_MAX_K_HIGH);
        // L_pulses would be adaptively chosen based on target bits/perceptual importance.
        // For stub, pick a fixed small number.
        L_pulses = std::min(K_dim, 8); // Example: max 8 pulses or K_dim if smaller

        // STUB: Serialize first K_dim elements, scaled to uint8_t for simplicity.
        // A real PVQ index is a single integer.
        std::vector<uint8_t> payload;
        if (K_dim == 0) return payload;

        payload.reserve(K_dim);
        for (int i = 0; i < K_dim; ++i) {
            // Dummy "quantization" of residual value to a byte
            double val = normalized_residual[i];
            // Scale to 0-255 range (assuming normalized_residual elements are roughly in [-1, 1] or similar)
            // This is NOT PVQ, just a placeholder.
            int8_t quantized_val_signed = static_cast<int8_t>(std::max(-127.0, std::min(127.0, val * 127.0)));
            payload.push_back(static_cast<uint8_t>(quantized_val_signed + 128)); // map to 0-255
        }
        return payload;
    }

    // Decodes PVQ index and pulses into a normalized vector.
    // For stub, it deserializes the vector passed by pvqEncode.
    AudioTypes::MdctCoeffs pvqDecode(const std::vector<uint8_t>& pvq_payload, int K_dim, int /* L_pulses */) {
        AudioTypes::MdctCoeffs normalized_residual(K_dim);
        if (K_dim == 0) return normalized_residual;

        if (pvq_payload.size() < static_cast<size_t>(K_dim)) {
             // Error or fill with zeros
             std::fill(normalized_residual.begin(), normalized_residual.end(), 0.0);
             return normalized_residual;
        }

        for (int i = 0; i < K_dim; ++i) {
            // Reverse the dummy "quantization"
            int8_t quantized_val_signed = static_cast<int8_t>(pvq_payload[i] - 128);
            normalized_residual[i] = static_cast<double>(quantized_val_signed) / 127.0;
        }
        // A real PVQ decode would then normalize this vector to unit L2 norm if L_pulses > 0
        // or return a zero vector if L_pulses == 0.
        // For the stub, we assume it's roughly normalized already.
        return normalized_residual;
    }
} // namespace Pvq

// --- quantizer.h/cpp (Mainly interfaces for Lloyd-Max and PVQ) ---
namespace Quantizer {
    // Lloyd-Max quantizers for cepstrum (using MathUtils placeholders)
    std::vector<int> quantizeCepstrumHigh(const AudioTypes::MdctCoeffs& cep_coeffs) {
        std::vector<int> indices(CodecParams::N_CEP_HIGH);
        for (int i = 0; i < CodecParams::N_CEP_HIGH; ++i) {
            indices[i] = MathUtils::quantizeScalar(cep_coeffs[i], CodecParams::BITS_PER_CEP_HIGH_COEFF,
                                                   MathUtils::K_CEP_HIGH_MIN_VALS[i], MathUtils::K_CEP_HIGH_MAX_VALS[i]);
        }
        return indices;
    }
    AudioTypes::MdctCoeffs dequantizeCepstrumHigh(const std::vector<int>& indices) {
        AudioTypes::MdctCoeffs cep_coeffs(CodecParams::N_CEP_HIGH);
        for (int i = 0; i < CodecParams::N_CEP_HIGH; ++i) {
            cep_coeffs[i] = MathUtils::dequantizeScalar(indices[i], CodecParams::BITS_PER_CEP_HIGH_COEFF,
                                                        MathUtils::K_CEP_HIGH_MIN_VALS[i], MathUtils::K_CEP_HIGH_MAX_VALS[i]);
        }
        return cep_coeffs;
    }

    std::vector<int> quantizeCepstrumMid(const AudioTypes::MdctCoeffs& cep_coeffs) {
        std::vector<int> indices(CodecParams::N_CEP_MID);
        for (int i = 0; i < CodecParams::N_CEP_MID; ++i) {
            indices[i] = MathUtils::quantizeScalar(cep_coeffs[i], CodecParams::BITS_PER_CEP_MID_COEFF,
                                                   MathUtils::K_CEP_MID_MIN_VALS[i], MathUtils::K_CEP_MID_MAX_VALS[i]);
        }
        return indices;
    }
    AudioTypes::MdctCoeffs dequantizeCepstrumMid(const std::vector<int>& indices) {
        AudioTypes::MdctCoeffs cep_coeffs(CodecParams::N_CEP_MID);
        for (int i = 0; i < CodecParams::N_CEP_MID; ++i) {
            cep_coeffs[i] = MathUtils::dequantizeScalar(indices[i], CodecParams::BITS_PER_CEP_MID_COEFF,
                                                        MathUtils::K_CEP_MID_MIN_VALS[i], MathUtils::K_CEP_MID_MAX_VALS[i]);
        }
        return cep_coeffs;
    }

    // PVQ Gain Quantization (using MathUtils placeholders)
    int quantizePVQGain(double gain) {
        return MathUtils::quantizeGainLog(gain, CodecParams::BITS_PVQ_GAIN_HIGH);
    }
    double dequantizePVQGain(int index) {
        return MathUtils::dequantizeGainLog(index, CodecParams::BITS_PVQ_GAIN_HIGH);
    }
    
    // PVQ Pulse Count Quantization (placeholder - direct value, assumes value fits in bits)
    int quantizePVQPulses(int pulses) {
        // Assumes pulses is already within valid range for BITS_PVQ_PULSES_HIGH
        // No real quantization here, just ensuring it fits.
        int max_pulses = (1 << CodecParams::BITS_PVQ_PULSES_HIGH) -1;
        return std::min(pulses, max_pulses);
    }
    int dequantizePVQPulses(int index) {
        return index; // Direct value
    }

} // namespace Quantizer


// --- encoder.h/cpp ---
class Encoder {
public:
    Encoder() : mdct_ (CodecParams::MDCT_FRAME_SIZE_N) {
        Psychoacoustic::initializePsychoacousticModel(CodecParams::MDCT_FRAME_SIZE_N, CodecParams::SAMPLING_FREQUENCY);
        previous_frame_half_ch1_.assign(CodecParams::NEW_SAMPLES_PER_FRAME, 0.0f);
        previous_frame_half_ch2_.assign(CodecParams::NEW_SAMPLES_PER_FRAME, 0.0f);
    }

    AudioTypes::EncodedFrameData encodeFrame(const AudioTypes::PcmFrame& pcm_frame) {
        AudioTypes::EncodedFrameData encoded_data;
        encoded_data.channel_mode = pcm_frame.is_stereo ? CodecParams::ChannelMode::STEREO_MS : CodecParams::ChannelMode::MONO;

        AudioTypes::Frame current_ch1_samples = pcm_frame.channel_left;
        AudioTypes::Frame current_ch2_samples;
        if (pcm_frame.is_stereo) {
            current_ch2_samples = pcm_frame.channel_right;
        }

        // 5.1 MS Processing (if stereo)
        AudioTypes::Frame processed_ch1 = current_ch1_samples; // Will be L or M
        AudioTypes::Frame processed_ch2;                       // Will be R or S (if stereo)

        if (encoded_data.channel_mode == CodecParams::ChannelMode::STEREO_MS) {
            if (current_ch1_samples.size() != current_ch2_samples.size()) {
                 throw std::runtime_error("Encoder: Stereo channels have different sizes for MS processing.");
            }
            MsProcessor::applyMSProcessing(current_ch1_samples, current_ch2_samples,
                                           processed_ch1, processed_ch2, encoded_data.ms_applied);
        }

        // 5.2 MDCT
        AudioTypes::MdctFrameData mdct_coeffs_frame;
        performMdctForChannel(processed_ch1, previous_frame_half_ch1_, mdct_coeffs_frame.channel1_coeffs);
        if (encoded_data.channel_mode == CodecParams::ChannelMode::STEREO_MS) {
            performMdctForChannel(processed_ch2, previous_frame_half_ch2_, mdct_coeffs_frame.channel2_coeffs);
        }
        
        // Store current half for next frame's overlap-add
        std::copy(current_ch1_samples.end() - CodecParams::NEW_SAMPLES_PER_FRAME, current_ch1_samples.end(), previous_frame_half_ch1_.begin());
        if (pcm_frame.is_stereo) {
            std::copy(current_ch2_samples.end() - CodecParams::NEW_SAMPLES_PER_FRAME, current_ch2_samples.end(), previous_frame_half_ch2_.begin());
        }


        // Process Channel 1 (L or Mid)
        processChannelMdct(mdct_coeffs_frame.channel1_coeffs, encoded_data.ch1_data);

        // Process Channel 2 (R or Side), if stereo
        if (encoded_data.channel_mode == CodecParams::ChannelMode::STEREO_MS) {
            processChannelMdct(mdct_coeffs_frame.channel2_coeffs, encoded_data.ch2_data);
        }
        
        return encoded_data;
    }

private:
    MdctHandler::MDCT mdct_;
    AudioTypes::Frame previous_frame_half_ch1_; // Stores N/2 samples from previous frame for overlap
    AudioTypes::Frame previous_frame_half_ch2_;

    void performMdctForChannel(const AudioTypes::Frame& current_samples, // N/2 new samples
                               const AudioTypes::Frame& previous_half_samples, // N/2 old samples
                               AudioTypes::MdctCoeffs& out_coeffs) {
        if (current_samples.size() != CodecParams::NEW_SAMPLES_PER_FRAME ||
            previous_half_samples.size() != CodecParams::NEW_SAMPLES_PER_FRAME) {
            throw std::runtime_error("Encoder: Incorrect sample counts for MDCT input block construction.");
        }

        std::vector<double> mdct_input_block(CodecParams::MDCT_FRAME_SIZE_N);
        // Create 2048 sample block for MDCT: [prev_half | current_new]
        for(int i=0; i < CodecParams::NEW_SAMPLES_PER_FRAME; ++i) {
            mdct_input_block[i] = static_cast<double>(previous_half_samples[i]);
            mdct_input_block[i + CodecParams::NEW_SAMPLES_PER_FRAME] = static_cast<double>(current_samples[i]);
        }
        
        MdctHandler::applyWindow(mdct_input_block); // Apply Sine window
        mdct_.transform(mdct_input_block, out_coeffs);
    }

    void processChannelMdct(const AudioTypes::MdctCoeffs& channel_mdct_coeffs, AudioTypes::EncodedChannelData& channel_data) {
        double total_channel_energy = 0;
        std::vector<double> band_energies(CodecParams::NUM_BANDS);

        // Calculate all band energies first for global decisions
        for (int i = 0; i < CodecParams::NUM_BANDS; ++i) {
            int start = CodecParams::BAND_OFFSETS[i];
            int end = CodecParams::BAND_OFFSETS[i+1];
            double current_band_energy = 0;
            for (int j = start; j < end; ++j) {
                current_band_energy += channel_mdct_coeffs[j] * channel_mdct_coeffs[j];
            }
            band_energies[i] = current_band_energy;
            total_channel_energy += current_band_energy;
        }

        for (int i = 0; i < CodecParams::NUM_BANDS; ++i) {
            AudioTypes::BandData& current_band_data = channel_data.bands_data[i];
            int start_coeff_idx = CodecParams::BAND_OFFSETS[i];
            int end_coeff_idx = CodecParams::BAND_OFFSETS[i+1];
            int num_coeffs_in_band = end_coeff_idx - start_coeff_idx;

            AudioTypes::MdctCoeffs band_coeffs_view(num_coeffs_in_band);
            for(int k=0; k<num_coeffs_in_band; ++k) {
                band_coeffs_view[k] = channel_mdct_coeffs[start_coeff_idx + k];
            }

            // 5.4 Band Importance Evaluation
            current_band_data.importance = Psychoacoustic::evaluateBandImportance(band_coeffs_view, i, total_channel_energy, band_energies);
            
            if (current_band_data.importance == CodecParams::BandImportance::HIGH) {
                // 5.5 High Importance: Cepstrum + PVQ
                AudioTypes::MdctCoeffs cep = Cepstrum::calculateCepstrum(band_coeffs_view, CodecParams::N_CEP_HIGH);
                current_band_data.cepstrum_indices = Quantizer::quantizeCepstrumHigh(cep);
                
                AudioTypes::MdctCoeffs dequant_cep = Quantizer::dequantizeCepstrumHigh(current_band_data.cepstrum_indices);
                AudioTypes::MdctCoeffs envelope = Cepstrum::reconstructEnvelopeFromCepstrum(dequant_cep, num_coeffs_in_band);

                AudioTypes::MdctCoeffs residual(num_coeffs_in_band);
                double residual_norm_sq = 0.0;
                for(int k=0; k<num_coeffs_in_band; ++k) {
                    // Residual: Original_Amp - Reconstructed_Envelope_Amp
                    // Or, Original_Amp / Reconstructed_Envelope_Amp (if ratio)
                    // Spec says "difference or ratio". Let's use difference for amplitude.
                    // For PVQ, typically signs are handled or magnitude is PVQ'd and signs sent separately.
                    // Here, let's assume we are PVQ'ing the signed residual directly after normalization.
                    // Or, PVQ the magnitudes and send signs separately.
                    // For simplicity, PVQ the (original MDCT coeff / reconstructed envelope) ratio
                    // This is more like how Opus does it with CELT.
                    if (std::abs(envelope[k]) > 1e-6) {
                        residual[k] = band_coeffs_view[k] / envelope[k];
                    } else {
                         // If envelope is zero, residual is tricky. Maybe original coeff if non-zero, or zero.
                         // If envelope is ~0 and original is also ~0, residual is ~0.
                         // If envelope is ~0 and original is not, then it's a large value.
                         // This needs careful handling. For now, if envelope is tiny, set residual to 0 or a large capped value.
                         residual[k] = (std::abs(band_coeffs_view[k]) > 1e-6) ? (band_coeffs_view[k] > 0 ? 10.0 : -10.0) : 0.0;
                    }
                    residual_norm_sq += residual[k] * residual[k];
                }
                
                double gain = (num_coeffs_in_band > 0 && residual_norm_sq > 1e-9) ? std::sqrt(residual_norm_sq / num_coeffs_in_band) : 1.0; // RMS of residual before normalization
                // Normalize residual vector for PVQ
                if (gain > 1e-6) {
                    for(int k=0; k<num_coeffs_in_band; ++k) residual[k] /= gain;
                } else { // if gain is effectively zero, residual is already zero
                    std::fill(residual.begin(), residual.end(), 0.0);
                }


                current_band_data.gain_index = Quantizer::quantizePVQGain(gain);
                
                // PVQ parameters for stub
                int K_dim_pvq; // Will be set by pvqEncode, max CodecParams::PVQ_MAX_K_HIGH
                // L_pulses (pulse_count) should be determined adaptively. For stub, pvqEncode sets it.
                current_band_data.pvq_indices_payload = Pvq::pvqEncode(residual, K_dim_pvq, current_band_data.pulse_count);
                current_band_data.pulse_count = Quantizer::quantizePVQPulses(current_band_data.pulse_count); // Ensure it fits in bits

            } else if (current_band_data.importance == CodecParams::BandImportance::MEDIUM) {
                // 5.6 Medium Importance: Envelope Quantization
                AudioTypes::MdctCoeffs cep = Cepstrum::calculateCepstrum(band_coeffs_view, CodecParams::N_CEP_MID);
                current_band_data.cepstrum_indices = Quantizer::quantizeCepstrumMid(cep);
            } else { // LOW Importance
                // 5.7 Low Importance: No data sent from encoder beyond importance flag
            }
        }
    }
};


// --- decoder.h/cpp ---
class Decoder {
public:
    Decoder() : mdct_(CodecParams::MDCT_FRAME_SIZE_N) {
        Psychoacoustic::initializePsychoacousticModel(CodecParams::MDCT_FRAME_SIZE_N, CodecParams::SAMPLING_FREQUENCY);
        // Initialize overlap buffers (for first frame output)
        // These store N/2 samples from the previously *reconstructed* N-sample block
        // Specifically, the second half of the IMDCT output.
        overlap_add_buffer_ch1_.assign(CodecParams::NEW_SAMPLES_PER_FRAME, 0.0f);
        overlap_add_buffer_ch2_.assign(CodecParams::NEW_SAMPLES_PER_FRAME, 0.0f);
    }

    AudioTypes::PcmFrame decodeFrame(const AudioTypes::EncodedFrameData& encoded_data) {
        AudioTypes::PcmFrame pcm_frame_out;
        pcm_frame_out.is_stereo = (encoded_data.channel_mode == CodecParams::ChannelMode::STEREO_MS);
        
        AudioTypes::MdctCoeffs reconstructed_mdct_ch1(CodecParams::MDCT_COEFFS_PER_CHANNEL);
        AudioTypes::MdctCoeffs reconstructed_mdct_ch2;
        if (pcm_frame_out.is_stereo) {
            reconstructed_mdct_ch2.resize(CodecParams::MDCT_COEFFS_PER_CHANNEL);
        }

        reconstructChannelMdct(encoded_data.ch1_data, reconstructed_mdct_ch1, nullptr); // No previous band for ch1 start
        if (pcm_frame_out.is_stereo) {
            reconstructChannelMdct(encoded_data.ch2_data, reconstructed_mdct_ch2, nullptr);
        }
        
        // Perform IMDCT and Overlap-Add
        AudioTypes::Frame current_output_ch1(CodecParams::NEW_SAMPLES_PER_FRAME);
        performImdctAndOverlapAdd(reconstructed_mdct_ch1, overlap_add_buffer_ch1_, current_output_ch1);
        pcm_frame_out.channel_left = current_output_ch1;

        if (pcm_frame_out.is_stereo) {
            AudioTypes::Frame current_output_ch2(CodecParams::NEW_SAMPLES_PER_FRAME);
            performImdctAndOverlapAdd(reconstructed_mdct_ch2, overlap_add_buffer_ch2_, current_output_ch2);
            pcm_frame_out.channel_right = current_output_ch2;

            // Inverse MS Processing if applied
            if (encoded_data.ms_applied) {
                MsProcessor::inverseMSProcessing(pcm_frame_out.channel_left, pcm_frame_out.channel_right, true);
            }
        }
        
        return pcm_frame_out;
    }

private:
    MdctHandler::MDCT mdct_;
    AudioTypes::Frame overlap_add_buffer_ch1_; // Stores N/2 samples for overlap-add from previous IMDCT
    AudioTypes::Frame overlap_add_buffer_ch2_;

    void reconstructChannelMdct(const AudioTypes::EncodedChannelData& channel_data,
                                AudioTypes::MdctCoeffs& full_channel_mdct_coeffs,
                                const AudioTypes::MdctCoeffs* /* prev_channel_coeffs_for_low_pred - unused in this simple stub */) {
        const AudioTypes::MdctCoeffs* prev_band_reconst_coeffs_ptr = nullptr; // For LOW importance prediction

        for (int i = 0; i < CodecParams::NUM_BANDS; ++i) {
            const AudioTypes::BandData& current_band_data = channel_data.bands_data[i];
            int start_coeff_idx = CodecParams::BAND_OFFSETS[i];
            int end_coeff_idx = CodecParams::BAND_OFFSETS[i+1];
            int num_coeffs_in_band = end_coeff_idx - start_coeff_idx;
            AudioTypes::MdctCoeffs band_reconst_coeffs(num_coeffs_in_band);

            if (current_band_data.importance == CodecParams::BandImportance::HIGH) {
                AudioTypes::MdctCoeffs dequant_cep = Quantizer::dequantizeCepstrumHigh(current_band_data.cepstrum_indices);
                AudioTypes::MdctCoeffs envelope = Cepstrum::reconstructEnvelopeFromCepstrum(dequant_cep, num_coeffs_in_band);
                
                double gain = Quantizer::dequantizePVQGain(current_band_data.gain_index);
                int pulses = Quantizer::dequantizePVQPulses(current_band_data.pulse_count);
                
                // PVQ decode needs K_dim. K_dim for PVQ was determined at encode time.
                // Here, we assume K_dim is num_coeffs_in_band clipped to PVQ_MAX_K_HIGH.
                int K_dim_pvq = std::min(num_coeffs_in_band, CodecParams::PVQ_MAX_K_HIGH);
                // If num_coeffs_in_band > PVQ_MAX_K_HIGH, spec says "subvectorに分割してPVQを複数回適用"
                // This stub does not implement subvector splitting. It processes only one block up to PVQ_MAX_K_HIGH.
                
                AudioTypes::MdctCoeffs normalized_residual;
                if (K_dim_pvq > 0) {
                     normalized_residual = Pvq::pvqDecode(current_band_data.pvq_indices_payload, K_dim_pvq, pulses);
                } else {
                     normalized_residual.assign(num_coeffs_in_band, 0.0); // K_dim_pvq might be 0 if band is tiny
                }


                // Resize normalized_residual if K_dim_pvq < num_coeffs_in_band (due to PVQ_MAX_K_HIGH clipping)
                // This would mean only part of the band's residual was coded. Fill rest with 0.
                if (K_dim_pvq < num_coeffs_in_band) {
                    normalized_residual.resize(num_coeffs_in_band, 0.0);
                }


                for (int k = 0; k < num_coeffs_in_band; ++k) {
                    // Reconstruct: Envelope_Amplitude * (Gain * Normalized_Residual_Element)
                    // Or, if residual was (Orig/Env), then Orig = Env * (Gain * NormRes)
                    band_reconst_coeffs[k] = envelope[k] * gain * normalized_residual[k];
                     // Phase: Spec says "位相は0またはランダム（あるいは符号情報を別途PVQで送る）"
                     // Current stub for PVQ implicitly encodes sign in the "payload".
                     // If envelope is only magnitude, and residual is also magnitude, sign needs to be added.
                     // For simplicity, assume pvqDecode provides signed normalized residual.
                }
                prev_band_reconst_coeffs_ptr = &band_reconst_coeffs; // Store for potential next low-band prediction

            } else if (current_band_data.importance == CodecParams::BandImportance::MEDIUM) {
                AudioTypes::MdctCoeffs dequant_cep = Quantizer::dequantizeCepstrumMid(current_band_data.cepstrum_indices);
                AudioTypes::MdctCoeffs envelope = Cepstrum::reconstructEnvelopeFromCepstrum(dequant_cep, num_coeffs_in_band);
                // Medium importance sends only envelope. Phase is random or 0.
                // For simplicity, use envelope as magnitude, phase 0.
                band_reconst_coeffs = envelope;
                prev_band_reconst_coeffs_ptr = &band_reconst_coeffs;

            } else { // LOW Importance
                // 5.7 Decode時の予測 / 無効化
                bool predicted = false;
                if (prev_band_reconst_coeffs_ptr && !prev_band_reconst_coeffs_ptr->empty()) {
                    // Predict from adjacent lower band (if it was High or Mid)
                    // Normalized shape copy, scaled by ATH or very low energy.
                    // This is a very simplified prediction.
                    // A better prediction would use the normalized *envelope* of the previous band.
                    AudioTypes::MdctCoeffs prev_env_shape = *prev_band_reconst_coeffs_ptr; // Use coeffs as shape for simplicity
                    double prev_norm_sq = 0.0;
                    for(double val : prev_env_shape) prev_norm_sq += val*val;
                    if (prev_norm_sq > 1e-9) {
                        double inv_sqrt_prev_norm = 1.0 / std::sqrt(prev_norm_sq);
                        for(double& val : prev_env_shape) val *= inv_sqrt_prev_norm; // Normalize shape
                    }

                    double target_energy_low_band = Psychoacoustic::precomputed_ath_power_per_band[i] * num_coeffs_in_band; // Energy from ATH
                    // Or fixed low energy: e.g. 0.01 of ATH for "barely audible"
                    // target_energy_low_band = 0.01 * Psychoacoustic::precomputed_ath_power_per_band[i] * num_coeffs_in_band;

                    double scale_factor = std::sqrt(target_energy_low_band); // Scale for amplitude

                    for(int k=0; k < num_coeffs_in_band; ++k) {
                        // Repeat or interpolate prev_env_shape if sizes differ
                        int prev_k = k % prev_env_shape.size(); // Simple repeat
                        band_reconst_coeffs[k] = prev_env_shape[prev_k] * scale_factor;
                    }
                    predicted = true;
                }
                
                if (!predicted) { // No valid previous band or prediction failed
                    std::fill(band_reconst_coeffs.begin(), band_reconst_coeffs.end(), 0.0); // Invalidate (set to zero)
                }
                // For LOW bands, we don't update prev_band_reconst_coeffs_ptr,
                // so prediction always comes from a HIGH or MID band.
            }

            // Copy reconstructed band coefficients to the full channel MDCT array
            for (int k = 0; k < num_coeffs_in_band; ++k) {
                full_channel_mdct_coeffs[start_coeff_idx + k] = band_reconst_coeffs[k];
            }
        }
    }

    void performImdctAndOverlapAdd(const AudioTypes::MdctCoeffs& channel_coeffs,
                                   AudioTypes::Frame& overlap_buffer, // N/2 samples from prev IMDCT's second half
                                   AudioTypes::Frame& output_pcm_frame) { // N/2 final output samples
        
        std::vector<double> imdct_output_block_double(CodecParams::MDCT_FRAME_SIZE_N);
        mdct_.inverse_transform(channel_coeffs, imdct_output_block_double); // Get N samples
        MdctHandler::applyWindow(imdct_output_block_double); // Apply window again for synthesis (TDAC)

        output_pcm_frame.resize(CodecParams::NEW_SAMPLES_PER_FRAME);

        // Overlap-add: first N/2 of imdct_output_block with overlap_buffer
        for (int i = 0; i < CodecParams::NEW_SAMPLES_PER_FRAME; ++i) {
            output_pcm_frame[i] = static_cast<AudioTypes::Sample>(imdct_output_block_double[i]) + overlap_buffer[i];
        }
        
        // Store second N/2 of imdct_output_block into overlap_buffer for next frame
        for (int i = 0; i < CodecParams::NEW_SAMPLES_PER_FRAME; ++i) {
            overlap_buffer[i] = static_cast<AudioTypes::Sample>(imdct_output_block_double[i + CodecParams::NEW_SAMPLES_PER_FRAME]);
        }
    }
};


// --- Bitstream serialization/deserialization logic for EncodedFrameData ---
namespace FrameSerializer {
    void serializeFrameData(const AudioTypes::EncodedFrameData& frame_data, Bitstream::BitstreamWriter& writer) {
        // MS処理フラグ (チャンネルモードがStereo/MSの場合のみ存在)
        if (frame_data.channel_mode == CodecParams::ChannelMode::STEREO_MS) {
            writer.writeBit(frame_data.ms_applied);
        }

        // Channel 1 Data
        for (const auto& band_data : frame_data.ch1_data.bands_data) {
            writer.writeBits(static_cast<uint8_t>(band_data.importance), 2);
            if (band_data.importance == CodecParams::BandImportance::HIGH) {
                for (int cep_idx : band_data.cepstrum_indices) writer.writeBits(cep_idx, CodecParams::BITS_PER_CEP_HIGH_COEFF);
                writer.writeBits(band_data.gain_index, CodecParams::BITS_PVQ_GAIN_HIGH);
                writer.writeBits(band_data.pulse_count, CodecParams::BITS_PVQ_PULSES_HIGH);
                // PVQインデックス (可変長) - For stub, this is a byte array.
                // A real PPMd would handle symbols more granularly.
                // Here, write size then bytes. Or PPMd handles termination.
                // For this simple bitstream, let's write size of PVQ payload first (e.g. 8 bits for size up to 255 bytes)
                // This part is tricky with PPMd. If PPMd is byte-oriented, PVQ payload can be passed.
                // If bit-oriented, PPMd needs to be fed bits.
                // The spec says "PVQインデックスはPPMdにより符号化される"
                // This implies the raw PVQ index (a large integer) is the symbol.
                // Our stub pvq_indices_payload is a byte array. Let's just write its size and then content.
                // This part needs refinement if a real PPMd is used.
                writer.writeBits(band_data.pvq_indices_payload.size(), 8); // Size of payload (max 255 bytes for K_dim_pvq=64)
                writer.writeBytes(band_data.pvq_indices_payload.data(), band_data.pvq_indices_payload.size());

            } else if (band_data.importance == CodecParams::BandImportance::MEDIUM) {
                for (int cep_idx : band_data.cepstrum_indices) writer.writeBits(cep_idx, CodecParams::BITS_PER_CEP_MID_COEFF);
            }
            // LOW: no data
        }

        // Channel 2 Data (if stereo)
        if (frame_data.channel_mode == CodecParams::ChannelMode::STEREO_MS) {
            for (const auto& band_data : frame_data.ch2_data.bands_data) {
                writer.writeBits(static_cast<uint8_t>(band_data.importance), 2);
                if (band_data.importance == CodecParams::BandImportance::HIGH) {
                    for (int cep_idx : band_data.cepstrum_indices) writer.writeBits(cep_idx, CodecParams::BITS_PER_CEP_HIGH_COEFF);
                    writer.writeBits(band_data.gain_index, CodecParams::BITS_PVQ_GAIN_HIGH);
                    writer.writeBits(band_data.pulse_count, CodecParams::BITS_PVQ_PULSES_HIGH);
                    writer.writeBits(band_data.pvq_indices_payload.size(), 8);
                    writer.writeBytes(band_data.pvq_indices_payload.data(), band_data.pvq_indices_payload.size());
                } else if (band_data.importance == CodecParams::BandImportance::MEDIUM) {
                    for (int cep_idx : band_data.cepstrum_indices) writer.writeBits(cep_idx, CodecParams::BITS_PER_CEP_MID_COEFF);
                }
            }
        }
        writer.flushByte(); // Ensure all bits are written to the buffer
    }

    AudioTypes::EncodedFrameData deserializeFrameData(Bitstream::BitstreamReader& reader, CodecParams::ChannelMode ch_mode) {
        AudioTypes::EncodedFrameData frame_data;
        frame_data.channel_mode = ch_mode;

        if (ch_mode == CodecParams::ChannelMode::STEREO_MS) {
            bool ms_flag_val;
            if (!reader.readBit(ms_flag_val)) throw std::runtime_error("Deserializer: Read MS flag failed.");
            frame_data.ms_applied = ms_flag_val;
        }

        // Channel 1 Data
        for (int i = 0; i < CodecParams::NUM_BANDS; ++i) {
            AudioTypes::BandData& band_data = frame_data.ch1_data.bands_data[i];
            uint64_t importance_val;
            if(!reader.readBits(importance_val, 2)) throw std::runtime_error("Deserializer: Read importance ch1 failed.");
            band_data.importance = static_cast<CodecParams::BandImportance>(importance_val);

            if (band_data.importance == CodecParams::BandImportance::HIGH) {
                band_data.cepstrum_indices.resize(CodecParams::N_CEP_HIGH);
                for (int j = 0; j < CodecParams::N_CEP_HIGH; ++j) {
                    uint64_t cep_val;
                    if(!reader.readBits(cep_val, CodecParams::BITS_PER_CEP_HIGH_COEFF)) throw std::runtime_error("Deserializer: Read cep_high ch1 failed.");
                    band_data.cepstrum_indices[j] = cep_val;
                }
                uint64_t gain_val, pulse_val;
                if(!reader.readBits(gain_val, CodecParams::BITS_PVQ_GAIN_HIGH)) throw std::runtime_error("Deserializer: Read gain_high ch1 failed.");
                band_data.gain_index = gain_val;
                if(!reader.readBits(pulse_val, CodecParams::BITS_PVQ_PULSES_HIGH)) throw std::runtime_error("Deserializer: Read pulse_high ch1 failed.");
                band_data.pulse_count = pulse_val;
                
                uint64_t pvq_payload_size;
                if(!reader.readBits(pvq_payload_size, 8)) throw std::runtime_error("Deserializer: Read pvq_size_high ch1 failed.");
                band_data.pvq_indices_payload.resize(pvq_payload_size);
                reader.alignToByte(); // PVQ payload is bytes
                if(!reader.readBytes(band_data.pvq_indices_payload.data(), pvq_payload_size)) throw std::runtime_error("Deserializer: Read pvq_payload_high ch1 failed.");


            } else if (band_data.importance == CodecParams::BandImportance::MEDIUM) {
                band_data.cepstrum_indices.resize(CodecParams::N_CEP_MID);
                 for (int j = 0; j < CodecParams::N_CEP_MID; ++j) {
                    uint64_t cep_val;
                    if(!reader.readBits(cep_val, CodecParams::BITS_PER_CEP_MID_COEFF)) throw std::runtime_error("Deserializer: Read cep_mid ch1 failed.");
                    band_data.cepstrum_indices[j] = cep_val;
                }
            }
        }

        // Channel 2 Data
        if (ch_mode == CodecParams::ChannelMode::STEREO_MS) {
            for (int i = 0; i < CodecParams::NUM_BANDS; ++i) {
                AudioTypes::BandData& band_data = frame_data.ch2_data.bands_data[i];
                uint64_t importance_val;
                if(!reader.readBits(importance_val, 2)) throw std::runtime_error("Deserializer: Read importance ch2 failed.");
                band_data.importance = static_cast<CodecParams::BandImportance>(importance_val);

                if (band_data.importance == CodecParams::BandImportance::HIGH) {
                    band_data.cepstrum_indices.resize(CodecParams::N_CEP_HIGH);
                    for (int j = 0; j < CodecParams::N_CEP_HIGH; ++j) {
                        uint64_t cep_val;
                        if(!reader.readBits(cep_val, CodecParams::BITS_PER_CEP_HIGH_COEFF)) throw std::runtime_error("Deserializer: Read cep_high ch2 failed.");
                        band_data.cepstrum_indices[j] = cep_val;
                    }
                    uint64_t gain_val, pulse_val;
                    if(!reader.readBits(gain_val, CodecParams::BITS_PVQ_GAIN_HIGH)) throw std::runtime_error("Deserializer: Read gain_high ch2 failed.");
                    band_data.gain_index = gain_val;
                    if(!reader.readBits(pulse_val, CodecParams::BITS_PVQ_PULSES_HIGH)) throw std::runtime_error("Deserializer: Read pulse_high ch2 failed.");
                    band_data.pulse_count = pulse_val;
                    
                    uint64_t pvq_payload_size;
                    if(!reader.readBits(pvq_payload_size, 8)) throw std::runtime_error("Deserializer: Read pvq_size_high ch2 failed.");
                    band_data.pvq_indices_payload.resize(pvq_payload_size);
                    reader.alignToByte();
                    if(!reader.readBytes(band_data.pvq_indices_payload.data(), pvq_payload_size)) throw std::runtime_error("Deserializer: Read pvq_payload_high ch2 failed.");

                } else if (band_data.importance == CodecParams::BandImportance::MEDIUM) {
                    band_data.cepstrum_indices.resize(CodecParams::N_CEP_MID);
                    for (int j = 0; j < CodecParams::N_CEP_MID; ++j) {
                        uint64_t cep_val;
                        if(!reader.readBits(cep_val, CodecParams::BITS_PER_CEP_MID_COEFF)) throw std::runtime_error("Deserializer: Read cep_mid ch2 failed.");
                        band_data.cepstrum_indices[j] = cep_val;
                    }
                }
            }
        }
        reader.alignToByte(); // Ensure reader is byte-aligned for next frame header or PPMd block
        return frame_data;
    }
}


// --- main.cpp (Simplified structure) ---
void encodeFile(const std::string& input_wav_path, const std::string& output_custom_path) {
    WavIO::WavReader reader(input_wav_path);
    if (!reader.isOpen()) {
        std::cerr << "Error opening input WAV file: " << input_wav_path << std::endl;
        return;
    }
    const WavIO::WavHeader& wav_header = reader.getHeader();
    if (wav_header.sample_rate != CodecParams::SAMPLING_FREQUENCY) {
         std::cerr << "Input WAV sample rate (" << wav_header.sample_rate << ") does not match codec (" << CodecParams::SAMPLING_FREQUENCY << ")" << std::endl;
        return;
    }
     if (wav_header.bits_per_sample != 16 && wav_header.bits_per_sample != 24) {
        std::cerr << "Input WAV bit depth (" << wav_header.bits_per_sample << ") not supported (only 16 or 24 bit)." << std::endl;
        return;
    }


    std::ofstream out_file(output_custom_path, std::ios::binary);
    if (!out_file.is_open()) {
        std::cerr << "Error opening output file: " << output_custom_path << std::endl;
        return;
    }

    Encoder audio_encoder;
    Ppmd::PPMdEncoder ppmd_encoder(CodecParams::PPMD_MAX_CONTEXT_ORDER, CodecParams::PPMD_MEMORY_LIMIT);

    AudioTypes::PcmFrame pcm_frame;
    int frame_count = 0;
    std::cout << "Encoding..." << std::endl;

    while (reader.readFrame(pcm_frame, CodecParams::NEW_SAMPLES_PER_FRAME)) {
        if (pcm_frame.channel_left.size() < CodecParams::NEW_SAMPLES_PER_FRAME) {
            // Handle partial last frame: pad with zeros to full NEW_SAMPLES_PER_FRAME
            // This is important for MDCT block construction.
            size_t needed_padding = CodecParams::NEW_SAMPLES_PER_FRAME - pcm_frame.channel_left.size();
            pcm_frame.channel_left.insert(pcm_frame.channel_left.end(), needed_padding, 0.0f);
            if (pcm_frame.is_stereo && !pcm_frame.channel_right.empty()) {
                pcm_frame.channel_right.insert(pcm_frame.channel_right.end(), needed_padding, 0.0f);
            } else if (pcm_frame.is_stereo && pcm_frame.channel_right.empty() && wav_header.num_channels == 2) {
                 // Should not happen if WavReader works correctly
                pcm_frame.channel_right.assign(CodecParams::NEW_SAMPLES_PER_FRAME, 0.0f);
                std::fill(pcm_frame.channel_right.begin() + (CodecParams::NEW_SAMPLES_PER_FRAME - needed_padding), pcm_frame.channel_right.end(), 0.0f);
            }
        }
        
        AudioTypes::EncodedFrameData encoded_frame_data = audio_encoder.encodeFrame(pcm_frame);

        Bitstream::BitstreamWriter bit_writer;
        FrameSerializer::serializeFrameData(encoded_frame_data, bit_writer);
        std::vector<uint8_t> serialized_payload = bit_writer.getBuffer();
        
        std::vector<uint8_t> ppmd_encoded_payload;
        ppmd_encoder.encodeData(serialized_payload, ppmd_encoded_payload); // Stub: no compression

        // Write frame header (PPMd対象外)
        uint32_t sync = CodecParams::SYNC_WORD;
        uint16_t payload_len = static_cast<uint16_t>(ppmd_encoded_payload.size());
        uint8_t sr_idx = static_cast<uint8_t>(CodecParams::SamplingRateIndex::SR_48000); // Hardcoded for now
        uint8_t ch_mode_val = static_cast<uint8_t>(encoded_frame_data.channel_mode);

        out_file.write(reinterpret_cast<const char*>(&sync), sizeof(sync));
        out_file.write(reinterpret_cast<const char*>(&payload_len), sizeof(payload_len));
        out_file.write(reinterpret_cast<const char*>(&sr_idx), sizeof(sr_idx));
        out_file.write(reinterpret_cast<const char*>(&ch_mode_val), sizeof(ch_mode_val));
        
        // Write PPMd payload
        out_file.write(reinterpret_cast<const char*>(ppmd_encoded_payload.data()), ppmd_encoded_payload.size());
        
        frame_count++;
        if (frame_count % 50 == 0) std::cout << "Encoded frame " << frame_count << std::endl;
    }
    std::cout << "Encoding finished. Total frames: " << frame_count << std::endl;
    out_file.close();
}

void decodeFile(const std::string& input_custom_path, const std::string& output_wav_path) {
    std::ifstream in_file(input_custom_path, std::ios::binary);
    if (!in_file.is_open()) {
        std::cerr << "Error opening input custom file: " << input_custom_path << std::endl;
        return;
    }

    Decoder audio_decoder;
    Ppmd::PPMdDecoder ppmd_decoder(CodecParams::PPMD_MAX_CONTEXT_ORDER, CodecParams::PPMD_MEMORY_LIMIT);
    
    WavIO::WavWriter* writer = nullptr; // Initialize later when we know num_channels

    int frame_count = 0;
    std::cout << "Decoding..." << std::endl;

    while (in_file.peek() != EOF) {
        uint32_t sync;
        uint16_t payload_len_bytes;
        uint8_t sr_idx_val;
        uint8_t ch_mode_val;

        in_file.read(reinterpret_cast<char*>(&sync), sizeof(sync));
        if (in_file.gcount() < sizeof(sync)) break; // EOF
        if (sync != CodecParams::SYNC_WORD) {
            std::cerr << "Decoder: Sync word mismatch on frame " << frame_count << "! Aborting." << std::endl;
            break;
        }
        in_file.read(reinterpret_cast<char*>(&payload_len_bytes), sizeof(payload_len_bytes));
        in_file.read(reinterpret_cast<char*>(&sr_idx_val), sizeof(sr_idx_val));
        in_file.read(reinterpret_cast<char*>(&ch_mode_val), sizeof(ch_mode_val));
        if (in_file.gcount() < (sizeof(payload_len_bytes) + sizeof(sr_idx_val) + sizeof(ch_mode_val))) {
             std::cerr << "Decoder: Could not read full frame header. EOF?" << std::endl;
             break;
        }


        CodecParams::SamplingRateIndex sr_idx = static_cast<CodecParams::SamplingRateIndex>(sr_idx_val);
        CodecParams::ChannelMode ch_mode = static_cast<CodecParams::ChannelMode>(ch_mode_val);

        if (sr_idx != CodecParams::SamplingRateIndex::SR_48000) {
            std::cerr << "Decoder: Unsupported sampling rate index in frame header. Aborting." << std::endl;
            break;
        }
        
        if (!writer) { // First frame, initialize WAV writer
            uint16_t num_wav_channels = (ch_mode == CodecParams::ChannelMode::MONO) ? 1 : 2;
            // Assuming 16-bit output for decoded WAV for simplicity
            writer = new WavIO::WavWriter(output_wav_path, num_wav_channels, CodecParams::SAMPLING_FREQUENCY, 16);
            if (!writer->isOpen()) {
                std::cerr << "Error opening output WAV file: " << output_wav_path << std::endl;
                delete writer; writer = nullptr;
                return;
            }
        }


        std::vector<uint8_t> ppmd_encoded_payload(payload_len_bytes);
        in_file.read(reinterpret_cast<char*>(ppmd_encoded_payload.data()), payload_len_bytes);
         if (in_file.gcount() < payload_len_bytes) {
             std::cerr << "Decoder: Could not read full PPMd payload. EOF?" << std::endl;
             break;
        }


        std::vector<uint8_t> serialized_payload;
        // The expected_decoded_size for PPMd is tricky if it's not self-terminating or length-prefixed.
        // Our stub just copies, so serialized_payload will be same as ppmd_encoded_payload.
        // A real PPMd would decompress. The length of serialized_payload after PPMd decoding is unknown
        // unless the PPMd stream itself encodes it or the bitstream structure does.
        // For this design, the FrameSerializer::deserializeFrameData consumes bits as needed.
        // The PPMd output (serialized_payload) length should ideally match what FrameSerializer expects.
        // For the stub, let's assume PPMd output length is the same as input.
        ppmd_decoder.decodeData(ppmd_encoded_payload, serialized_payload, ppmd_encoded_payload.size()); 

        Bitstream::BitstreamReader bit_reader(serialized_payload);
        AudioTypes::EncodedFrameData encoded_frame_data;
        try {
            encoded_frame_data = FrameSerializer::deserializeFrameData(bit_reader, ch_mode);
        } catch (const std::runtime_error& e) {
            std::cerr << "Error deserializing frame " << frame_count << ": " << e.what() << std::endl;
            break;
        }
        
        AudioTypes::PcmFrame pcm_output_frame = audio_decoder.decodeFrame(encoded_frame_data);
        if (writer) {
            writer->writeFrame(pcm_output_frame);
        }
        
        frame_count++;
         if (frame_count % 50 == 0) std::cout << "Decoded frame " << frame_count << std::endl;
    }
    std::cout << "Decoding finished. Total frames: " << frame_count << std::endl;

    if (writer) {
        writer->finalize();
        delete writer;
    }
    in_file.close();
}


int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage:\n";
        std::cerr << "  " << argv[0] << " encode <input.wav> <output.custom>\n";
        std::cerr << "  " << argv[0] << " decode <input.custom> <output.wav>\n";
        return 1;
    }

    std::string mode = argv[1];
    std::string input_path = argv[2];
    std::string output_path = argv[3];

    try {
        if (mode == "encode") {
            encodeFile(input_path, output_path);
        } else if (mode == "decode") {
            decodeFile(input_path, output_path);
        } else {
            std::cerr << "Invalid mode. Use 'encode' or 'decode'." << std::endl;
            return 1;
        }
    } catch (const std::exception& e) {
        std::cerr << "An error occurred: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
