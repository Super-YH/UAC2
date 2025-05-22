# Global parameters for the audio codec

SAMPLE_RATE = 48000  # Hz
MDCT_FRAME_SIZE = 2048  # samples
NEW_SAMPLES_PER_FRAME = 1024  # samples
NUM_MDCT_COEFFS = MDCT_FRAME_SIZE // 2
MDCT_WINDOW_TYPE = 'sine'  # or an enum/constant for it
PPMD_MAX_ORDER = 5
PPMD_MEMORY = 16 * 1024 * 1024  # bytes
NUM_BANDS = 16
BAND_BOUNDARIES = [0, 16, 32, 48, 64, 96, 128, 160, 192, 256, 320, 384, 512, 640, 768, 896, 1024]
