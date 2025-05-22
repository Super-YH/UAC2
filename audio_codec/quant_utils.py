import numpy as np

EPSILON = 1e-12 # Small constant for numerical stability

def quantize_scalar(value: float, levels_table: np.ndarray) -> tuple[int, float]:
    """
    Quantizes a scalar value using a given table of quantization levels.
    Finds the closest level and returns its index and value.
    """
    if not isinstance(levels_table, np.ndarray) or levels_table.ndim != 1:
        raise ValueError("levels_table must be a 1D NumPy array.")
    if levels_table.size == 0:
        raise ValueError("levels_table cannot be empty.")

    # Find the index of the closest quantization level
    index = np.argmin(np.abs(levels_table - value))
    quantized_value = levels_table[index]
    return int(index), float(quantized_value)

def dequantize_scalar(index: int, levels_table: np.ndarray) -> float:
    """
    Dequantizes an index using a given table of quantization levels.
    """
    if not isinstance(levels_table, np.ndarray) or levels_table.ndim != 1:
        raise ValueError("levels_table must be a 1D NumPy array.")
    if not (0 <= index < len(levels_table)):
        raise ValueError(f"Index {index} out of bounds for levels_table of size {len(levels_table)}")
    return float(levels_table[index])

# Gain Quantization Helpers - can also be here if they are generic enough.
# For now, keeping them in quantization_high.py as they are specific to that module's needs.
# If other modules need identical gain quantization, these could be moved/generalized.
# LOG_GAIN_MIN = -10.0
# LOG_GAIN_MAX = 10.0
# def quantize_gain_log(gain: float, num_bits: int) -> tuple[int, float]: ...
# def dequantize_gain_log(index: int, num_bits: int) -> float: ...
