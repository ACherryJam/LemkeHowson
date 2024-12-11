import numpy as np

ROUNDING_EPS = 1e-10


def fix_zero_rounding_errors(matrix: np.ndarray) -> np.ndarray:
    """Заменяет значения, близкие к нуля, на ноль"""
    matrix[np.isclose(matrix, 0)] = np.float64(0)
    return matrix

