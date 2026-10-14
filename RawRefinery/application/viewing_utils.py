import numpy as np
from PySide6.QtGui import QImage


# Linear Rec.2020 (D65) -> linear sRGB, used to display the pipeline's working space.
_REC2020_TO_XYZ = np.array([
    [0.636958, 0.144617, 0.168881],
    [0.262700, 0.677998, 0.059302],
    [0.000000, 0.028073, 1.060985],
])
_XYZ_TO_SRGB = np.array([
    [ 3.2404542, -1.5371385, -0.4985314],
    [-0.9692660,  1.8760108,  0.0415560],
    [ 0.0556434, -0.2040259,  1.0572252],
])
REC2020_TO_SRGB = (_XYZ_TO_SRGB @ _REC2020_TO_XYZ).astype(np.float32)


def numpy_to_qimage_rgb(array, gamma_correct=True, exposure=1):
    """Convert a linear Rec.2020 (H, W, 3) array to a displayable sRGB QImage.

    With gamma_correct=False the array is assumed to be display-ready already.
    """
    height, width, channels = array.shape
    if channels != 3:
        raise ValueError("Expected array shape (H, W, 3) for RGB")

    array = array * exposure
    if gamma_correct:
        array = np.clip(array @ REC2020_TO_SRGB.T, 0, 1)
        array = linear_to_srgb(array)
    array_uint8 = np.ascontiguousarray((np.clip(array, 0, 1) * 255).astype(np.uint8))

    # QImage does not own the buffer, so copy it before array_uint8 goes out of scope.
    bytes_per_line = 3 * width
    return QImage(array_uint8.data, width, height, bytes_per_line, QImage.Format_RGB888).copy()

def apply_gamma(img, gamma: float = 2.2):
    img = img.clip(0, 1)  
    return (img ** (1.0 / gamma)).clip(0, 1) 

def apply_gamma_v2(tensor):   
    img_mask = tensor > 0.0031308
    tensor[img_mask] = (
        1.055 * np.pow(tensor[img_mask], 1.0 / 2.4) - 0.055
    )
    tensor[~img_mask] *= 12.92
    return tensor


def linear_to_srgb(x):
    a = 0.055
    return np.where(x <= 0.0031308, 12.92 * x, (1 + a) * np.power(x, 1/2.4) - a)