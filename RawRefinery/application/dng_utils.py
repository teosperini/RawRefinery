import json
import shutil
import subprocess
import numpy as np
from pidng.core import RAW2DNG, DNGTags, Tag
from pidng.defs import *

EXIFTOOL_PATHS = ("/opt/homebrew/bin/exiftool", "/usr/local/bin/exiftool", "/usr/bin/exiftool")


def find_exiftool():
    """exiftool path, also when the app is started without the shell PATH (e.g. from the Dock)."""
    found = shutil.which("exiftool")
    if found:
        return found
    return next((p for p in EXIFTOOL_PATHS if shutil.which(p)), None)


def _default_crop(source_path, dng_path, exiftool):
    """DefaultCropOrigin/Size so the DNG shows the same area as the original file.

    The DNG holds the whole visible raw area, including a few masked border pixels.
    """
    try:
        import rawpy
        out = subprocess.run(
            [exiftool, "-json", "-n", "-ImageWidth", "-ImageHeight", "-SensorLeftBorder", "-SensorTopBorder",
             source_path, dng_path], capture_output=True, text=True, check=True).stdout
        src, dng = json.loads(out)
        crop_w, crop_h = src["ImageWidth"], src["ImageHeight"]
        dng_w, dng_h = dng["ImageWidth"], dng["ImageHeight"]
        if crop_w >= dng_w and crop_h >= dng_h:
            return None
        if "SensorLeftBorder" in src and "SensorTopBorder" in src:
            with rawpy.imread(source_path) as raw:
                left = src["SensorLeftBorder"] - raw.sizes.left_margin
                top = src["SensorTopBorder"] - raw.sizes.top_margin
        else:
            left, top = (dng_w - crop_w) // 2, (dng_h - crop_h) // 2
        if left < 0 or top < 0 or left + crop_w > dng_w or top + crop_h > dng_h:
            left, top = (dng_w - crop_w) // 2, (dng_h - crop_h) // 2
        return left, top, min(crop_w, dng_w), min(crop_h, dng_h)
    except Exception:
        return None


def copy_metadata(source_path, dng_path):
    """Copy every metadata tag (EXIF, GPS, XMP, maker notes, dates...) from the original file into the DNG.

    Returns (ok, message). The DNG is only modified in place by exiftool, never the source.
    """
    exiftool = find_exiftool()
    if exiftool is None:
        return False, "exiftool not found (brew install exiftool): metadata not copied"
    cmd = [exiftool, "-overwrite_original", "-TagsFromFile", source_path, "-all:all", "--Orientation"]
    crop = _default_crop(source_path, dng_path, exiftool)
    if crop:
        left, top, width, height = crop
        cmd += [f"-IFD0:DefaultCropOrigin={left} {top}", f"-IFD0:DefaultCropSize={width} {height}"]
    cmd.append(dng_path)
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired) as e:
        return False, f"metadata not copied: {e}"
    if result.returncode != 0 or "1 image files updated" not in result.stdout:
        return False, f"metadata not copied: {(result.stderr or result.stdout).strip()[:200]}"
    return True, "metadata copied from original"

def get_ratios(string, rh):
    return [x.as_integer_ratio() for x in rh.full_metadata[string].values]


def get_as_shot_neutral(rh, denominator=10000):

    cam_mul = rh.core_metadata.camera_white_balance
    
    if cam_mul[0] == 0 or cam_mul[2] == 0:
        return [[denominator, denominator], [denominator, denominator], [denominator, denominator]]

    r_neutral = cam_mul[1] / cam_mul[0]
    g_neutral = 1.0 
    b_neutral = cam_mul[1] / cam_mul[2]

    return [
        [int(r_neutral * denominator), denominator],
        [int(g_neutral * denominator), denominator],
        [int(b_neutral * denominator), denominator],
    ]
def convert_ccm_to_rational(matrix_3x3, denominator=10000):

    numerator_matrix = np.round(matrix_3x3 * denominator).astype(int)
    numerators_flat = numerator_matrix.flatten()
    ccm_rational = [[num, denominator] for num in numerators_flat]
    
    return ccm_rational


   
def simulate_CFA(image, pattern="RGGB", cfa_type="bayer"):
    """
    Simulate a CFA image from an RGB image.

    Args:
        image: numpy array (H, W, 3), RGB image.
        pattern: CFA pattern string, one of {"RGGB","BGGR","GRBG","GBRG"} for Bayer,
                 or ignored if cfa_type="xtrans".
        cfa_type: "bayer" or "xtrans".

    Returns:
        cfa: numpy array (H, W) CFA image.
        sparse_mask:  numpy array (H, W, r), mask of pixels.
    """
    width = image.shape[1]
    height = image.shape[0]
    cfa = np.zeros((height, width, 3), dtype=image.dtype)
    sparse_mask = np.zeros((height, width, 3), dtype=image.dtype)
    if cfa_type == "bayer":
        # 2×2 Bayer masks
        masks = {
            "RGGB": np.array([["R", "G"], ["G", "B"]]),
            "BGGR": np.array([["B", "G"], ["G", "R"]]),
            "GRBG": np.array([["G", "R"], ["B", "G"]]),
            "GBRG": np.array([["G", "B"], ["R", "G"]]),
        }
        if pattern not in masks:
            raise ValueError(f"Unknown Bayer pattern: {pattern}")

        mask = masks[pattern]
        cmap = {"R": 0, "G": 1, "B": 2}
         
        for i in range(2):
            for j in range(2):
                ch = cmap[mask[i, j]]
                cfa[i::2, j::2, ch] = image[i::2, j::2, ch]
                sparse_mask[i::2, j::2, ch] = 1
    elif cfa_type == "xtrans":
        # Fuji X-Trans 6×6 repeating pattern
        xtrans_pattern = np.array([
            ["G","B","R","G","R","B"],
            ["R","G","G","B","G","G"],
            ["B","G","G","R","G","G"],
            ["G","R","B","G","B","R"],
            ["B","G","G","R","G","G"],
            ["R","G","G","B","G","G"],
        ])
        cmap = {"R":0, "G":1, "B":2}

        for i in range(6):
            for j in range(6):
                ch = cmap[xtrans_pattern[i, j]]
                cfa[i::6, j::6, ch] = image[i::6, j::6, ch]
                sparse_mask[i::2, j::2, ch] = 1
    else:
        raise ValueError(f"Unknown CFA type: {cfa_type}")

    return cfa.sum(axis=2), sparse_mask

def to_dng(uint_img, rh, filepath, ccm1, save_cfa=True, convert_to_cfa=True, use_orig_wb_points=False):
    width = uint_img.shape[1]
    height = uint_img.shape[0]
    bpp = 16 

    t = DNGTags()

    if save_cfa:
      if convert_to_cfa:
        cfa, _ = simulate_CFA(uint_img, pattern="RGGB", cfa_type="bayer")
        uint_img = cfa.astype(np.uint16)
      t.set(Tag.BitsPerSample, bpp)
      t.set(Tag.SamplesPerPixel, 1) 
      t.set(Tag.PhotometricInterpretation, PhotometricInterpretation.Color_Filter_Array)
      t.set(Tag.CFARepeatPatternDim, [2,2])
      t.set(Tag.CFAPattern, CFAPattern.RGGB)
      t.set(Tag.BlackLevelRepeatDim, [2,2])
      # This should not be used except to save testing patches
      if use_orig_wb_points:
        bl = rh.core_metadata.black_level_per_channel
        t.set(Tag.BlackLevel, bl)
        t.set(Tag.WhiteLevel, rh.core_metadata.white_level)
      else:
        t.set(Tag.BlackLevel, [0, 0, 0, 0])
        t.set(Tag.WhiteLevel, 65535)
    else:
      t.set(Tag.BitsPerSample, [bpp, bpp, bpp]) # 3 channels for RGB
      t.set(Tag.SamplesPerPixel, 3) # 3 for RGB
      t.set(Tag.PhotometricInterpretation, PhotometricInterpretation.Linear_Raw)
      t.set(Tag.BlackLevel,[0,0,0])
      t.set(Tag.WhiteLevel, [65535, 65535, 65535])

    t.set(Tag.ImageWidth, width)
    t.set(Tag.ImageLength, height)
    t.set(Tag.PlanarConfiguration, 1) # 1 for chunky (interleaved RGB)

    t.set(Tag.TileWidth, width)
    t.set(Tag.TileLength, height)

    t.set(Tag.ColorMatrix1, ccm1)
    t.set(Tag.CalibrationIlluminant1, CalibrationIlluminant.D65)
    wb = get_as_shot_neutral(rh)
    t.set(Tag.AsShotNeutral, wb)
    t.set(Tag.BaselineExposure, [[0,100]])


    try:
      t.set(Tag.Make, rh.full_metadata['Image Make'].values)
      t.set(Tag.Model, rh.full_metadata['Image Model'].values)
      # Lets Lightroom match its camera profile, otherwise colors come out generic (yellowish)
      t.set(Tag.UniqueCameraModel, str(rh.full_metadata['Image Model'].values))
      t.set(Tag.Orientation, rh.full_metadata['Image Orientation'].values[0])
      exposures = get_ratios('EXIF ExposureTime', rh)
      fnumber = get_ratios('EXIF FNumber', rh)
      ExposureBiasValue = get_ratios('EXIF ExposureBiasValue', rh) 
      FocalLength = get_ratios('EXIF FocalLength', rh) 
      t.set(Tag.FocalLength, FocalLength)
      t.set(Tag.EXIFPhotoLensModel, rh.full_metadata['EXIF LensModel'].values)
      t.set(Tag.ExposureBiasValue, ExposureBiasValue)
      t.set(Tag.ExposureTime, exposures)
      t.set(Tag.FNumber, fnumber)
      t.set(Tag.PhotographicSensitivity, rh.full_metadata['EXIF ISOSpeedRatings'].values)
    except:
      print("Could not save EXIF")
    # PiDNG already writes DNGVersion / DNGBackwardVersion; setting them again duplicates the tags
    t.set(Tag.PreviewColorSpace, PreviewColorSpace.Adobe_RGB)

    r = RAW2DNG()

    r.options(t, path="", compress=False)

    r.convert(uint_img, filename=filepath)



def convert_color_matrix(matrix):
  """
  Converts a 3x3 NumPy matrix of floats into a list of integer pairs.

  Each float value in the matrix is converted to a fractional representation
  with a denominator of 10000. The numerator is calculated by scaling the
  float value by 10000 and rounding to the nearest integer.

  Args:
    matrix: A 3x3 NumPy array with floating-point numbers.

  Returns:
    A list of 9 lists, where each inner list contains two integers
    representing the numerator and denominator.
  """
  # Ensure the input is a NumPy array
  if not isinstance(matrix, np.ndarray):
    raise TypeError("Input must be a NumPy array.")

  # Flatten the 3x3 matrix into a 1D array of 9 elements
  flattened_matrix = matrix.flatten()

  # Initialize the list for the converted matrix
  converted_list = []
  denominator = 10000

  # Iterate over each element in the flattened matrix
  for element in flattened_matrix:
    # Scale the element, round it to the nearest integer, and cast to int
    numerator = int(round(element * denominator))
    # Append the [numerator, denominator] pair to the result list
    converted_list.append([numerator, denominator])

  return converted_list