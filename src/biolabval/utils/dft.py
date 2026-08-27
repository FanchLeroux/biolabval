import numpy as np
from numpy import typing as npt


def get_fft_grid(
    shape: tuple[int, ...], *, center=None, dtype=np.float64
) -> tuple[np.ndarray, ...]:
    """
    author: François Leroux
    Return fftshifted integer frequency grids for arbitrary dimensions.

    Parameters
    ----------
    shape : tuple[int, ...]
        Shape of the output grids.
    center : tuple[int, ...] | None
        Center of the grid. If None, the center is set to (0, ...).

    Returns
    -------
    grids : tuple[np.ndarray, ...]
        Tuple of arrays representing the frequency coordinates along each dimension.

    Notes
    -----
    Each returned array has shape `shape`.
    If center is None zero frequency is aligned with fftshifted FFT output.
    If center is provided, the returned grids are shifted so that the specified center is aligned with the null frequency.
    Center coordinates are specified as folows: (shift along dim 0, shift along dim 1, ...), where each shift is in units of pixels.
    Shifts can be positives (down, right, ...) or negatives (up, left, ...).
    For example, in 2D, center(0,0) corresponds to no shifts, and center(1, 1) corresponds to a shift of one pixel down, one pixel right
    """
    freq_axes = [np.fft.fftshift(np.fft.fftfreq(n)) * n for n in shape]

    grids = np.meshgrid(*freq_axes, indexing="ij")

    if center is not None:
        grids = [grid - c for grid, c in zip(grids, center)]

    return tuple(grid.astype(dtype) for grid in grids)


def pad_array(array: np.ndarray, factor: int) -> np.ndarray:
    """
    Pad an array with zeros on all sides by a given factor.
    Dimensions agnostic, supports 1D and 2D arrays.

    Parameters
    ----------
    array : ndarray
        Input array to pad.
    factor : int
        Factor by which to pad the array.
    Returns
    -------
    ndarray
        Padded array.
    """
    pad = [((factor - 1) * s // 2,) * 2 for s in array.shape]
    return np.pad(array, pad)


def crop_array(
    arr: npt.NDArray,
    new_shape: tuple[int, ...],
) -> npt.NDArray:
    """
    Crop an array around the null frequency (center) of an fftshifted FFT grid:
        [-N/2 ... -1 | 0 | 1 ... N/2-1]
    Dimensions agnostic, supports 1D and 2D arrays.

    Parameters
    ----------
    arr : ndarray
        Input array to crop.
    new_shape : tuple[int, ...] | int
        Desired output shape (must have same number of dimensions).
        If int is provided, the same value will be used for all dimensions.

    Returns
    -------
    ndarray
        Center-cropped array.
    """

    if isinstance(new_shape, int):
        new_shape = (new_shape,) * arr.ndim

    if arr.ndim != len(new_shape):
        raise ValueError("new_shape must have the same number of dimensions as arr")

    slices = []
    for size, new_size in zip(arr.shape, new_shape):

        if new_size > size:
            raise ValueError("new_shape must be smaller than arr.shape")

        center = size // 2
        start = center - new_size // 2
        stop = start + new_size

        slices.append(slice(start, stop))

    return arr[tuple(slices)]
