"""
Functions to compute modal basis (Fourier, KL, Zernike)
"""

# %% Imports

import numpy as np

# %% Compute Fourier Basis


def compute_fourier_mode(
    n_pixels: int, nu_x: int, nu_y: int, kind="cos", pupil_mask=None
):
    coords = np.arange(-n_pixels // 2, n_pixels // 2)
    X, Y = np.meshgrid(coords, coords)

    phase = 2 * np.pi * (nu_x * X + nu_y * Y) / n_pixels

    if kind == "sin":
        mode = np.sin(phase)
    elif kind == "cos":
        mode = np.cos(phase)
    else:
        raise ValueError("kind must be 'sin' or 'cos'")

    if pupil_mask is None:
        pupil_mask = np.ones(mode.shape, dtype=bool)

    mode *= pupil_mask

    mode[pupil_mask] -= mode[pupil_mask].mean()
    mode[pupil_mask] /= mode[pupil_mask].std()

    return mode


def compute_fourier_basis(n_pixels: int, remove_piston: bool = False):
    basis = []
    labels = []

    half = n_pixels // 2
    nyquist = n_pixels % 2 == 0

    freq_x = np.arange(-half + 1, half + 1)
    freq_y = np.arange(0, half + 1)

    for nu_x in freq_x:
        for nu_y in freq_y:

            # -------------------
            # Special cases
            # -------------------
            special = (
                (nu_x == 0 and nu_y == 0)
                or (nyquist and nu_x == 0 and nu_y == half)
                or (nyquist and nu_x == half and nu_y == 0)
                or (nyquist and nu_x == half and nu_y == half)
            )

            if special:
                if nu_x == 0 and nu_y == 0:
                    if not remove_piston:
                        piston = np.ones((n_pixels, n_pixels))
                        piston /= np.sqrt(np.mean(piston**2))
                        basis.append(piston)
                        labels.append((0, 0, "piston"))
                else:
                    cos_mode = compute_fourier_mode(n_pixels, nu_x, nu_y, "cos")
                    basis.append(cos_mode)
                    labels.append((nu_x, nu_y, "cos"))

                continue

            # -------------------
            # General case
            # -------------------
            if (nu_y != 0 or nu_x >= 0) and (nu_y != half or nu_x >= 0):

                cos_mode = compute_fourier_mode(n_pixels, nu_x, nu_y, "cos")
                sin_mode = compute_fourier_mode(n_pixels, nu_x, nu_y, "sin")

                basis.append(cos_mode)
                labels.append((nu_x, nu_y, "cos"))

                basis.append(sin_mode)
                labels.append((nu_x, nu_y, "sin"))

    basis = np.array(basis)

    expected = n_pixels**2 if not remove_piston else n_pixels**2 - 1

    if basis.shape[0] != expected:
        raise RuntimeError(
            f"Basis incomplete: got {basis.shape[0]}, expected {expected}"
        )

    return basis, labels


# def compute_fourier_mode(
#     n_pixels: int,
#     nu_x,
#     nu_y,
# ):

#     [X, Y] = np.meshgrid(
#         np.arange(-n_pixels // 2 + 1, n_pixels // 2 + 1),
#         np.flip(np.arange(-n_pixels // 2 + 1, n_pixels // 2 + 1)),
#     )

#     if (
#         (nu_x == 0 and nu_y == 0)
#         or (nu_x == 0 and nu_y == n_pixels // 2)
#         or (nu_x == n_pixels // 2 and nu_y == n_pixels // 2)
#         or (nu_x == n_pixels // 2 and nu_y == 0)
#     ):

#         fourier_mode = np.sin(2.0 * np.pi / X.shape[0] * (nu_x * X + nu_y * Y))

#     elif (nu_y != 0 or nu_x >= 0) and (nu_y != n_pixels // 2 or nu_x >= 0):
#         fourier_mode = np.sin(2.0 * np.pi / X.shape[0] * (nu_x * X + nu_y * Y))

#     return fourier_mode / fourier_mode.std()


def sort_real_fourier_basis(basis_map):

    n_pixels = basis_map.shape[2]
    my_list = []

    for nu_x in np.arange(-n_pixels // 2 + 1, n_pixels // 2 + 1):
        for nu_y in np.arange(0, n_pixels // 2 + 1):
            if (
                (nu_x == 0 and nu_y == 0)
                or (nu_x == 0 and nu_y == n_pixels // 2)
                or (nu_x == n_pixels // 2 and nu_y == n_pixels // 2)
                or (nu_x == n_pixels // 2 and nu_y == 0)
            ):
                my_list.append(
                    [
                        (nu_x**2 + nu_y**2) ** 0.5,
                        basis_map[:, :, nu_x, nu_y, 0].tolist(),
                    ]
                )
            elif (nu_y != 0 or nu_x >= 0) and (nu_y != n_pixels // 2 or nu_x >= 0):
                my_list.append(
                    [
                        (nu_x**2 + nu_y**2) ** 0.5,
                        basis_map[:, :, nu_x, nu_y, 0].tolist(),
                    ]
                )
                my_list.append(
                    [
                        (nu_x**2 + nu_y**2) ** 0.5,
                        basis_map[:, :, nu_x, nu_y, 1].tolist(),
                    ]
                )

    my_list.sort()

    basis = np.empty((basis_map.shape[0], basis_map.shape[1], len(my_list)))
    basis.fill(np.nan)

    for k in range(len(my_list)):
        basis[:, :, k] = np.array(my_list[k][1])

    return basis


def compute_real_fourier_basis(
    n_pixels: int, *, return_map=False, n_subapertures=False
):

    if n_subapertures:
        n_pixels2 = n_subapertures
    else:
        n_pixels2 = n_pixels

    basis_map = np.empty(
        (n_pixels, n_pixels, n_pixels2, n_pixels2 // 2 + 1, 2), dtype=float
    )
    basis_map.fill(np.nan)

    [X, Y] = np.meshgrid(
        np.arange(-n_pixels // 2 + 1, n_pixels // 2 + 1),
        np.flip(np.arange(-n_pixels // 2 + 1, n_pixels // 2 + 1)),
    )

    for nu_x in np.arange(-n_pixels2 // 2 + 1, n_pixels2 // 2 + 1):
        for nu_y in np.arange(0, n_pixels2 // 2 + 1):

            if (
                (nu_x == 0 and nu_y == 0)
                or (nu_x == 0 and nu_y == n_pixels2 // 2)
                or (nu_x == n_pixels2 // 2 and nu_y == n_pixels2 // 2)
                or (nu_x == n_pixels2 // 2 and nu_y == 0)
            ):

                basis_map[:, :, nu_x, nu_y, 0] = (
                    1.0
                    / X.shape[0]
                    * np.cos(2.0 * np.pi / X.shape[0] * (nu_x * X + nu_y * Y))
                )

            elif (nu_y != 0 or nu_x >= 0) and (nu_y != n_pixels2 // 2 or nu_x >= 0):

                basis_map[:, :, nu_x, nu_y, 0] = (
                    2**0.5
                    / X.shape[0]
                    * np.cos(2.0 * np.pi / X.shape[0] * (nu_x * X + nu_y * Y))
                )
                basis_map[:, :, nu_x, nu_y, 1] = (
                    2**0.5
                    / X.shape[0]
                    * np.sin(2.0 * np.pi / X.shape[0] * (nu_x * X + nu_y * Y))
                )

    if return_map:
        return basis_map
    else:
        return np.transpose(
            sort_real_fourier_basis(basis_map), (2, 0, 1)
        )  # reshape as [n_modes, n_pixels, n_pixels]


def extract_subset(complete_real_fourier_basis, new_n_pixels):
    return np.roll(
        complete_real_fourier_basis[
            :,
            :,
            np.arange(-new_n_pixels // 2 + 1, new_n_pixels // 2 + 1),
            0 : new_n_pixels // 2 + 1,
            :,
        ],
        -new_n_pixels // 2 + 1,
        axis=2,
    )


def extract_horizontal_frequencies(basis_map):

    horizontal_frequencies = basis_map[:, :, : basis_map.shape[2] // 2 + 1, 0, 0]
    return horizontal_frequencies


def extract_vertical_frequencies(basis_map):

    vertical_frequencies = basis_map[:, :, 0, :, 0]
    return vertical_frequencies


def extract_diagonal_frequencies(basis_map, complete=0):

    diagonal_frequencies = np.empty(
        (basis_map.shape[0], basis_map.shape[1], basis_map.shape[3])
    )
    diagonal_frequencies.fill(np.nan)

    for k in range(basis_map.shape[3]):
        diagonal_frequencies[:, :, k] = basis_map[:, :, k, k, 0]

    if complete:
        diagonal_frequencies = np.empty(
            (basis_map.shape[0], basis_map.shape[1], 2 * basis_map.shape[3] - 2)
        )
        diagonal_frequencies.fill(np.nan)
        index = 0
        for k in range(basis_map.shape[3]):
            diagonal_frequencies[:, :, index] = basis_map[:, :, k, k, 0]
            if k != basis_map.shape[3] - 1 and k != 0:
                diagonal_frequencies[:, :, index + 1] = basis_map[:, :, k, k, 1]
                index += 2
            else:
                index += 1

    return np.array(diagonal_frequencies)


# %% Zernike modes
