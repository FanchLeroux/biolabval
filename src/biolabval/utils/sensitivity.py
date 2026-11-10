import numpy as np


def compute_photon_noise_sensitivity(
    interaction_matrix: np.ndarray, reference_signal: np.ndarray
) -> np.ndarray:
    """
    author: fleroux
    interaction_matrix: 2D array (n_valid_pixels, n_modes) Should have been normalized by the stroke in [rad]
    reference_signal: 1D array (n_valid_pixels) Should have been normalized by the total flux
    """
    return (
        np.diag(
            (interaction_matrix / reference_signal.reshape((-1, 1)) ** 0.5).T
            @ (interaction_matrix / reference_signal.reshape((-1, 1)) ** 0.5)
        )
        ** 0.5
    )


def compute_readout_noise_sensitivity(
    interaction_matrix: np.ndarray, n_subapertures: int
) -> np.ndarray:
    """
    author: fleroux
    interaction_matrix: 2D array (n_valid_pixels, n_modes) Should have been normalized by the stroke in [rad]
    n_subapertures: int Number of subapertures
    """
    return n_subapertures**0.5 * (
        np.diag((interaction_matrix).T @ (interaction_matrix)) ** 0.5
    )
