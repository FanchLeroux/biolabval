# %%

import pathlib
import h5py
import matplotlib.pyplot as plt
import numpy as np

from biolabval.numerical_twin import build_numerical_twin
from biolabval.utils.sensitivity import (
    compute_photon_noise_sensitivity,
    compute_readout_noise_sensitivity,
)
from biolabval.utils.config import Config

# %% Parameters

config = Config()

utc_imat = "utc_2026-06-24_14-11-43"  # KL modes imat

# optical setup
grey_width_diameter = 5.51  # Bi-O edge full grey width [lambda/D]

# simulation sampling
sim_zero_padding_factor = 4
sim_tel_resolution_factor = 2

# dirc figures
fig_dir = config.paths.root_dir / "outputs"

compute_misreg = True


thorcam_pixel_pitch = 3.45e-6  # [m]

# experimental mask imperfection
polarization_leakage_factor = 0.0497

# plotting parameters
width_single_column = 3.45  # [Inche]
width_double_column = 6.9  # [Inche]
fontsize_label = 10
fontsize_legend = 7.7
fontsize_title = 8
fontsize_big_title = 10
fontsize_tick = 10
linewidth = 1.1
transparency = 0.5

# %% Import data from interaction matrix file

with h5py.File(
    config.paths.data_dir / "interaction_matrix" / f"{utc_imat}_interaction_matrix.h5"
) as f:
    interaction_matrix_grp = f["interaction_matrix_grp"]
    interaction_matrix_exp = interaction_matrix_grp["interaction_matrix"][...]
    strokes_exp = interaction_matrix_grp["interaction_matrix"].attrs["strokes"]
    modes_limits_exp = interaction_matrix_grp["interaction_matrix"].attrs[
        "modes_limits"
    ]
    valid_pixels_exp = interaction_matrix_grp["valid_pixels"][...]
    modal_basis_exp = interaction_matrix_grp["modal_basis"][...]
    reference_intensities_exp = interaction_matrix_grp["reference_intensities"][...]

# %% Build numerical twins

numerical_twin_ideal_hwp = build_numerical_twin(
    grey_width=grey_width_diameter / 2,
    valid_pixels_exp=valid_pixels_exp,
    modal_basis_exp=modal_basis_exp,
    interaction_matrix_exp=interaction_matrix_exp,
    stroke_rad=strokes_exp.min(),
    polarization_leakage_factor=0.0,
    sim_zero_padding_factor=sim_zero_padding_factor,
    sim_tel_resolution_factor=sim_tel_resolution_factor,
    compute_misreg=compute_misreg,
)

numerical_twin_real_hwp = build_numerical_twin(
    grey_width=grey_width_diameter / 2,
    valid_pixels_exp=valid_pixels_exp,
    modal_basis_exp=modal_basis_exp,
    interaction_matrix_exp=interaction_matrix_exp,
    stroke_rad=strokes_exp.min(),
    polarization_leakage_factor=polarization_leakage_factor,
    sim_zero_padding_factor=sim_zero_padding_factor,
    sim_tel_resolution_factor=sim_tel_resolution_factor,
    compute_misreg=compute_misreg,
)

# %% Unpack numerical twins

interaction_matrix_sim_ideal_hwp = numerical_twin_ideal_hwp.interaction_matrix_sim
reference_intensities_sim_ideal_hwp = numerical_twin_ideal_hwp.reference_intensities_sim

interaction_matrix_sim_real_hwp = numerical_twin_real_hwp.interaction_matrix_sim
reference_intensities_sim_real_hwp = numerical_twin_real_hwp.reference_intensities_sim

# %% Compute sensitivities

photon_noise_sensitivity_exp = compute_photon_noise_sensitivity(
    interaction_matrix_exp,
    reference_intensities_exp,
)
readout_noise_sensitivity_exp = compute_readout_noise_sensitivity(
    interaction_matrix_exp,
    n_subapertures=interaction_matrix_exp.shape[0] // 4,  # 4 pupils
)

photon_noise_sensitivity_sim_ideal_hwp = compute_photon_noise_sensitivity(
    interaction_matrix_sim_ideal_hwp, reference_intensities_sim_ideal_hwp
)
readout_noise_sensitivity_sim_ideal_hwp = compute_readout_noise_sensitivity(
    interaction_matrix_sim_ideal_hwp,
    n_subapertures=interaction_matrix_sim_ideal_hwp.shape[0] // 4,  # 4 pupils
)

photon_noise_sensitivity_sim = compute_photon_noise_sensitivity(
    interaction_matrix_sim_real_hwp, reference_intensities_sim_real_hwp
)
readout_noise_sensitivity_sim = compute_readout_noise_sensitivity(
    interaction_matrix_sim_real_hwp,
    n_subapertures=interaction_matrix_sim_real_hwp.shape[0] // 4,  # 4 pupils
)

# %% Plot sensitivities - non ideal hwp vs ideal hwp vs experimental

nrows, ncols = 2, 1
fig_sensitivities, axs = plt.subplots(
    nrows=nrows,
    ncols=ncols,
    sharex=True,
    figsize=(width_single_column, 1.5 * width_single_column),
    constrained_layout=True,
)
axs = np.atleast_1d(axs).ravel()

# ideal hwp
axs[0].plot(
    [0] + photon_noise_sensitivity_exp.tolist(),
    label="experimental",
    linewidth=linewidth,
)
axs[0].plot(
    [0] + photon_noise_sensitivity_sim_ideal_hwp.tolist(),
    label="simulation,\nideal HWP",
    linestyle="-.",
    linewidth=linewidth,
    color="#ff7f0e",
    alpha=transparency,
)
axs[1].plot(
    [0] + readout_noise_sensitivity_exp.tolist(),
    label="experimental",
    linewidth=linewidth,
)
axs[1].plot(
    [0] + readout_noise_sensitivity_sim_ideal_hwp.tolist(),
    label="simulation,\nideal HWP",
    linestyle="-.",
    linewidth=linewidth,
    color="#ff7f0e",
    alpha=transparency,
)

# non ideal
axs[0].plot(
    [0] + photon_noise_sensitivity_sim.tolist(),
    label=f"simulation,\n{polarization_leakage_factor:.2%} of polarization leakage",
    linestyle="-.",
    linewidth=linewidth,
    color="#ff7f0e",
)
axs[1].plot(
    [0] + readout_noise_sensitivity_sim.tolist(),
    label=f"simulation,\n{polarization_leakage_factor:.2%} of polarization leakage",
    linestyle="-.",
    linewidth=linewidth,
    color="#ff7f0e",
)
axs[0].axhline(
    2**0.5,
    color="k",
    linestyle="--",
    label=r"$\sqrt{2}$",
    linewidth=0.8 * linewidth,
)
axs[0].set_ylim(0, 1.5)
axs[0].set_ylabel(r"$S_{ph}$", fontsize=fontsize_label)
axs[0].legend(loc="lower right", fontsize=fontsize_legend)

axs[1].set_ylabel(r"$S_{ron}$", fontsize=fontsize_label)
axs[1].legend(loc="lower right", fontsize=fontsize_legend)

fig_sensitivities.supxlabel("# KL mode", fontsize=fontsize_label, x=0.6)

for ax in axs:
    ax.tick_params(axis="both", labelsize=fontsize_tick)

fig_name = f"figure_15_KL_modes_sensitivity.pdf"

fig_sensitivities.savefig(
    fig_dir / fig_name,
    bbox_inches="tight",
    pad_inches=0.01,
)

# %%

print(f"Run completed successfully.\nFigure saved in: {fig_dir / fig_name}")
