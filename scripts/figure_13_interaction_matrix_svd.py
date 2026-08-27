# %%

import pathlib
import h5py
import matplotlib.pyplot as plt
import numpy as np

from biolabval.utils.pattern import get_circular_pupil
from biolabval.numerical_twin import build_numerical_twin
from biolabval.utils.config import Config

# %% Parameters

config = Config()

# data utc
utc_imat = "utc_2026-06-24_14-11-43"  # "utc_2026-03-25_08-55-54"

# optical setup
grey_width_diameter = 5.51  # Bi-O edge full grey width [lambda/D]

# simulation sampling
sim_zero_padding_factor = 10
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
fontsize_legend = 8
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
    modal_basis_exp = -interaction_matrix_grp["modal_basis"][...]
    reference_intensities_exp = interaction_matrix_grp["reference_intensities"][...]

# %% Build numerical twin

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

# %% Unpack numerical twin

interaction_matrix_sim = numerical_twin_real_hwp.interaction_matrix_sim

# %% SVD

n_controlled_modes = 342
u_exp, s_exp, vt_exp = np.linalg.svd(
    interaction_matrix_exp[:, :n_controlled_modes], full_matrices=False
)
u_sim, s_sim, vt_sim = np.linalg.svd(
    interaction_matrix_sim[:, :n_controlled_modes], full_matrices=False
)

# %%

fig_svd = plt.figure(
    figsize=(width_single_column, 0.7 * width_single_column), constrained_layout=True
)
plt.plot(
    s_exp,
    label=f"experimental - condition number: {(s_exp[0]/s_exp[-1]):.2f}",
    linewidth=linewidth,
)
plt.plot(
    s_sim,
    label=f"simulated - condition number: {(s_sim[0]/s_sim[-1]):.2f}",
    linestyle="-.",
    linewidth=linewidth,
)
plt.yscale("log")
plt.legend(loc="lower left", fontsize=7)
plt.xlabel("# Eigen mode", fontsize=fontsize_label)
plt.ylabel("Eigenvalue", fontsize=fontsize_label)
plt.tick_params(axis="both", labelsize=fontsize_tick)

fig_name = f"figure_13_interaction_matrix_svd.pdf"

fig_svd.savefig(
    fig_dir / fig_name,
    bbox_inches="tight",
    pad_inches=0.01,
)

# %%

print(f"Run completed successfully.\nFigure saved in: {fig_dir / fig_name}")
