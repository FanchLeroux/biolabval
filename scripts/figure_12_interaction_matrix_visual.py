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
utc_imat = "utc_2026-06-24_14-11-43"

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
    modal_basis_exp = interaction_matrix_grp["modal_basis"][...]
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

# %% plot visual comparison

pupil_exp = get_circular_pupil(modal_basis_exp.shape[1])
valid_pixels_sim = numerical_twin_real_hwp.wfs.valid_signal_2D

modes = [0, 5, 160]

support_modal_basis = np.full(pupil_exp.shape, np.nan)

support_exp = np.full(valid_pixels_exp.shape, np.nan)
support_sim = np.full(valid_pixels_sim.shape, np.nan)

empty_rows_exp = ~np.any(valid_pixels_exp, axis=1)
empty_cols_exp = ~np.any(valid_pixels_exp, axis=0)
empty_rows_sim = ~np.any(valid_pixels_sim, axis=1)
empty_cols_sim = ~np.any(valid_pixels_sim, axis=0)

fig_imat_visual, axs = plt.subplots(
    nrows=len(modes),
    ncols=3,
    figsize=(width_single_column, 1.1 * len(modes) / 2 * width_single_column),
    constrained_layout=True,
)

# ---- column titles (once) ----
axs[0, 1].set_title("Experimental\n", fontsize=fontsize_big_title)
axs[0, 2].set_title("Simulation\n", fontsize=fontsize_big_title)

for i, mode in enumerate(modes):

    support_modal_basis[pupil_exp] = modal_basis_exp[mode, pupil_exp]

    support_exp[valid_pixels_exp] = interaction_matrix_exp[:, mode]
    support_sim[valid_pixels_sim] = interaction_matrix_sim[:, mode]

    vmin = support_exp[valid_pixels_exp].min()
    vmax = support_exp[valid_pixels_exp].max()

    # ---- mode index (text only) ----
    axs[i, 0].imshow(support_modal_basis, cmap="plasma")
    axs[i, 0].axis("off")
    if mode == 0:
        text = "Tip mode"
    else:
        text = f"KL mode {mode}"
    axs[i, 0].text(
        0.5,
        1.1,
        text,
        ha="center",
        va="center",
        fontsize=fontsize_title,
        transform=axs[i, 0].transAxes,
    )

    # ---- experimental ----
    im_exp = axs[i, 1].imshow(
        np.delete(
            np.delete(support_exp, empty_rows_exp, axis=0), empty_cols_exp, axis=1
        ),
        vmin=vmin,
        vmax=vmax,
    )
    axs[i, 1].axis("off")

    # ---- simulation ----
    im_sim = axs[i, 2].imshow(
        np.delete(
            np.delete(support_sim, empty_rows_sim, axis=0), empty_cols_sim, axis=1
        ),
        vmin=vmin,
        vmax=vmax,
    )
    axs[i, 2].axis("off")

    cbar_imat = fig_imat_visual.colorbar(
        im_exp, ax=[axs[i, 1], axs[i, 2]], aspect=30, shrink=0.95, location="bottom"
    )
    cbar_imat.ax.tick_params(axis="x", direction="out", labelsize=fontsize_tick)
    cbar_imat.ax.xaxis.get_offset_text().set_fontsize(fontsize_tick)

fig_name = f"figure_12_interaction_matrix_visual.pdf"

fig_imat_visual.savefig(
    fig_dir / fig_name,
    bbox_inches="tight",
    pad_inches=0.01,
)

# %%

print(f"Run completed successfully.\nFigure saved in: {fig_dir / fig_name}")
