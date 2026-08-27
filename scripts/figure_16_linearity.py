# %%

import h5py
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from biolabval.utils.plotting import compact_square_layout
from biolabval.numerical_twin import build_numerical_twin
from biolabval.utils.config import Config

config = Config()

# %% Parameters

# optical setup
grey_width_diameter = 5.51  # Bi-O edge full grey width [lambda/D]

# simulation sampling
sim_zero_padding_factor = 20
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

# %% Import data from linearity file and corresponding interaction matrix file

utc_linearity = "utc_2026-03-12_15-02-45"

with h5py.File(
    config.paths.data_dir / "linearity" / (utc_linearity + "_linearity.h5")
) as f:
    linearity_grp = f["linearity_group"]
    reconstructor_group = f["reconstructor_grp"]
    interaction_matrix_grp = reconstructor_group["interaction_matrix_grp"]
    injected_amplitudes_rad = linearity_grp["injected_amplitudes"][...]
    reconstructed_amplitudes_rad_bioedge_exp = linearity_grp[
        "reconstructed_amplitudes"
    ][...]
    interaction_matrix_exp = interaction_matrix_grp["interaction_matrix"][...]
    strokes_exp = interaction_matrix_grp["interaction_matrix"].attrs["strokes"]
    modes_limits_exp = interaction_matrix_grp["interaction_matrix"].attrs[
        "modes_limits"
    ]
    valid_pixels_exp = interaction_matrix_grp["valid_pixels"][...]
    modal_basis_exp = interaction_matrix_grp["modal_basis"][...]
    reference_intensities_exp = interaction_matrix_grp["reference_intensities"][...]

# measured modes indices
modes_numbers = [0, 1, 2, 10, 50, 150, 300]

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
    orthogonalize_modal_basis=True,
)

# %% Unpack numerical twins

src = numerical_twin_real_hwp.src
tel = numerical_twin_real_hwp.tel
wfs = numerical_twin_real_hwp.wfs
dm = numerical_twin_real_hwp.dm
interaction_matrix_sim = numerical_twin_real_hwp.interaction_matrix_sim
modal_basis_sim = numerical_twin_real_hwp.modal_basis_sim
mis_reg = numerical_twin_real_hwp.mis_reg

# %% Simulation

# compute reconstructor
reconstructor_sim = np.linalg.pinv(
    interaction_matrix_sim * (2 * np.pi) / src.wavelength
)  # reconstructor in [m] to match OOPAO convention

injected_amplitudes_meter = (
    injected_amplitudes_rad * src.wavelength / (2 * np.pi)
)  # [m]

reconstructed_amplitudes_rad_bioedge_sim = np.full(
    (
        len(modes_numbers),
        injected_amplitudes_meter.shape[0],
        reconstructor_sim.shape[0],
    ),
    np.nan,
)

for mode_index, n_mode in enumerate(tqdm(modes_numbers, desc="Modes")):
    for amplitude_index, amplitude_meter in enumerate(
        tqdm(injected_amplitudes_meter, desc="Amplitudes", leave=False)
    ):
        coefs = np.zeros(dm.nValidAct)
        coefs[n_mode] = amplitude_meter
        dm.coefs = coefs
        src**tel * dm * wfs
        reconstructed_amplitudes_rad_bioedge_sim[mode_index, amplitude_index, :] = (
            2 * np.pi / src.wavelength * (reconstructor_sim @ wfs.signal)
        )

# %% Linearity plot

# Define grid size
n_cols, n_rows = compact_square_layout(len(modes_numbers) - 1)

fig_linearity, axs = plt.subplots(
    nrows=n_rows,
    ncols=n_cols,
    figsize=(width_single_column, 1.4 * width_single_column),
    constrained_layout=True,
    sharex=True,
    sharey=True,
)
axs = axs.flatten()  # flatten to 1D array for easy indexing

for mode_index, n_mode in enumerate(modes_numbers):

    plot_index = mode_index

    if mode_index == 1:
        continue  # skip mode 1
    if mode_index > 1:
        plot_index -= 1  # adjust index after skipping

    axs[plot_index].plot(
        injected_amplitudes_rad,
        reconstructed_amplitudes_rad_bioedge_exp[mode_index, :, n_mode],
        label="experimental",
        linestyle="-",
        linewidth=linewidth,
        # marker="+",
    )

    axs[plot_index].plot(
        injected_amplitudes_rad,
        reconstructed_amplitudes_rad_bioedge_sim[mode_index, :, n_mode],
        label="simulation",
        linestyle="-.",
        linewidth=linewidth,
        # marker="*",
    )

    axs[plot_index].plot(
        injected_amplitudes_rad,
        injected_amplitudes_rad,
        label="y=x",
        color="k",
        linestyle="--",
        zorder=-1,
        linewidth=0.8 * linewidth,
    )
    axs[plot_index].axvline(0, color="gray", linestyle="-", linewidth=0.5 * linewidth)
    axs[plot_index].axhline(0, color="gray", linestyle="-", linewidth=0.5 * linewidth)

    text = f"KL mode {n_mode}"
    if n_mode == 0:
        text = "Tip mode"

    axs[plot_index].set_title(text, fontsize=fontsize_title)

axs[0].legend(fontsize=fontsize_legend, loc="upper left")

for ax in axs.flat:
    ax.tick_params(axis="both", labelsize=fontsize_tick)

# Turn off any unused subplots
for ax in axs[len(modes_numbers) - 1 :]:
    ax.axis("off")

fig_linearity.supxlabel(
    "Input mode standard deviation [rad]",
    x=0.57,
    fontsize=fontsize_label,
)
fig_linearity.supylabel(
    "Reconstructed mode standard deviation [rad]", fontsize=fontsize_label
)

fig_name = f"figure_16_linearity.pdf"

fig_linearity.savefig(
    fig_dir / fig_name,
    bbox_inches="tight",
    pad_inches=0.01,
)

# %%

print(f"Run completed successfully\nFigures saved in {fig_dir / fig_name}")
