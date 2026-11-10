# %%

import h5py
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from biolabval.utils import oopao_config
from biolabval.utils.config import Config

from biolabval.utils.plotting import compact_square_layout
from biolabval.numerical_twin import build_numerical_twin

# %% Parameters

config = Config()

# optical setup
grey_width_diameters = [5.51, 4.51]  # Bi-O edge full grey width [lambda/D]

# simulation sampling
sim_zero_padding_factor = 16
sim_tel_resolution_factor = 2

# dirc figures
fig_dir = config.paths.root_dir / "outputs"

compute_misreg = False

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

# %% Linearity test

# Build numerical twin

injected_amplitudes_meter = injected_amplitudes_rad * 640e-9 / (2 * np.pi)  # [m]

reconstructed_amplitudes_rad_bioedge_sim = np.full(
    (
        len(grey_width_diameters),
        len(modes_numbers),
        injected_amplitudes_meter.shape[0],
        interaction_matrix_exp.shape[1],
    ),
    np.nan,
)

for grey_width_diameter_index, grey_width_diameter in enumerate(grey_width_diameters):
    numerical_twin_real_hwp = build_numerical_twin(
        grey_width=grey_width_diameter / 2,
        valid_pixels_exp=valid_pixels_exp,
        modal_basis_exp=modal_basis_exp,
        interaction_matrix_exp=interaction_matrix_exp,
        stroke_rad=strokes_exp.min(),
        polarization_leakage_factor=polarization_leakage_factor,
        sim_zero_padding_factor=sim_zero_padding_factor,
        sim_tel_resolution_factor=sim_tel_resolution_factor,
        modulation=0,  # [lambda/D] modulation radius for the WFS, emulate vibrations
        compute_misreg=compute_misreg,
        orthogonalize_modal_basis=False,
        equalize_flux=False,
    )

    # Unpack numerical twins

    src = numerical_twin_real_hwp.src
    tel = numerical_twin_real_hwp.tel
    wfs = numerical_twin_real_hwp.wfs
    dm = numerical_twin_real_hwp.dm
    interaction_matrix_sim = numerical_twin_real_hwp.interaction_matrix_sim
    modal_basis_sim = numerical_twin_real_hwp.modal_basis_sim
    mis_reg = numerical_twin_real_hwp.mis_reg

    # Simulation

    # compute reconstructor
    reconstructor_sim = np.linalg.pinv(
        interaction_matrix_sim * (2 * np.pi) / src.wavelength
    )

    for mode_index, n_mode in enumerate(tqdm(modes_numbers, desc="Modes")):
        for amplitude_index, amplitude_meter in enumerate(
            tqdm(injected_amplitudes_meter, desc="Amplitudes", leave=False)
        ):
            coefs = np.zeros(dm.nValidAct)
            coefs[n_mode] = amplitude_meter
            dm.coefs = coefs
            src**tel * dm * wfs
            reconstructed_amplitudes_rad_bioedge_sim[
                grey_width_diameter_index, mode_index, amplitude_index, :
            ] = (2 * np.pi / src.wavelength * (reconstructor_sim @ wfs.signal))

# %% add curve to plot

# Define grid size
n_cols, n_rows = compact_square_layout(len(modes_numbers) - 1)

fig_linearity, axs = plt.subplots(
    nrows=n_rows,
    ncols=n_cols,
    figsize=(width_single_column, 0.7 * 3 * width_single_column),
    constrained_layout=True,
    sharex=True,
    sharey=True,
)
axs = axs.flatten()  # flatten to 1D array for easy indexing

for grey_width_diameter_index, grey_width_diameter in enumerate(grey_width_diameters):

    for mode_index, n_mode in enumerate(modes_numbers):

        plot_index = mode_index

        if mode_index == 1:
            continue  # skip mode 1
        if mode_index > 1:
            plot_index -= 1  # adjust index after skipping

        if grey_width_diameter_index == 0:
            axs[plot_index].plot(
                injected_amplitudes_rad,
                reconstructed_amplitudes_rad_bioedge_exp[mode_index, :, n_mode],
                label="experimental",
                linestyle="-",
                linewidth=linewidth,
                color="C0",
                # marker="+",
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

            axs[plot_index].axvline(
                0, color="gray", linestyle="-", linewidth=0.5 * linewidth
            )
            axs[plot_index].axhline(
                0, color="gray", linestyle="-", linewidth=0.5 * linewidth
            )

        axs[plot_index].plot(
            injected_amplitudes_rad,
            reconstructed_amplitudes_rad_bioedge_sim[
                grey_width_diameter_index, mode_index, :, n_mode
            ],
            label="simulation: " rf"grey width = {grey_width_diameter:.2f} $\lambda/D$",
            linestyle=":" if grey_width_diameter_index == 1 else "-.",
            linewidth=linewidth,
            color="red" if grey_width_diameter_index == 1 else "C1",
            alpha=1.0,
            # marker="*",
        )

        text = f"KL mode {n_mode}"
        if n_mode == 0:
            text = "Tip mode"

        axs[plot_index].set_title(text, fontsize=fontsize_title)

# Add legend to the first subplot

fig_linearity.set_size_inches(
    1.2 * width_single_column, 1.6 * width_single_column, forward=True
)

# axs[0].legend(fontsize=7, loc="upper center", bbox_to_anchor=(1.0, 1.9), frameon=True)

for ax in axs.flat:
    ax.set_xticks([-2, -1, 0, 1, 2])
    ax.set_yticks([-2, -1, 0, 1, 2])
    ax.tick_params(axis="both", labelsize=fontsize_tick)

# Turn off any unused subplots
for ax in axs[len(modes_numbers) - 1 :]:
    ax.axis("off")

fig_linearity.supxlabel(
    "Input modal coeficient [rad RMS]",
    x=0.57,
    fontsize=fontsize_label,
)
fig_linearity.supylabel(
    "Reconstructed modal coeficient [rad RMS]", fontsize=fontsize_label
)

fig_name = f"figure_16_linearity.pdf"

fig_linearity.savefig(
    fig_dir / fig_name,
    bbox_inches="tight",
    pad_inches=0.01,
)

plt.show()

# %%

print(f"Run completed successfully\nFigures saved in {fig_dir / fig_name}")

# %%
