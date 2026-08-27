# %%

import pathlib
from copy import deepcopy
import h5py
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import FuncFormatter, NullFormatter
from matplotlib.patches import Rectangle

from biolabval.OOPAO.tools.interpolateGeometricalTransformation import interpolate_cube

from biolabval.utils.pattern import get_circular_pupil
from biolabval.utils.dft import pad_array
from biolabval.numerical_twin import build_numerical_twin
from biolabval.utils.closed_loop import close_the_loop
from biolabval.utils.config import Config

# %% Parameters

config = Config()

utc_closed_loop = "utc_2026-04-30_10-08-27"
closed_loop_filename = utc_closed_loop + "_closed_loop.h5"

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

# %%
n_iter = 1000

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

# %% Load data from closed loop file

with h5py.File(config.paths.data_dir / "closed_loop" / closed_loop_filename) as f:
    closed_loop_grp = f["closed_loop_grp"]
    reconstructor_grp = f["reconstructor_grp"]
    interaction_matrix_grp = reconstructor_grp["interaction_matrix_grp"]
    interaction_matrix_exp = interaction_matrix_grp["interaction_matrix"][...]
    strokes_exp = interaction_matrix_grp["interaction_matrix"].attrs["strokes"]
    n_controlled_modes = closed_loop_grp.attrs["n_controlled_modes"]
    valid_pixels_exp = interaction_matrix_grp["valid_pixels"][...]
    valid_pixels_exp_default = deepcopy(valid_pixels_exp)
    modal_basis_exp = interaction_matrix_grp["modal_basis"][...]
    turbulence = closed_loop_grp["turbulence"][:n_iter]  # [rad]
    wfs_frames_exp = closed_loop_grp["wfs_frames"][:31]
    total_exp = closed_loop_grp["total"][:n_iter]
    residual_exp = closed_loop_grp["residual"][:n_iter]
    turbulent_phases_modal_decomposition_exp = closed_loop_grp[
        "turbulent_phases_modal_decomposition"
    ][:n_iter]
    reconstructed_phases_modal_decomposition_exp = closed_loop_grp[
        "reconstructed_phases_modal_decomposition"
    ][:n_iter]
    residual_phases_modal_decomposition_exp = closed_loop_grp[
        "residual_phases_modal_decomposition"
    ][:n_iter]
    residual_phases_exp = closed_loop_grp["residual_phases"][:n_iter]
    frequency = closed_loop_grp["turbulence"].attrs["frequency"]
    r0 = closed_loop_grp["turbulence"].attrs["r0"]
    delay = closed_loop_grp.attrs["delay"]
    loop_gain = closed_loop_grp.attrs["gain"][()]
    reference_intensities_exp = interaction_matrix_grp["reference_intensities"][...]
    focal_plane_images_exp = closed_loop_grp["focal_plane_images"][:n_iter]
    focal_plane_images_open_loop_exp = f["open_loop_grp"]["focal_plane_images"][:n_iter]
    reference_psf_exp = f["closed_loop_grp"]["reference_psf"][...]

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

# %% interpolate turbulence

turbulence_sim = interpolate_cube(
    turbulence,
    pixel_size_in=tel.D / modal_basis_exp.shape[1],
    pixel_size_out=tel.D / tel.resolution,
    resolution_out=tel.resolution,
    shape_out=[tel.resolution, tel.resolution],
    mis_registration=mis_reg,
    fliplr=False,
    flipud=False,
)
turbulence_sim[:, ~tel.pupil] = 0

# adapt turbulent phase screens to bench convention (rotation and flip)
turbulence_sim = np.rot90(turbulence_sim, k=1, axes=(1, 2))
turbulence_sim = np.flip(turbulence_sim, axis=2)

# %% closed loop simulation

valid_pixels_sim = wfs.valid_signal_2D

# compute reconstructor
reconstructor_sim = np.linalg.pinv(
    interaction_matrix_sim * (2 * np.pi) / src.wavelength
)  # reconstructor in [m] to match OOPAO convention

closed_loop_data_sim = close_the_loop(
    src,
    tel,
    dm,
    wfs,
    modal_basis=modal_basis_sim,
    turbulence_phase_screens=turbulence_sim,
    valid_pixels=valid_pixels_sim,
    reconstructor=deepcopy(reconstructor_sim),
    loop_gain=loop_gain,
    delay=delay,
    n_iter=n_iter,
    n_controlled_modes=n_controlled_modes,
    reference_intensities=wfs.referenceSignal,
    display=False,
    save_telemetry=True,
)

# %% Numerical twin illustration

support_ol_wfs_exp = np.full(valid_pixels_exp_default.shape, np.nan)
support_ol_wfs_sim = np.full(valid_pixels_sim.shape, np.nan)

support_cl_wfs_exp = np.full(valid_pixels_exp_default.shape, np.nan)
support_cl_wfs_sim = np.full(valid_pixels_sim.shape, np.nan)

# remove empty rows/cols masks
empty_rows_exp = ~np.any(valid_pixels_exp_default, axis=1)
empty_cols_exp = ~np.any(valid_pixels_exp_default, axis=0)

empty_rows_sim = ~np.any(valid_pixels_sim, axis=1)
empty_cols_sim = ~np.any(valid_pixels_sim, axis=0)

iter_number = 29

# frames
ol_wfs_frame_exp = wfs_frames_exp[0]
ol_wfs_frame_sim = closed_loop_data_sim.wfs_frames[0]

cl_wfs_frame_exp = wfs_frames_exp[iter_number]
cl_wfs_frame_sim = closed_loop_data_sim.wfs_frames[iter_number]

# flatten valid pixels
ol_wfs_frame_exp_flat = ol_wfs_frame_exp[valid_pixels_exp_default]
ol_wfs_frame_sim_flat = ol_wfs_frame_sim[valid_pixels_sim]
ol_wfs_signal_exp_flat = (
    ol_wfs_frame_exp_flat / ol_wfs_frame_exp_flat.sum() - reference_intensities_exp
)

cl_wfs_frame_exp_flat = cl_wfs_frame_exp[valid_pixels_exp_default]
cl_wfs_frame_sim_flat = cl_wfs_frame_sim[valid_pixels_sim]
cl_wfs_signal_exp_flat = (
    cl_wfs_frame_exp_flat / cl_wfs_frame_exp_flat.sum() - reference_intensities_exp
)

# fill supports
support_ol_wfs_exp[valid_pixels_exp_default] = ol_wfs_signal_exp_flat
support_ol_wfs_sim[valid_pixels_sim] = closed_loop_data_sim.wfs_signals[0]

support_cl_wfs_exp[valid_pixels_exp_default] = cl_wfs_signal_exp_flat
support_cl_wfs_sim[valid_pixels_sim] = closed_loop_data_sim.wfs_signals[iter_number]

# plot
fig_twin, axs = plt.subplots(
    nrows=2,
    ncols=2,
    figsize=(width_single_column, 1.31 * width_single_column),
    constrained_layout=True,
)

# Row 0 (Iteration 0)
data_exp_0 = np.delete(
    np.delete(support_ol_wfs_exp, empty_rows_exp, axis=0),
    empty_cols_exp,
    axis=1,
)

data_sim_0 = np.delete(
    np.delete(support_ol_wfs_sim, empty_rows_sim, axis=0),
    empty_cols_sim,
    axis=1,
)

vmin_0 = np.nanmin([data_exp_0, data_sim_0])
vmax_0 = np.nanmax([data_exp_0, data_sim_0])

im0 = axs[0, 0].imshow(data_exp_0, vmin=vmin_0, vmax=vmax_0)
axs[0, 0].set_title("Experimental\n", fontsize=fontsize_big_title)
axs[0, 0].axis("off")

im1 = axs[0, 1].imshow(data_sim_0, vmin=vmin_0, vmax=vmax_0)
axs[0, 1].set_title("Simulation\n", fontsize=fontsize_big_title)
axs[0, 1].axis("off")

fig_twin.colorbar(
    im0,
    ax=[axs[0, 0], axs[0, 1]],
    orientation="horizontal",
    fraction=0.055,
    pad=0.04,
    aspect=40,
    shrink=0.95,
    location="bottom",
)

# Row 1 (Iteration 30)
data_exp_30 = np.delete(
    np.delete(support_cl_wfs_exp, empty_rows_exp, axis=0),
    empty_cols_exp,
    axis=1,
)

data_sim_30 = np.delete(
    np.delete(support_cl_wfs_sim, empty_rows_sim, axis=0),
    empty_cols_sim,
    axis=1,
)
vmin_30 = np.nanmin([data_exp_30, data_sim_30])
vmax_30 = np.nanmax([data_exp_30, data_sim_30])

im2 = axs[1, 0].imshow(data_exp_30, vmin=vmin_30, vmax=vmax_30)
axs[1, 0].axis("off")

im3 = axs[1, 1].imshow(data_sim_30, vmin=vmin_30, vmax=vmax_30)
axs[1, 1].axis("off")

fig_twin.colorbar(
    im2,
    ax=[axs[1, 0], axs[1, 1]],
    orientation="horizontal",
    fraction=0.055,
    pad=0.04,
    aspect=40,
    shrink=0.95,
    location="bottom",
)
fig_twin.subplots_adjust(wspace=0.000001)

fig_names = []
fig_name = f"figure_11_numerical_twin_illustration.pdf"
fig_names.append(fig_name)

fig_twin.savefig(
    fig_dir / fig_name,
    bbox_inches="tight",
    pad_inches=0.001,
)

# %% compute fitting error

pupil_sim = ~np.any(modal_basis_sim == 0, axis=0)
modal_basis_sim_flat = modal_basis_sim[:, pupil_sim].T
turbulence_sim_flat = turbulence_sim[:, pupil_sim].T
modal_projector_sim = np.linalg.pinv(modal_basis_sim_flat)

residual_turbulence_sim_flat = (
    turbulence_sim_flat
    - ((modal_projector_sim @ turbulence_sim_flat).T @ modal_basis_sim_flat.T).T
)

residual_turbulence_sim = np.full_like(turbulence_sim, np.nan)
residual_turbulence_sim[:, pupil_sim] = residual_turbulence_sim_flat.T

# %% figure residual std

fig_phase_std, axs = plt.subplots(
    2,
    1,
    figsize=(width_single_column, 1.5 * width_single_column),
    constrained_layout=True,
)
axs[0].plot(total_exp, label="open loop", color="grey")
axs[0].plot(residual_exp, label="experimental")
axs[0].plot(closed_loop_data_sim.residual, label="simulation")
axs[0].plot(
    residual_turbulence_sim_flat.std(axis=0),
    label="fitting error only",
    linewidth=linewidth,
)

limit = 30
axs[1].plot(total_exp[:limit], label="open loop", color="grey", linewidth=linewidth)
axs[1].plot(residual_exp[:limit], label="experimental", linewidth=linewidth)
axs[1].plot(
    closed_loop_data_sim.residual[:limit], label="simulation", linewidth=linewidth
)
axs[1].plot(
    residual_turbulence_sim_flat.std(axis=0)[:limit],
    label="fitting error only",
    linewidth=linewidth,
)
axs[1].set_ylim(0, 1.5)

x0, x1 = axs[1].get_xlim()
y0, y1 = axs[1].get_ylim()

rect = Rectangle(
    (x0, y0),
    x1 - x0,
    y1 - y0,
    fill=False,
    edgecolor="purple",
    linestyle="--",
    linewidth=linewidth * 1.5,
    zorder=10,
)

axs[0].add_patch(rect)

for spine in axs[1].spines.values():
    spine.set_color("purple")
    spine.set_linestyle((0, (8, 6)))
    spine.set_linewidth(linewidth * 1.5)

fig_phase_std.supxlabel("# Closed loop iteration", fontsize=fontsize_label, x=0.55)
fig_phase_std.supylabel(
    "Residual phase standard deviation [rad]", fontsize=fontsize_label, y=0.54
)

axs[1].legend(loc="upper right", fontsize=8)

fig_name = f"figure_18_residual.pdf"
fig_names.append(fig_name)

fig_phase_std.savefig(fig_dir / fig_name, bbox_inches="tight", pad_inches=0.01)

print(f"Mean residual ratio - experimental: {residual_exp.mean():.3f} [rad RMS]")
print(
    f"Mean residual ratio - simulation: {closed_loop_data_sim.residual[30:].mean():.3f} [rad RMS]"
)
print(
    f"Mean fitting error: {residual_turbulence_sim_flat.std(axis=0)[30:].mean():.3f} [rad RMS]"
)

# %% compute strehl exp from residual phases with otf ratio

pupil_exp = get_circular_pupil(modal_basis_exp.shape[1])

denominator = np.sum(
    np.real(np.fft.fft2(np.abs(np.fft.fft2(pad_array(pupil_exp, factor=2))) ** 2))
)
strehl_exp = np.array(
    [
        np.sum(
            np.real(
                np.fft.fft2(
                    np.abs(
                        np.fft.fft2(
                            pad_array(
                                pupil_exp * np.exp(1j * residual_phases_exp[i]),
                                factor=2,
                            )
                        )
                    )
                    ** 2
                )
            )
            / denominator
        )
        for i in range(residual_phases_exp.shape[0])
    ]
)

# %% compute strehl sim from residual phases with otf ratio

denominator = np.sum(
    np.real(np.fft.fft2(np.abs(np.fft.fft2(pad_array(tel.pupil, factor=2))) ** 2))
)
strehl_sim = np.array(
    [
        np.sum(
            np.real(
                np.fft.fft2(
                    np.abs(
                        np.fft.fft2(
                            pad_array(
                                tel.pupil
                                * np.exp(1j * closed_loop_data_sim.residual_phases[i]),
                                factor=2,
                            )
                        )
                    )
                    ** 2
                )
            )
            / denominator
        )
        for i in range(closed_loop_data_sim.residual_phases.shape[0])
    ]
)

# %% figure strehl ratio

fig_strehl = plt.figure(
    figsize=(width_single_column, 0.7 * width_single_column), layout="constrained"
)
plt.plot(strehl_exp[:n_iter], linestyle="-", label="experimental", linewidth=linewidth)
plt.plot(
    closed_loop_data_sim.strehl, linestyle="-", label="simulation", linewidth=linewidth
)
plt.plot(
    np.exp(-residual_turbulence_sim_flat.std(axis=0) ** 2)[:n_iter],
    label="fiting error only",
    linewidth=linewidth,
)
plt.ylabel("Strehl ratio", fontsize=fontsize_label)
plt.xlabel("# Closed loop iteration", fontsize=fontsize_label)
plt.legend(fontsize=8, loc="lower right")

fig_name = "figure_19_strehl.pdf"
fig_names.append(fig_name)

fig_strehl.savefig(fig_dir / fig_name, bbox_inches="tight", pad_inches=0.001)

print(f"Mean Strehl ratio - experimental: {strehl_exp[30:].mean():.3f}")
print(f"Mean Strehl ratio - simulation: {closed_loop_data_sim.strehl[30:].mean():.3f}")
print(
    f"Mean Strehl ratio - fitting error only: {np.exp(-residual_turbulence_sim_flat.std(axis=0)[30:] ** 2).mean():.3f}"
)

# %% Modal decomposition illustration

fig_modal_decomposition, axs = plt.subplots(
    2,
    1,
    figsize=(width_single_column, 1.5 * width_single_column),
    constrained_layout=True,
)
axs[0].plot(
    turbulent_phases_modal_decomposition_exp.std(axis=0),
    label="open loop",
    color="grey",
    zorder=0,
    linewidth=linewidth,
)

axs[0].plot(
    residual_phases_modal_decomposition_exp[30:].std(axis=0),
    label="closed loop - exp",
    linestyle="dashed",
    zorder=0,
    linewidth=linewidth,
)
axs[0].plot(
    closed_loop_data_sim.residual_phases_modal_decomposition[30:].std(axis=0),
    label="closed loop - sim",
    linestyle="dotted",
    linewidth=linewidth,
)

axs[0].axvline(
    n_controlled_modes,
    color="k",
    linestyle="--",
    label="controlled modes cutoff",
    linewidth=0.8 * linewidth,
)

axs[0].set_xscale("log")
axs[0].set_yscale("log")
axs[0].legend(loc="upper right", fontsize=6.5)

limit = 150
axs[1].plot(
    np.arange(limit, len(turbulent_phases_modal_decomposition_exp.std(axis=0))),
    turbulent_phases_modal_decomposition_exp.std(axis=0)[limit:],
    label="open loop",
    color="grey",
    zorder=0,
    linewidth=linewidth,
)
axs[1].plot(
    np.arange(limit, len(turbulent_phases_modal_decomposition_exp.std(axis=0))),
    residual_phases_modal_decomposition_exp[30:].std(axis=0)[limit:],
    label="closed loop - exp",
    linestyle="dashed",
    zorder=0,
    linewidth=linewidth,
)
line = axs[1].plot(
    np.arange(limit, len(turbulent_phases_modal_decomposition_exp.std(axis=0))),
    closed_loop_data_sim.residual_phases_modal_decomposition[30:].std(axis=0)[limit:],
    label="closed loop - sim",
    linestyle=":",
    linewidth=linewidth,
)

axs[1].axvline(
    n_controlled_modes,
    color="k",
    linestyle="--",
    label="controlled modes cutoff",
    linewidth=0.8 * linewidth,
)

axs[1].set_xscale("log")
axs[1].set_yscale("log")

x0 = limit
x1 = turbulent_phases_modal_decomposition_exp.std(axis=0).size - 1

y0, y1 = axs[1].get_ylim()

rect = Rectangle(
    (x0, y0),
    x1 - x0,
    y1 - y0,
    fill=False,
    edgecolor="purple",
    linestyle="--",
    linewidth=2,
)

axs[0].add_patch(rect)

for spine in axs[1].spines.values():
    spine.set_color("purple")
    spine.set_linestyle((0, (8, 6)))
    spine.set_linewidth(2)

fmt = FuncFormatter(lambda x, _: f"{x:g}")

ax = axs[1]  # bottom figure only

# ax.xaxis.set_major_formatter(fmt)
ax.yaxis.set_major_formatter(fmt)

# ax.xaxis.set_minor_formatter(NullFormatter())
ax.yaxis.set_minor_formatter(NullFormatter())

fig_modal_decomposition.supxlabel("# KL mode", fontsize=fontsize_label, x=0.6)
fig_modal_decomposition.supylabel(
    "Modal coefficient standard deviation [rad]", fontsize=fontsize_label, y=0.54
)

fig_name = f"figure_20_modal_decomposition_open_vs_closed_loop.pdf"
fig_names.append(fig_name)

fig_modal_decomposition.savefig(
    fig_dir / fig_name,
    bbox_inches="tight",
    pad_inches=0.05,
)

# %% PSF illustration

# experimental
focal_plane_images_exp = deepcopy(focal_plane_images_exp)
focal_plane_images_open_loop_exp = deepcopy(focal_plane_images_open_loop_exp)
reference_psf_exp = deepcopy(reference_psf_exp)

long_exposure_psf_exp = focal_plane_images_exp.mean(axis=0)
long_exposure_open_loop_psf_exp = focal_plane_images_open_loop_exp.mean(axis=0)

long_exposure_psf_exp /= long_exposure_psf_exp.sum()
long_exposure_open_loop_psf_exp /= long_exposure_open_loop_psf_exp.sum()
reference_psf_exp /= reference_psf_exp.sum()

max_coord = np.unravel_index(np.argmax(reference_psf_exp), reference_psf_exp.shape)
n_px = 250
focal_plane_images_exp_cropped = focal_plane_images_exp[
    :,
    max_coord[0] - n_px : max_coord[0] + n_px,
    max_coord[1] - n_px : max_coord[1] + n_px,
]
focal_plane_images_open_loop_exp_cropped = focal_plane_images_open_loop_exp[
    :,
    max_coord[0] - n_px : max_coord[0] + n_px,
    max_coord[1] - n_px : max_coord[1] + n_px,
]
reference_psf_exp_cropped = reference_psf_exp[
    max_coord[0] - n_px : max_coord[0] + n_px,
    max_coord[1] - n_px : max_coord[1] + n_px,
]
long_exposure_psf_exp_cropped = long_exposure_psf_exp[
    max_coord[0] - n_px : max_coord[0] + n_px,
    max_coord[1] - n_px : max_coord[1] + n_px,
]
long_exposure_open_loop_psf_exp_cropped = long_exposure_open_loop_psf_exp[
    max_coord[0] - n_px : max_coord[0] + n_px,
    max_coord[1] - n_px : max_coord[1] + n_px,
]


# compute simulated long exposure PSF

zp_factor = 12  # adapted to match the focal plane camera sampling
n_px_psf_sim = zp_factor * 21

# simulated reference psf
reference_psf_sim = np.fft.fftshift(
    np.abs(np.fft.fft2(pad_array(tel.pupil, factor=zp_factor))) ** 2
)
reference_psf_sim /= reference_psf_sim.sum()
reference_psf_sim_cropped = reference_psf_sim[
    reference_psf_sim.shape[0] // 2
    - n_px_psf_sim : reference_psf_sim.shape[0] // 2
    + n_px_psf_sim,
    reference_psf_sim.shape[1] // 2
    - n_px_psf_sim : reference_psf_sim.shape[1] // 2
    + n_px_psf_sim,
]

# simulated open loop psf
long_exposure_open_loop_psf_sim = np.zeros(
    zp_factor * np.asarray(turbulence_sim.shape[1:])
)

for index, turbulent_phase in tqdm(
    enumerate(turbulence_sim), total=turbulence_sim.shape[0]
):

    phase_screen = turbulent_phase

    complex_amplitude = pupil_sim * np.exp(1j * phase_screen)

    complex_amplitude_padded = pad_array(complex_amplitude, factor=zp_factor)

    psf = np.abs(np.fft.fft2(complex_amplitude_padded)) ** 2

    long_exposure_open_loop_psf_sim += np.fft.fftshift(psf)

long_exposure_open_loop_psf_sim /= long_exposure_open_loop_psf_sim.sum()

long_exposure_open_loop_psf_sim_cropped = long_exposure_open_loop_psf_sim[
    reference_psf_sim.shape[0] // 2
    - n_px_psf_sim : reference_psf_sim.shape[0] // 2
    + n_px_psf_sim,
    reference_psf_sim.shape[1] // 2
    - n_px_psf_sim : reference_psf_sim.shape[1] // 2
    + n_px_psf_sim,
]

# simulated closed loop psf
simulated_residual_phases = closed_loop_data_sim.residual_phases

long_exposure_psf_sim = np.zeros(
    zp_factor * np.asarray(simulated_residual_phases.shape[1:])
)

for index, residual_phase in tqdm(
    enumerate(simulated_residual_phases), total=len(simulated_residual_phases)
):

    pupil = tel.pupil
    phase_screen = residual_phase

    complex_amplitude = pupil * np.exp(1j * phase_screen)

    complex_amplitude_padded = pad_array(complex_amplitude, factor=zp_factor)

    psf = np.abs(np.fft.fft2(complex_amplitude_padded)) ** 2

    long_exposure_psf_sim += np.fft.fftshift(psf)


long_exposure_psf_sim /= long_exposure_psf_sim.sum()

long_exposure_psf_sim_cropped = long_exposure_psf_sim[
    reference_psf_sim.shape[0] // 2
    - n_px_psf_sim : reference_psf_sim.shape[0] // 2
    + n_px_psf_sim,
    reference_psf_sim.shape[1] // 2
    - n_px_psf_sim : reference_psf_sim.shape[1] // 2
    + n_px_psf_sim,
]

# plot simulated VS experimental long exposure PSF
fontsize_title_psf = 8
fontsize_label_psf = 8
fontsize_tick_psf = 6
tick_length_psf = 2

pixel_per_lambda_over_D_exp = 635e-9 * 200e-3 / (3 * 1e-3) / thorcam_pixel_pitch

half_size = long_exposure_psf_exp_cropped.shape[0] / (2 * pixel_per_lambda_over_D_exp)
extent_exp = [-half_size, half_size, -half_size, half_size]
extent_sim = [-n_px_psf_sim // zp_factor, n_px_psf_sim // zp_factor] * 2

linthresh = 3e-4

vmin = min(
    (long_exposure_open_loop_psf_exp_cropped / reference_psf_exp_cropped.max()).min(),
    (long_exposure_psf_exp_cropped / reference_psf_exp_cropped.max()).min(),
    (reference_psf_exp_cropped / reference_psf_exp_cropped.max()).min(),
    (long_exposure_psf_sim_cropped / reference_psf_sim_cropped.max()).min(),
    (long_exposure_open_loop_psf_sim_cropped / reference_psf_sim_cropped.max()).min(),
    (reference_psf_sim_cropped / reference_psf_sim_cropped.max()).min(),
)
vmax = max(
    (long_exposure_psf_exp_cropped / reference_psf_exp_cropped.max()).max(),
    (long_exposure_open_loop_psf_exp_cropped / reference_psf_exp_cropped.max()).max(),
    (reference_psf_exp_cropped / reference_psf_exp_cropped.max()).max(),
    (long_exposure_psf_sim_cropped / reference_psf_sim_cropped.max()).max(),
    (long_exposure_open_loop_psf_sim_cropped / reference_psf_sim_cropped.max()).max(),
    (reference_psf_sim_cropped / reference_psf_sim_cropped.max()).max(),
)

nrows, ncols, figsize = 2, 3, 5

axs: tuple[plt.Axes, ...]
fig_focal_plane_images, axs = plt.subplots(
    nrows=nrows,
    ncols=ncols,
    figsize=(width_single_column, 0.95 * width_single_column),
    constrained_layout=True,
    sharey="row",
)

axs[0, 0].imshow(
    long_exposure_open_loop_psf_exp_cropped / reference_psf_exp_cropped.max(),
    norm=mcolors.SymLogNorm(linthresh=linthresh, vmin=vmin, vmax=vmax),
    cmap="inferno",
    extent=extent_exp,
)

axs[0, 1].imshow(
    long_exposure_psf_exp_cropped / reference_psf_exp_cropped.max(),
    norm=mcolors.SymLogNorm(linthresh=linthresh, vmin=vmin, vmax=vmax),
    cmap="inferno",
    extent=extent_exp,
)
axs[0, 1].set_title("Experimental", fontsize=fontsize_title_psf)

axs[0, 2].imshow(
    reference_psf_exp_cropped / reference_psf_exp_cropped.max(),
    norm=mcolors.SymLogNorm(linthresh=linthresh, vmin=vmin, vmax=vmax),
    cmap="inferno",
    extent=extent_exp,
)

axs[1, 0].imshow(
    long_exposure_open_loop_psf_sim_cropped / reference_psf_sim_cropped.max(),
    norm=mcolors.SymLogNorm(linthresh=linthresh, vmin=vmin, vmax=vmax),
    cmap="inferno",
    extent=extent_sim,
)
axs[1, 1].imshow(
    long_exposure_psf_sim_cropped / reference_psf_sim_cropped.max(),
    norm=mcolors.SymLogNorm(linthresh=linthresh, vmin=vmin, vmax=vmax),
    cmap="inferno",
    extent=extent_sim,
)
axs[1, 1].set_title("Simulation", fontsize=fontsize_title_psf)
im = axs[1, 2].imshow(
    reference_psf_sim_cropped / reference_psf_sim_cropped.max(),
    norm=mcolors.SymLogNorm(linthresh=linthresh, vmin=vmin, vmax=vmax),
    cmap="inferno",
    extent=extent_sim,
)
cbar = fig_focal_plane_images.colorbar(
    im, ax=axs, aspect=45, shrink=1.0, location="bottom"
)
cbar.ax.tick_params(labelsize=0.95 * fontsize_tick_psf, length=tick_length_psf)
cbar.set_label(
    r"Normalized irradiance  $\frac{I}{\max(I_{\mathrm{ref}})}$",
    fontsize=fontsize_label_psf,
)

fig_focal_plane_images.supylabel(
    "Angular coordinate [λ/D]",
    fontsize=fontsize_label_psf,
    y=0.635,
    x=-0.035,
)
axs[1, 1].set_xlabel("Angular coordinate [λ/D]", fontsize=fontsize_label_psf)

for ax in axs.flat:
    ax.tick_params(labelsize=fontsize_tick_psf, length=tick_length_psf)

fig_name = "figure_17_psf_visualization.pdf"
fig_names.append(fig_name)

fig_focal_plane_images.savefig(
    fig_dir / fig_name,
    bbox_inches="tight",
    pad_inches=0.01,
)

# %%

print(
    f"Run completed successfully\nFigures saved in {fig_dir}\n"
    f"{fig_names[0]}\n"
    f"{fig_names[4]}\n"
    f"{fig_names[1]}\n"
    f"{fig_names[2]}\n"
    f"{fig_names[3]}"
)
