# -*- coding: utf-8 -*-
"""
Created on Wed Jan 28 09:28:01 2026

@author: fleroux

Run in tmux session with:

tmux new-session -d -s my_session_name "cd /net/SRVSTK12C/harmoni/fleroux/code/project/bioedge_bench/ && source .venv/bin/activate && python /net/SRVSTK12C/harmoni/fleroux/code/project/bioedge_bench/scripts/sim_bioedgeulations/bioedge_prototype_bench_paper/sim_bioedgeulation_vs_bioedge_bench.py"


"""

# %%

import h5py
import numpy as np
import matplotlib.pyplot as plt

from biolabval.OOPAO.Source import Source
from biolabval.OOPAO.Telescope import Telescope
from biolabval.OOPAO.DeformableMirror import DeformableMirror
from biolabval.OOPAO.Pyramid import Pyramid
from biolabval.OOPAO.BioEdge import BioEdge
from biolabval.OOPAO.calibration.InteractionMatrix import InteractionMatrix
from biolabval.OOPAO.tools.interpolateGeometricalTransformation import interpolate_cube
from biolabval.utils.sensitivity import compute_photon_noise_sensitivity
from biolabval.utils.config import Config

# %% Parameters

config = Config()

# utc_imat = "utc_2026-06-24_14-11-43"
utc_imat = "utc_2026-03-25_08-55-54"

# optical setup
wavelength = 635e-9  # [m]
modulation_diameter = 5  # [lambda/D]
modulation = modulation_diameter / 2  # radius or half grey width [lambda/D]

# sim_bioedge simulation sampling
sim_bioedge_zero_padding_factor = 4
sim_bioedge_tel_resolution_factor = 2

# dirc figures
fig_dir = config.paths.root_dir / "outputs"


# plotting parameters
width_single_column = 3.45  # [Inche]
width_double_column = 6.9  # [Inche]
fontsize_label = 10
fontsize_legend = 8
fontsize_title = 8
fontsize_big_title = 10
fontsize_tick = 10
linewidth = 1.1

# %% Import modal basis

with h5py.File(
    config.paths.data_dir / "interaction_matrix" / f"{utc_imat}_interaction_matrix.h5"
) as f:
    modal_basis_exp = f["interaction_matrix_grp"]["modal_basis"][...]

# %% Extract bioedge simulation parameters

wfs_resolution = 144

# resolution for the FFT
wfs_sim_bioedge_cam_resolution = wfs_resolution * sim_bioedge_zero_padding_factor

# %% build numerical twin

src = Source("R", magnitude=0)

# consider a slightly padded pupil (avoid edge effects for the numerical twin)
n_pixel_padded = 2

tel = Telescope(
    resolution=wfs_resolution * sim_bioedge_tel_resolution_factor - 2 * n_pixel_padded,
    diameter=2,
)  # n_subp per tel diameter
tel.pad(n_pixel_padded)

# interpolate and flip the data
modal_basis_sim_bioedge = interpolate_cube(
    modal_basis_exp,
    pixel_size_in=tel.D / modal_basis_exp.shape[1],
    pixel_size_out=tel.D / tel.resolution,
    resolution_out=tel.resolution,
    shape_out=[tel.resolution, tel.resolution],
    mis_registration=None,
    fliplr=True,
    flipud=False,
)

src**tel

# %% non_modulated_pyramid numerical twin

non_modulated_pyramid = Pyramid(
    nSubap=wfs_resolution,
    telescope=tel,
    modulation=0,
    lightRatio=0.0,  # no need to use it since we input the valid pixel map
    n_pix_edge=wfs_resolution // 2,
    n_pix_separation=wfs_resolution,
    postProcessing="fullFrame_sum_flux",  # normalisation based on the flux on the detector (same as what is done on the bench)
    userValidSignal=None,
)

src**tel * non_modulated_pyramid

# %% modulated_pyramid numerical twin

modulated_pyramid = Pyramid(
    nSubap=wfs_resolution,
    telescope=tel,
    modulation=modulation,
    lightRatio=0.1,  # no need to use it since we input the valid pixel map
    n_pix_edge=wfs_resolution // 2,
    n_pix_separation=wfs_resolution,
    postProcessing="fullFrame_sum_flux",  # normalisation based on the flux on the detector (same as what is done on the bench)
    userValidSignal=None,
)

src**tel * modulated_pyramid

# %% Bi- O Edge numerical twin

grey_bioedge = BioEdge(
    nSubap=wfs_resolution,
    telescope=tel,
    modulation=0,
    lightRatio=0.1,  # no need to use it since we input the valid pixel map
    n_pix_edge=(wfs_sim_bioedge_cam_resolution - 2 * wfs_resolution) // 4
    - wfs_resolution // 2,
    n_pix_separation=0,
    postProcessing="fullFrame_sum_flux",  # normalisation based on the flux on the detector (same as what is done on the bench)
    grey_width=modulation,
    userValidSignal=None,
)

src**tel * grey_bioedge

dm = DeformableMirror(
    tel,
    nSubap=modal_basis_sim_bioedge.shape[1],
    modes=modal_basis_sim_bioedge.reshape((modal_basis_sim_bioedge.shape[0], -1)).T,
)

# %% compute sim_non_modulated_pyramid interaction matrix

stroke_rad = 0.01  # [rad]
stroke_m = (
    stroke_rad * src.wavelength / (2 * np.pi)
)  # [modal_basis_sim_non_modulated_pyramid]

sim_non_modulated_pyramid_calib = InteractionMatrix(
    ngs=src,
    tel=tel,
    wfs=non_modulated_pyramid,
    dm=dm,
    M2C=np.identity(
        dm.coefs.shape[0]
    ),  # full M2C => identity matrix since we consider a modal DM
    stroke=stroke_m,
    invert=False,
    display=True,
    single_pass=False,
)

interaction_matrix_sim_non_modulated_pyramid = (
    sim_non_modulated_pyramid_calib.D * src.wavelength / (2 * np.pi)
)

# %% compute sim_modulated_pyramid interaction matrix

sim_modulated_pyramid_calib = InteractionMatrix(
    ngs=src,
    tel=tel,
    wfs=modulated_pyramid,
    dm=dm,
    M2C=np.identity(
        dm.coefs.shape[0]
    ),  # full M2C => identity matrix since we consider a modal DM
    stroke=stroke_m,
    invert=False,
    display=True,
    single_pass=False,
)

interaction_matrix_sim_modulated_pyramid = (
    sim_modulated_pyramid_calib.D * src.wavelength / (2 * np.pi)
)

# %% compute sim_bioedge interaction matrix

sim_bioedge_calib = InteractionMatrix(
    ngs=src,
    tel=tel,
    wfs=grey_bioedge,
    dm=dm,
    M2C=np.identity(
        dm.coefs.shape[0]
    ),  # full M2C => identity matrix since we consider a modal DM
    stroke=stroke_m,
    invert=False,
    display=True,
    single_pass=False,
)

interaction_matrix_sim_bioedge = sim_bioedge_calib.D * src.wavelength / (2 * np.pi)

# %% Compute sensitivities

photon_noise_sensitivity_sim_non_modulated_pyramid = compute_photon_noise_sensitivity(
    interaction_matrix_sim_non_modulated_pyramid, non_modulated_pyramid.referenceSignal
)

photon_noise_sensitivity_sim_modulated_pyramid = compute_photon_noise_sensitivity(
    interaction_matrix_sim_modulated_pyramid, modulated_pyramid.referenceSignal
)

photon_noise_sensitivity_sim_bioedge = compute_photon_noise_sensitivity(
    interaction_matrix_sim_bioedge, grey_bioedge.referenceSignal
)

# %% Plot photon noise sensitivities

nrows, ncols = 1, 1
fig_phot_sensitivity, axs = plt.subplots(
    nrows=nrows,
    ncols=ncols,
    sharex=True,
    figsize=(width_single_column, 0.95 * width_single_column),
    constrained_layout=True,
)
axs = np.atleast_1d(axs).ravel()

axs[0].plot(
    photon_noise_sensitivity_sim_non_modulated_pyramid.tolist(),
    label="non modulated pyramid",
    linestyle="-",
    linewidth=linewidth,
    color="#2ca02c",
)
axs[0].plot(
    photon_noise_sensitivity_sim_modulated_pyramid.tolist(),
    label="modulated pyramid",
    linestyle="-",
    linewidth=linewidth,
    color="#ff7f0e",
)
axs[0].plot(
    photon_noise_sensitivity_sim_bioedge.tolist(),
    label="grey bioedge",
    linestyle="-.",
    linewidth=linewidth,
    color="#1f77b4",
)
axs[0].axhline(
    2**0.5,
    color="k",
    linestyle="--",
    label=r"$\sqrt{2}$",
    linewidth=0.8 * linewidth,
)

axs[0].axhline(
    1,
    color="grey",
    linestyle="-",
    label="1",
    linewidth=0.8 * linewidth,
)

axs[0].axhline(
    2**0.5 / 2,
    color="grey",
    linestyle="--",
    label=r"$\frac{\sqrt{2}}{2}$",
    linewidth=0.8 * linewidth,
)
axs[0].set_ylim(0, 1.5)
axs[0].set_xlabel("# KL mode", fontsize=fontsize_label)
axs[0].set_ylabel(r"$S_{ph}$", fontsize=fontsize_label)
axs[0].legend(loc="lower right", fontsize=fontsize_legend)

fig_name = f"figure_Dp3_pyramid_vs_bioedge_KL_modes.pdf"

fig_phot_sensitivity.savefig(
    fig_dir / fig_name,
    bbox_inches="tight",
    pad_inches=0.01,
)

# %%

print(f"Run completed successfully.\nFigures saved in: {fig_dir/fig_name}")
