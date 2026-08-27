# %%

import math

from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import numpy as np

from biolabval.OOPAO.Source import Source
from biolabval.OOPAO.Telescope import Telescope
from biolabval.OOPAO.DeformableMirror import DeformableMirror
from biolabval.OOPAO.BioEdge import BioEdge
from biolabval.OOPAO.Pyramid import Pyramid
from biolabval.OOPAO.calibration.InteractionMatrix import InteractionMatrix

from biolabval.utils.modal_basis import compute_fourier_mode
from biolabval.utils.sensitivity import compute_photon_noise_sensitivity

from biolabval.utils.config import Config

config = Config()

# %% Parameters

# sampling
resolution = 128

# modal basis
n_subapertures = resolution // 1
n_fourier_modes = resolution // 8 - 1
kind = "vertical"  # "diagonal" or "vertical"
kinds = ["diagonal", "vertical"]

# optical setup
modulation = 5  # radius or half grey width [lambda/D]
profile = lambda x: x

n_pix_edge_bioedge = 0
n_pix_separation_bioedge = 0

n_pix_edge_pyramid = resolution // 2
n_pix_separation_pyramid = resolution

stroke_rad = 0.01  # [rad]
extra_modulation_factor = 1  # half_grey_width = extra_modulation_factor*modulation

post_processing = "fullFrame_sum_flux"  # fullFrame_input_pupil_flux ; fullFrame_sum_flux ; fullFrame or slopesMaps

post_processing_grey_bioedge = "fullFrame_sum_flux"  # "fullFrame_input_pupil_flux"

# dirc figures
fig_dir = config.paths.root_dir / "outputs"

# linearity tests

amplitudes_rad = np.linspace(-5, 5, 30)

# plot
fontsize_label = 10
fontsize_legend = 5.5
fontsize_title = 8
fontsize_big_title = 10
fontsize_tick = 8
linewidth = 1.1
width_single_column = 3.45  # [Inche]
width_double_column = 6.9  # [Inche]

# %% build numerical twin

src = Source("R", magnitude=0)
stroke_m = stroke_rad * src.wavelength / (2 * np.pi)  # [modal_basis_sim]

# consider a slightly padded pupil (avoid edge effects for the numerical twin)
tel = Telescope(resolution=resolution, diameter=2)
# tel.pupil = np.ones(tel.pupil.shape)
src * tel

# %% compute modal basis

diagonal_fourier_modes = np.full([n_fourier_modes, *([tel.resolution] * 2)], np.nan)
vertical_fourier_modes = np.full([n_fourier_modes, *([tel.resolution] * 2)], np.nan)

for k in range(1, n_fourier_modes + 1):
    diagonal_fourier_modes[k - 1] = tel.pupil * compute_fourier_mode(resolution, k, k)
    vertical_fourier_modes[k - 1] = tel.pupil * compute_fourier_mode(resolution, k, 0)
diagonal_fourier_modes /= diagonal_fourier_modes[:, tel.pupil].std(axis=1)[
    :, None, None
]
vertical_fourier_modes /= vertical_fourier_modes[:, tel.pupil].std(axis=1)[
    :, None, None
]

# %% Bi- O Edge numerical twin

# compute the bi-o edge model
grey_bioedge = BioEdge(
    nSubap=n_subapertures,
    telescope=tel,
    modulation=0,
    lightRatio=0.1,  # no need to use it since we input the valid pixel map
    n_pix_edge=n_pix_edge_bioedge,
    n_pix_separation=n_pix_separation_bioedge,
    postProcessing=post_processing_grey_bioedge,  # normalisation based on the flux on the detector (same as what is done on the bench)
    grey_width=extra_modulation_factor * modulation,
)

# %%

modulated_bioedge = BioEdge(
    nSubap=n_subapertures,
    telescope=tel,
    modulation=modulation,
    lightRatio=0.0,  # no need to use it since we input the valid pixel map
    n_pix_edge=n_pix_edge_bioedge,
    n_pix_separation=n_pix_separation_bioedge,
    postProcessing=post_processing,  # normalisation based on the flux on the detector (same as what is done on the bench)
    grey_width=0,
)

# %%

non_modulated_bioedge = BioEdge(
    nSubap=n_subapertures,
    telescope=tel,
    modulation=0,
    lightRatio=0.0,  # no need to use it since we input the valid pixel map
    n_pix_edge=n_pix_edge_bioedge,
    n_pix_separation=n_pix_separation_bioedge,
    postProcessing=post_processing,  # normalisation based on the flux on the detector (same as what is done on the bench)
    grey_width=0,
)

# %% modulated pyramid

modulated_pyramid = Pyramid(
    nSubap=n_subapertures,
    telescope=tel,
    modulation=modulation,  #  modulation radius
    lightRatio=0.1,  # no need to use it since we input the valid pixel map
    n_pix_edge=n_pix_edge_pyramid,
    extraModulationFactor=12,  # make sure the modulation is well sampled
    n_pix_separation=n_pix_separation_pyramid,
    postProcessing=post_processing,  # normalisation based on the flux on the detector (same as what is done on the bench)
    userValidSignal=None,  # input user-define valid pixel map
)

# %% non modulated pyramid

non_modulated_pyramid = Pyramid(
    nSubap=n_subapertures,
    telescope=tel,
    modulation=0,  # modulation radius
    lightRatio=0.0,
    n_pix_edge=n_pix_edge_pyramid,
    n_pix_separation=n_pix_separation_pyramid,
    postProcessing=post_processing,  # normalisation based on the flux on the detector (same as what is done on the bench)
    userValidSignal=None,
)

fig_names = []
for kind in kinds:

    # %% select modal basis

    if kind == "vertical":
        modes = vertical_fourier_modes.reshape(diagonal_fourier_modes.shape[0], -1).T

    if kind == "diagonal":
        modes = diagonal_fourier_modes.reshape(diagonal_fourier_modes.shape[0], -1).T

    # %% create modal modal_dm

    modal_dm = DeformableMirror(
        tel,
        nSubap=resolution,
        modes=modes,
    )

    # %% compute simulated interaction matrix - bioedge

    calib_grey_bioedge = InteractionMatrix(
        ngs=src,
        tel=tel,
        wfs=grey_bioedge,
        dm=modal_dm,
        M2C=np.identity(modal_dm.nValidAct),
        stroke=stroke_m,
        display=True,
        single_pass=False,
    )

    interaction_matrix_grey_bioedge = (
        calib_grey_bioedge.D * src.wavelength / (2 * np.pi)
    )

    # %%

    calib_modulated_bioedge = InteractionMatrix(
        ngs=src,
        tel=tel,
        wfs=modulated_bioedge,
        dm=modal_dm,
        M2C=np.identity(modal_dm.nValidAct),
        stroke=stroke_m,
        display=True,
        single_pass=False,
    )

    interaction_matrix_modulated_bioedge = (
        calib_modulated_bioedge.D * src.wavelength / (2 * np.pi)
    )

    calib_non_modulated_bioedge = InteractionMatrix(
        ngs=src,
        tel=tel,
        wfs=non_modulated_bioedge,
        dm=modal_dm,
        M2C=np.identity(modal_dm.nValidAct),
        stroke=stroke_m,
        display=True,
        single_pass=False,
    )

    interaction_matrix_non_modulated_bioedge = (
        calib_non_modulated_bioedge.D * src.wavelength / (2 * np.pi)
    )

    # %% compute simulated interaction matrix - pyramid

    calib_modulated_pyramid = InteractionMatrix(
        ngs=src,
        tel=tel,
        wfs=modulated_pyramid,
        dm=modal_dm,
        M2C=np.identity(modal_dm.nValidAct),
        stroke=stroke_m,
        display=True,
        single_pass=False,
    )

    interaction_matrix_modulated_pyramid = (
        calib_modulated_pyramid.D * src.wavelength / (2 * np.pi)
    )

    calib_non_modulated_pyramid = InteractionMatrix(
        ngs=src,
        tel=tel,
        wfs=non_modulated_pyramid,
        dm=modal_dm,
        M2C=np.identity(modal_dm.nValidAct),
        stroke=stroke_m,
        display=True,
        single_pass=False,
    )

    interaction_matrix_non_modulated_pyramid = (
        calib_non_modulated_pyramid.D * src.wavelength / (2 * np.pi)
    )

    # %% Sensitivity

    # photon noise sensitivity
    photon_noise_sensitivity_grey_bioedge = compute_photon_noise_sensitivity(
        interaction_matrix_grey_bioedge, grey_bioedge.referenceSignal
    )
    photon_noise_sensitivity_modulated_pyramid = compute_photon_noise_sensitivity(
        interaction_matrix_modulated_pyramid, modulated_pyramid.referenceSignal
    )
    photon_noise_sensitivity_non_modulated_pyramid = compute_photon_noise_sensitivity(
        interaction_matrix_non_modulated_pyramid,
        non_modulated_pyramid.referenceSignal,
    )

    # %% plots

    # photon noise sensitivity fourier modes
    plt.figure(figsize=(width_single_column, 0.7 * width_single_column))
    plt.plot(
        [np.nan, *photon_noise_sensitivity_non_modulated_pyramid],
        "-",
        label="non-modulated pyramid",
        linewidth=linewidth,
        color="#2ca02c",
    )
    plt.plot(
        [0, *photon_noise_sensitivity_modulated_pyramid],
        "-",
        label="modulated pyramid",
        linewidth=linewidth,
        color="#ff7f0e",
    )
    plt.plot(
        [0, *photon_noise_sensitivity_grey_bioedge],
        "-",
        label="grey bioedge",
        linewidth=linewidth,
        color="#1f77b4",
    )
    plt.axhline(
        y=2**0.5,
        color="k",
        linewidth=0.8 * linewidth,
        linestyle="--",
        label="$\\sqrt{2}$",
        zorder=0,
    )
    if kind == "diagonal":
        plt.axhline(
            y=2**0.5 / 2,
            color="grey",
            linewidth=0.8 * linewidth,
            linestyle="--",
            label=r"$\frac{\sqrt{2}}{2}$",
            zorder=0,
        )
    if kind == "vertical":
        plt.axhline(
            y=1,
            color="grey",
            linewidth=0.8 * linewidth,
            linestyle="--",
            label="1",
            zorder=0,
        )

    plt.legend(fontsize=fontsize_legend)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlabel("# cycle per pupil", fontsize=fontsize_label)
    plt.ylabel("$S_{ph}$", fontsize=fontsize_label)

    fig_name = f"{kind}_fourier_modes_photon_noise_sensitivity_bioedge_pyramid.pdf"
    fig_names.append(fig_name)

    plt.savefig(
        fig_dir / fig_name,
        bbox_inches="tight",
        pad_inches=0.01,
        dpi=300,
    )

# %%

print(
    f"Run completed successfully.\nFigures saved in: {fig_dir/fig_names[0]}\nand {fig_dir/fig_names[1]}"
)
