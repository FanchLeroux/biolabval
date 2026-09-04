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
    turbulence = closed_loop_grp["turbulence"][:n_iter]  # [rad]
    r0 = closed_loop_grp["turbulence"].attrs["r0"]  # [m]
    reconstructor_grp = f["reconstructor_grp"]
    interaction_matrix_grp = reconstructor_grp["interaction_matrix_grp"]
    n_controlled_modes = closed_loop_grp.attrs["n_controlled_modes"]
    modal_basis_exp = interaction_matrix_grp["modal_basis"][...]

# %%

pixel_scale = 2 / turbulence.shape[1]  # [m]
pupil = get_circular_pupil(turbulence.shape[1])

# %%

plt.figure()
plt.plot(turbulence[:, pupil].std(axis=1))

# %% compute simulated long exposure PSF

zp_factor = 2  # adapted to match the focal plane camera sampling
n_px_psf_sim = zp_factor * 21  # define visualization window

# simulated open loop psf
long_exposure_open_loop_psf_sim = np.zeros(zp_factor * np.asarray(turbulence.shape[1:]))

for index, turbulent_phase in tqdm(enumerate(turbulence), total=turbulence.shape[0]):

    phase_screen = turbulent_phase

    complex_amplitude = pupil * np.exp(1j * phase_screen)

    complex_amplitude_padded = pad_array(complex_amplitude, factor=zp_factor)

    psf = np.abs(np.fft.fft2(complex_amplitude_padded)) ** 2

    long_exposure_open_loop_psf_sim += np.fft.fftshift(psf)

long_exposure_open_loop_psf_sim /= long_exposure_open_loop_psf_sim.sum()

long_exposure_open_loop_psf_sim_cropped = long_exposure_open_loop_psf_sim[
    long_exposure_open_loop_psf_sim.shape[0] // 2
    - n_px_psf_sim : long_exposure_open_loop_psf_sim.shape[0] // 2
    + n_px_psf_sim,
    long_exposure_open_loop_psf_sim.shape[1] // 2
    - n_px_psf_sim : long_exposure_open_loop_psf_sim.shape[1] // 2
    + n_px_psf_sim,
]

# %%

plt.figure()
plt.imshow(long_exposure_open_loop_psf_sim_cropped)

# %% compute fitting error

modal_basis_flat = modal_basis_exp[:n_controlled_modes, pupil].T
turbulence_flat = turbulence[:, pupil].T
modal_projector = np.linalg.pinv(modal_basis_flat)

residual_turbulence_flat = (
    turbulence_flat - ((modal_projector @ turbulence_flat).T @ modal_basis_flat.T).T
)

residual_turbulence = np.full_like(turbulence, 0.0)
residual_turbulence[:, pupil] = residual_turbulence_flat.T

mean_fitting_error = residual_turbulence_flat.std(axis=0).mean() ** 2
sr_fitting_error = np.exp(-(mean_fitting_error))

# %%

print(f"Empirical fitting error: {mean_fitting_error:.4f} [rad^2 RMS]")
print(f"SR - empirical fitting error only: {sr_fitting_error:.4f}")

# %% analytic

r0_500_nm = 12.79e-2  # [m]

# fitting

n_act = np.ceil(2 * (n_controlled_modes / np.pi) ** 0.5)
d = 2 / (n_act)  # [m]

r0_635_nm = r0_500_nm * (635e-9 / 500e-9) ** (6 / 5)

fitting_error_analytic = 0.275 * (r0_635_nm / d) ** (-5 / 3)

sr_analytic = np.exp(-(fitting_error_analytic))

print(f"r0 at 500 nm: {1e2*r0_500_nm:.2f} [cm]")
print(f"Analytic fitting error: {fitting_error_analytic:.4f} [rad^2 RMS]")
print(f"SR - analytic fitting only: {sr_analytic:.4f}")

# temporal error

loop_frequency = 1000.0  # [Hz]
loop_delay = 1 / loop_frequency  # [s]
integrator_gain = 0.7

bandwidth = (
    loop_frequency
    / (2 * np.pi)
    * (integrator_gain / (1 + 2 * loop_delay * loop_frequency)) ** 0.5
)  # [Hz]

Nr = (-1 + (1 + 8 * n_controlled_modes) ** 0.5) / 2

D = 2  # [m]

V = 10  # [m/s]

var_temporal_error = (
    0.135
    * (V / (bandwidth * D)) ** 2
    * (D / r0_635_nm) ** (5 / 3)
    * ((Nr + 1) ** (1 / 3) - 1.15)
)

print(f"Analytic Temporal error: {var_temporal_error:.4f} [rad^2 RMS]")

sr_total_analytic = np.exp(-(fitting_error_analytic + var_temporal_error))
sr_total_empiric = np.exp(-(mean_fitting_error + var_temporal_error))

print("SR analytic fitting + analytic temporal: ", sr_total_analytic)
print("SR empirical fitting + analytic temporal: ", sr_total_empiric)

# %%


def von_karman_psd(fx, fy, r0, L0):
    """
    Compute the von Karman power spectral density (PSD) of atmospheric turbulence.

    Parameters:
    fx : 2D array
        Spatial frequency grid in the x-direction.
    fy : 2D array
        Spatial frequency grid in the y-direction.
    r0 : float
        Fried parameter (coherence length) in meters.
    L0 : float
        Outer scale of turbulence in meters.

    Returns:
    psd : 2D array
        The von Karman PSD evaluated at the given spatial frequencies.
    """
    f = np.sqrt(fx**2 + fy**2)
    psd = 0.023 * r0 ** (-5 / 3) * (f**2 + (1 / L0) ** 2) ** (-11 / 6)
    return psd


# %%

n_pix = residual_turbulence.shape[1]

residual_phase_psd = (
    np.mean(
        np.abs(np.fft.fftshift(np.fft.fft2(residual_turbulence, axes=(1, 2)))) ** 2,
        axis=0,
    )
    * pixel_scale**2
    / n_pix**2
)

# %%

fx, fy = np.meshgrid(
    np.fft.fftshift(np.fft.fftfreq(residual_turbulence.shape[1], d=pixel_scale)),
    np.fft.fftshift(np.fft.fftfreq(residual_turbulence.shape[2], d=pixel_scale)),
)  # [1/m]

# %%

iy = residual_phase_psd.shape[0] // 2
f_cut = np.abs(fx[iy, :])
psd_cut = residual_phase_psd[iy, :]

# %%

plt.figure()

plt.loglog(
    f_cut,
    psd_cut,
    ".",
    label="Residual PSD",
)

plt.loglog(
    f_cut,
    von_karman_psd(f_cut, 0, r0_635_nm, 30),
    label="von Karman PSD",
)
plt.legend()
plt.xlabel(r"Spatial frequency [m$^{-1}$]")
plt.ylabel(r"PSD [rad$^2$ m$^2$]")

# %% 1D cut along fy = 0 + von Kármán fit

from scipy.optimize import least_squares

# Center row: fy = 0
iy = residual_phase_psd.shape[0] // 2

# Positive fx frequencies only
positive = fx[iy, :] >= 0

f_cut = fx[iy, positive]
psd_cut = residual_phase_psd[iy, positive]


# %% Frequency range used for the fit

f_fit_min = 6.0  # [m^-1]
f_fit_max = 10.0  # [m^-1]

fit = (f_cut >= f_fit_min) & (f_cut <= f_fit_max)

f_fit = f_cut[fit]
psd_fit = psd_cut[fit]


# %% 1D von Kármán model, evaluated along fy = 0

L0 = 30.0  # [m]


def von_karman_psd_1d(fx, r0):
    return 0.023 * r0 ** (-5 / 3) * (fx**2 + (1 / L0) ** 2) ** (-11 / 6)


# %% Fit r0


def residuals(log_r0):
    r0 = np.exp(log_r0[0])

    return np.log(von_karman_psd_1d(f_fit, r0)) - np.log(psd_fit)


result = least_squares(
    residuals,
    x0=[np.log(r0_635_nm)],
)

r0_fit = np.exp(result.x[0])

print(f"Fitted r0 = {r0_fit:.3f} m")
print(f"Fixed L0 = {L0:.1f} m")


# %%

print(f"r0 at 500 nm: {1e2*r0_500_nm:.2f} [cm]")
print(f"r0 at 635 nm: {1e2*r0_635_nm:.2f} [cm]\n")

print(f"Empirical fitting error: {mean_fitting_error:.4f} [rad^2 RMS]")
print(f"SR - empirical fitting error only: {sr_fitting_error:.4f}\n")

print(f"Analytic fitting error: {fitting_error_analytic:.4f} [rad^2 RMS]")
print(f"SR - analytic fitting only: {sr_analytic:.4f}")

print(f"Analytic Temporal error: {var_temporal_error:.4f} [rad^2 RMS]\n")

print("SR analytic fitting + analytic temporal: ", sr_total_analytic)
print("SR empirical fitting + analytic temporal: ", sr_total_empiric)

# %% Plot

r0_500_nm_init = 0.1  # [m]
r0_635_nm_init = r0_500_nm_init * (635e-9 / 500e-9) ** (6 / 5)

f_model = f_cut

plt.figure(figsize=(width_double_column, 0.7 * width_double_column))

plt.loglog(
    f_cut,
    psd_cut,
    ".",
    label="Residual PSD",
)

plt.loglog(
    f_model,
    von_karman_psd_1d(f_model, r0_635_nm_init),
    label="Initial von Kármán\n"
    rf"$r_0={1e2 * r0_635_nm_init:.2f}$ cm at 635 nm"
    "\n"
    rf"$r_0={1e2 * r0_500_nm_init:.2f}$ cm at 500 nm",
    linestyle=":",
)

plt.loglog(
    f_model,
    von_karman_psd_1d(f_model, r0_fit),
    "-",
    label="Fitted von Kármán\n"
    rf"$r_0={1e2*r0_fit:.2f}$ cm at 635 nm"
    "\n"
    rf"$r_0={1e2 * r0_fit * (500e-9 / 635e-9) ** (6 / 5):.2f}$ cm at 500 nm)",
    zorder=0,
)

# Highlight fitting range
plt.axvspan(
    f_fit_min,
    f_fit_max,
    alpha=0.15,
    label="Fitting range",
)
plt.ylim(1e-9, 2e1)
plt.xlabel(r"Spatial frequency $f_x$ [m$^{-1}$]")
plt.ylabel(r"$W_\phi(f_x,0)$ [rad$^2$ m$^2$]")
plt.legend()
plt.show()

# %%
