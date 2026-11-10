# %% Imports

import h5py
import matplotlib.pyplot as plt
import numpy as np
from biolabval.utils.config import Config
from maoppy.utils import circavg, compute_cog
from scipy.optimize import curve_fit
from scipy.special import j1

# %% Parameters

DATA_FILE = "utc_2026-11-03_17-12-39_psf_temporal_serie.h5"

CROP_HALF_WIDTH = 70  # [px] half-size of the crop around the mean PSF
CUT_HALF_WIDTH = 40  # [px] half-length of the PSF cuts / radial profiles

GREY_WIDTH_DIAMETER = 240e-6  # [m]
PIXEL_PITCH_ORCA = 6.5e-6  # [m]

# Figure style
FONTSIZE_LEGEND = 7
FONTSIZE_LABEL = 7
FONTSIZE_TITLE = 8
WIDTH_SINGLE_COLUMN = 3.5  # [inch]

# %% Load data

config = Config()
path = config.paths.data_dir / "vibrations" / DATA_FILE

with h5py.File(path, "r") as f:
    frames = f["frames"][...]
    exposure_time = f["frames"].attrs["exposure_time"]
    timestamps = f["timestamps"][...]
    dark = f["dark"][...]

# %% Helpers


def log_image(image):
    """Log of an image, clipped to avoid NaN/-inf on non-positive pixels."""
    return np.log(image)


def airy_2d(coords, x0, y0, I0, pixels_per_lD, background):
    """Flattened 2D Airy pattern. `coords` is a tuple (x, y) of index grids."""
    x, y = coords
    r = np.hypot(x - x0, y - y0)
    rho = np.pi * r / pixels_per_lD

    airy = np.ones_like(rho)  # limit value at rho = 0
    nonzero = rho != 0
    airy[nonzero] = (2 * j1(rho[nonzero]) / rho[nonzero]) ** 2

    return (I0 * airy + background).ravel()


def fit_airy_psf(image):
    """Fit a 2D Airy pattern to `image` and return the fitted parameters."""
    y, x = np.indices(image.shape)
    y0, x0 = np.unravel_index(np.argmax(image), image.shape)

    p0 = (x0, y0, image.max(), 10.0, np.median(image))
    popt, pcov = curve_fit(airy_2d, (x, y), image.ravel(), p0=p0)

    keys = ("x0", "y0", "I0", "pixels_per_lambda_over_D", "background")
    return {**dict(zip(keys, popt)), "covariance": pcov}


def model_image(shape, params):
    """Evaluate the fitted Airy model on an image grid of the given shape."""
    y, x = np.indices(shape)
    return airy_2d(
        (x, y),
        params["x0"],
        params["y0"],
        params["I0"],
        params["pixels_per_lambda_over_D"],
        params["background"],
    ).reshape(shape)


def peak_index(image):
    """(row, col) index of the image maximum."""
    return np.unravel_index(image.argmax(), image.shape)


def get_cut(image, axis, half_width):
    """1D cut through the image maximum, along x (axis=1) or y (axis=0)."""
    row, col = peak_index(image)
    if axis == 1:
        lo, hi = max(col - half_width, 0), col + half_width
        return image[row, lo:hi]
    lo, hi = max(row - half_width, 0), row + half_width
    return image[lo:hi, col]


def new_figure(height_ratio=0.7):
    return plt.subplots(
        figsize=(WIDTH_SINGLE_COLUMN, height_ratio * WIDTH_SINGLE_COLUMN),
        constrained_layout=True,
    )


def style_log_axis(ax, title):
    ax.set_xlabel("Radius [px]", fontsize=FONTSIZE_LABEL)
    ax.set_ylabel("Intensity [a.u.]", fontsize=FONTSIZE_LABEL)
    ax.set_yscale("log")
    ax.set_title(title, fontsize=FONTSIZE_TITLE)
    ax.legend(fontsize=FONTSIZE_LEGEND)


def plot_spectrum(signal, dt, title):
    """Plot the one-sided amplitude spectrum of `signal` sampled every `dt` s."""
    frequencies = np.fft.rfftfreq(len(signal), d=dt)
    spectrum = np.abs(np.fft.rfft(signal))

    plt.figure()
    plt.plot(frequencies, spectrum)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude [a.u.]")
    plt.title(title)
    plt.show()


def robust_cog(image, radius, n_iter=3):
    """Background-subtracted centroid in a circular window, re-centred iteratively."""
    ny, nx = image.shape
    y, x = np.indices(image.shape)

    # per-frame background from the border of the crop
    border = np.concatenate([image[0], image[-1], image[:, 0], image[:, -1]])
    img = np.clip(image - np.median(border), 0, None)

    # start at the brightest pixel (of a lightly smoothed image if it's noisy)
    y0, x0 = np.unravel_index(np.argmax(img), img.shape)
    for _ in range(n_iter):
        w = img * (np.hypot(x - x0, y - y0) <= radius)
        x0 = (w * x).sum() / w.sum()
        y0 = (w * y).sum() / w.sum()
    return x0, y0


# %% Crop around the PSF

mean_frame = frames.mean(axis=0) - dark
cog_x, cog_y = (int(c) for c in compute_cog(mean_frame))

crop = (
    slice(cog_y - CROP_HALF_WIDTH, cog_y + CROP_HALF_WIDTH),
    slice(cog_x - CROP_HALF_WIDTH, cog_x + CROP_HALF_WIDTH),
)
frames = frames[(slice(None), *crop)]
dark = dark[crop]

# %% Centre-of-gravity displacement

cogs = np.array([compute_cog(frame - dark) for frame in frames])
mean_cog = cogs.mean(axis=0)
dx = cogs[:, 0] - mean_cog[0]
dy = cogs[:, 1] - mean_cog[1]

print(f"Displacement in x: {dx.min():.2f} px to {dx.max():.2f} px")
print(f"Displacement in y: {dy.min():.2f} px to {dy.max():.2f} px")
print(f"Displacement in x: {np.ptp(dx):.2f} px")
print(f"Displacement in y: {np.ptp(dy):.2f} px")

# %% Reference PSF and Airy fit

mean_psf = frames.mean(axis=0) - dark
psf = mean_psf / mean_psf.max()

plt.figure()
plt.imshow(log_image(psf))

result = fit_airy_psf(psf)
sampling = result["pixels_per_lambda_over_D"]  # [px / (λ/D)]

print(f"Sampling = {sampling:.3f} pixels/(λ/D)")
print(
    "grey width diameter = "
    f"{GREY_WIDTH_DIAMETER / (sampling * PIXEL_PITCH_ORCA):.3e} (λ/D)"
)
print(f"Displacement in x: {np.ptp(dx) / sampling:.2f} (λ/D)")
print(f"Displacement in y: {np.ptp(dy) / sampling:.2f} (λ/D)")

psf_fit = model_image(psf.shape, result)

psf_profile = circavg(psf, center=peak_index(psf))
psf_fit_profile = circavg(psf_fit, center=peak_index(psf_fit))

# %% Plot: PSF vs fit

fig, axs = plt.subplots(
    1, 2, figsize=(WIDTH_SINGLE_COLUMN, WIDTH_SINGLE_COLUMN), constrained_layout=True
)

for ax, image, title in zip(axs, (psf, psf_fit), ("Reference PSF", "Fitted Airy PSF")):
    ax.imshow(log_image(image), cmap="inferno", origin="lower")
    ax.set_title(title, fontsize=FONTSIZE_TITLE)
    ax.set_xlabel("x [px]", fontsize=FONTSIZE_LABEL)
    ax.set_ylabel("y [px]", fontsize=FONTSIZE_LABEL)

# %% Plot: horizontal and vertical cuts

for axis, name in ((1, "horizontal"), (0, "vertical")):
    fig_cut, ax = new_figure()
    ax.plot(get_cut(psf, axis, CUT_HALF_WIDTH), label="experimental PSF", color="C0")
    ax.plot(
        get_cut(psf_fit, axis, CUT_HALF_WIDTH),
        label="Fitted Airy PSF",
        color="C1",
        linestyle="--",
    )
    style_log_axis(ax, f"PSF cut - {name}")

# %% Plot: azimuthal average

fig_avg, ax = new_figure()
ax.plot(psf_profile[:CUT_HALF_WIDTH], label="experimental PSF", color="C0")
ax.plot(
    psf_fit_profile[:CUT_HALF_WIDTH],
    label="Fitted Airy PSF",
    color="C1",
    linestyle="--",
)
style_log_axis(ax, "Radial profile - average")

# %% Temporal analysis

dt = np.mean(np.diff(timestamps))  # [s]
sampling_frequency = 1 / dt  # [Hz]
print(f"Sampling frequency = {sampling_frequency:.1f} Hz")

peak_values = frames.max(axis=(1, 2))
plot_spectrum(
    peak_values - peak_values.mean(), dt, "Vibration Spectrum - Max Pixel Value"
)
plot_spectrum(dx, dt, "Vibration Spectrum - X Displacement")
plot_spectrum(dy, dt, "Vibration Spectrum - Y Displacement")
