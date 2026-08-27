# %%

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

from biolabval.utils.config import Config

config = Config()

# %% plot parameters

fontsize_label = 10
fontsize_legend = 5.5
fontsize_title = 8
fontsize_big_title = 10
fontsize_tick = 8

linewidth = 1.1

width_single_column = 3.45  # [Inche]
width_double_column = 6.9  # [Inche]

# %% Function definitions


def piecewise_linear(x, x0, x1, y0, y1):
    return np.piecewise(
        x,
        [x < x0, (x >= x0) & (x <= x1), x > x1],
        [
            lambda x: y0,
            lambda x: y0 + (y1 - y0) * (x - x0) / (x1 - x0),
            lambda x: y1,
        ],
    )


def piecewise_linear_fixed_width(x, x0, y0, y1, width):
    x1 = x0 + width
    return piecewise_linear(x, x0, x1, y0, y1)


def fit_piecewise(profile, fixed_width=None):
    x = np.arange(len(profile), dtype=float)

    if fixed_width is None:
        p0 = [
            0.1 * len(profile),
            0.9 * len(profile),
            profile.max(),
            profile.min(),
        ]

        popt, _ = curve_fit(piecewise_linear, x, profile, p0=p0)

        return {
            "x": x,
            "x0": popt[0],
            "x1": popt[1],
            "y0": popt[2],
            "y1": popt[3],
            "fit": piecewise_linear(x, *popt),
        }

    else:

        p0 = [
            0.5 * len(profile),
            profile.max(),
            profile.min(),
        ]

        popt, _ = curve_fit(
            lambda x, x0, y0, y1: piecewise_linear_fixed_width(
                x, x0, y0, y1, fixed_width
            ),
            x,
            profile,
            p0=p0,
        )

        x0, y0, y1 = popt
        x1 = x0 + fixed_width

        return {
            "x": x,
            "x0": x0,
            "x1": x1,
            "y0": y0,
            "y1": y1,
            "fit": piecewise_linear(x, x0, x1, y0, y1),
        }


# %% load data

dirc = config.paths.data_dir / "measure_mask"
utc_measurement = "utc_2026-06-05_14-12-15"

fig_dir = config.paths.root_dir / "outputs"

my_slice = (slice(70, 1700), slice(None))

with h5py.File(dirc / f"{utc_measurement}_mask_transmission.h5", "r") as f:
    dark = f["dark"][my_slice]
    frame_colinear_mask_full = f["frame_colinear_mask"][...]
    frame_colinear_full = f["frame_colinear"][...]
    frame_colinear_mask = f["frame_colinear_mask"][my_slice]
    frame_colinear = f["frame_colinear"][my_slice]
    n_frames_avg = f.attrs["n_frames_avg"]
    frame_spatial_calibration = f["frame_spatial_calibration"][my_slice]
    frame_orthogonal = f["frame_orthogonal"][my_slice]

# %% compute transmission

mask_transmission = frame_colinear_mask / frame_colinear
mask_transmission_full = frame_colinear_mask_full / frame_colinear_full

# %% Adapt orientation for display

mask_transmission_plot = np.rot90(mask_transmission_full, k=1)[500:1700, :1890]

# %% horizontal cut - 240µm

fixed_width_px = 240 / 6.5

xstart, xend = 280, 800
ystart, yend = 520, 720
crop = mask_transmission_plot[ystart:yend, xstart:xend]

profile = np.flip(crop.mean(axis=1))

result = fit_piecewise(profile, fixed_width=fixed_width_px)

x_profile = result["x"]
x0 = result["x0"]
x1 = result["x1"]
y0 = result["y0"]
y1 = result["y1"]
fit_curve = result["fit"]

fig_mask_transmission_h, ax2 = plt.subplots(
    1,
    1,
    figsize=(width_single_column, 0.9 * width_single_column),
    constrained_layout=True,
)

x_axis = x_profile - x0 - (x1 - x0) / 2  # set 0 at the start of the profile
x_axis = x_axis * 6.5  # convert to microns

ax2.plot(x_axis, profile, "-", label="data", linewidth=linewidth)
ax2.plot(x_axis, fit_curve, ":", label="fit", linewidth=linewidth)

ax2.axvline(
    -120,
    color="k",
    ls=":",
    linewidth=0.8 * linewidth,
    label=f"{fixed_width_px*6.5:.0f} µm width",
)
ax2.axvline(120, color="k", ls=":", linewidth=0.8 * linewidth)
ax2.axhline(
    y1, color="grey", ls="--", label=f"bottom plateau = {y1:.1%}", linewidth=linewidth
)
ax2.axhline(
    y0, color="k", ls="--", label=f"top plateau = {y0:.1%}", linewidth=linewidth
)

ax2.set_xlabel("Spatial coordinate [µm]", fontsize=fontsize_label)
ax2.set_ylabel("Transmission", fontsize=fontsize_label)
ax2.legend(fontsize=7, loc="center left", framealpha=0.8, bbox_to_anchor=(0.0, 0.7))

fig_name = f"figure_6_bio_profile_illustration.pdf"

fig_mask_transmission_h.savefig(
    fig_dir / fig_name,
    bbox_inches="tight",
    pad_inches=0.01,
)

# %%

print(f"Run completed successfully.\nFigure saved in: {fig_dir / fig_name}")
