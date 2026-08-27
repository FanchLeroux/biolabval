# %%

import numpy as np
import matplotlib.pyplot as plt

from biolabval.utils.config import Config

config = Config()

# %%


def piecewise_linear(x, x0, x1, y0, y1):
    """
    Compute a piecewise linear function.
    inputs:
        x: array-like, input values
        x0: float, first breakpoint
        x1: float, second breakpoint
        y0: float, value at x0
        y1: float, value at x1
    """
    return np.piecewise(
        x,
        [x < x0, (x >= x0) & (x <= x1), x > x1],
        [
            lambda x: y0,
            lambda x: y0 + (y1 - y0) * (x - x0) / (x1 - x0),
            lambda x: y1,
        ],
    )


# %% parameters

# dirc figures
fig_dir = config.paths.root_dir / "outputs"

# plot
fontsize_label = 10
fontsize_legend = 5.5
fontsize_title = 8
fontsize_big_title = 10
fontsize_tick = 8
linewidth = 1.1
width_single_column = 3.45  # [Inche]
width_double_column = 6.9  # [Inche]
transparancy = 0.5

# physical parameters
polarization_leakage_factor = 0.0497
x0 = 0.3
x1 = 1 - x0

# %%

x = np.linspace(0, 1, 100)
x_plot = x - x.mean()
y_ideal_hwp = piecewise_linear(x, x0, x1, 0.0, 1.0)
y_non_ideal_hwp = piecewise_linear(x, x0, x1, polarization_leakage_factor, 1.0)

# %%

fig = plt.figure(figsize=(width_single_column, 0.7 * width_single_column))
plt.plot(
    x_plot,
    y_ideal_hwp,
    label="ideal HWP",
    color="k",
    linewidth=linewidth,
    linestyle="--",
    zorder=10,
)
plt.plot(
    x_plot,
    y_non_ideal_hwp,
    label=f"non-ideal HWP\n{polarization_leakage_factor:.2%} polarization leakage",
    color="r",
    linewidth=linewidth,
    zorder=2,
)
plt.plot(
    x_plot,
    1 - y_ideal_hwp,
    # label="Ideal HWP - horizontal polarization",
    color="k",
    linewidth=linewidth,
    linestyle="--",
    zorder=1,
    alpha=transparancy,
)
plt.plot(
    x_plot,
    1 - y_non_ideal_hwp,
    # label="Non-ideal HWP - horizontal polarization",
    color="r",
    linewidth=linewidth,
    zorder=2,
    alpha=transparancy,
)
# plt.axhline(0.5, color="grey", linewidth=0.8 * linewidth, linestyle="-", zorder=1)
# plt.axvline(0.0, color="grey", linewidth=0.8 * linewidth, linestyle="-", zorder=1)
plt.xlabel("Spatial coordinate (arbitrary units)", fontsize=fontsize_label)
plt.ylabel("Effective transmission", fontsize=fontsize_label)
plt.xticks(fontsize=fontsize_tick)
plt.yticks(fontsize=fontsize_tick)
plt.legend(
    fontsize=6.12,
    loc="lower right",
    bbox_to_anchor=(1.0, 0.05),
)

fig_name = f"figure_5_polarization_leakage_impact.pdf"

fig.savefig(
    fig_dir / fig_name,
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.01,
)

# %%

print(f"Run completed successfully.\nFigures saved in: {fig_dir/fig_name}")
