import time

import tqdm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection


def custom_warning(message: str) -> None:
    """
    Print a custom warning message in bold yellow text.
    """
    bold_yellow = "\033[1m\033[33m"
    reset = "\033[0m"
    tqdm.tqdm.write(f"\r{bold_yellow}Custom Warning: {message}{reset}")


# ---------------------------------------------
# Config-driven Display Manager
# ---------------------------------------------
class DisplayManager:
    """
    Flexible display manager: images, curves, modal plots.
    Fully explicit: n_points/n_modes required, optional colors, multiple lines.
    """

    def __init__(
        self,
        grid_shape,
        plot_config,
        fig_title=None,
        pause_s=None,
        min_update_interval_s=None,
    ):
        """
        grid_shape: (n_rows, n_cols)
        plot_config: list of dicts with keys:
            - type: 'image' | 'curve' | 'modal'
            - pos: (row, col)
            - data_shape: required for 'image'
            - n_points: required for 'curve'
            - n_lines: required for 'curve'
            - n_modes: required for 'modal'
            - title: optional
            - xlabel: optional, label for x-axis of 'curve' or 'modal'
            - ylabel: optional, label for x-axis of 'curve' or 'modal'
            - colors: optional list of colors (length = n_lines or n_modes)
            - labels: optional list of labels (length = n_lines or n_modes)
        """
        plt.ion()
        self.fig, axs = plt.subplots(
            *grid_shape, figsize=(5 * grid_shape[1], 4 * grid_shape[0])
        )
        self.axs = np.atleast_2d(axs)
        self.plots = []
        self.pause_s = 0.001 if pause_s is None else pause_s
        self.min_update_interval_s = (
            0.0 if min_update_interval_s is None else min_update_interval_s
        )
        self._last_draw_time = 0.0

        if fig_title:
            self.fig.suptitle(
                fig_title,
                fontsize=14,
                fontfamily="monospace",
                y=0.96,  # keeps it away from the axes
            )

        for cfg in plot_config:
            row, col = cfg["pos"]
            ax = self.axs[row, col]
            title = cfg.get("title")
            if title:
                ax.set_title(title)

            if cfg["type"] == "image":
                shape = cfg["data_shape"]
                im = ax.imshow(np.zeros(shape), cmap="viridis", animated=True)
                ax.set_aspect("equal", adjustable="box")
                # ax.axis("off")
                self.plots.append(
                    {
                        "type": "image",
                        "ax": ax,
                        "im": im,
                        "clim": cfg.get("clim"),
                        "dynamic_clim": cfg.get("dynamic_clim", True),
                    }
                )

            elif cfg["type"] == "curve":
                n_points = cfg["n_points"]
                n_lines = cfg["n_lines"]
                colors = cfg.get("colors", [None] * n_lines)
                labels = cfg.get("labels", [None] * n_lines)
                xlabel = cfg.get("xlabel", None)
                ylabel = cfg.get("ylabel", None)
                lines = []
                for color, label in zip(colors, labels):
                    (line,) = ax.plot(np.zeros(n_points), color=color, label=label)
                    lines.append(line)
                if any(labels):
                    ax.legend(loc="upper right")
                if xlabel:
                    ax.set_xlabel(xlabel)
                if ylabel:
                    ax.set_ylabel(ylabel)
                self.plots.append({"type": "curve", "ax": ax, "lines": lines})

            elif cfg["type"] == "modal":
                n_modes = cfg["n_modes"]
                colors = cfg.get("colors")
                labels = cfg.get("labels", [None] * len(colors) if colors else [None])
                xlabel = cfg.get("xlabel", None)
                ylabel = cfg.get("ylabel", None)
                modal_lines = []
                x = np.arange(n_modes)
                for color, label in zip(colors, labels):
                    segments = [((xi, 0), (xi, 0)) for xi in x]
                    lines = LineCollection(segments, colors=color, linewidths=1)
                    ax.add_collection(lines)
                    scatter = ax.scatter(
                        x, np.zeros(n_modes), s=16, color=color, label=label, zorder=3
                    )
                    modal_lines.append({"lines": lines, "scatter": scatter})
                ax.set_xlim(-1, n_modes + 1)
                ax.set_xticks(np.arange(0, n_modes, 10))
                if any(labels):
                    ax.legend(loc="upper right")
                if xlabel:
                    ax.set_xlabel(xlabel)
                if ylabel:
                    ax.set_ylabel(ylabel)
                self.plots.append(
                    {
                        "type": "modal",
                        "ax": ax,
                        "modal_lines": modal_lines,
                        "n_modes": n_modes,
                    }
                )

            else:
                raise ValueError(f"Unknown plot type {cfg['type']}")

        # Hide unused axes
        n_rows, n_cols = grid_shape
        for r in range(n_rows):
            for c in range(n_cols):
                if not any((p["ax"] == self.axs[r, c]) for p in self.plots):
                    self.axs[r, c].axis("off")

        plt.tight_layout(rect=[0, 0, 1, 0.90])
        plt.show(block=False)

    def update(
        self, images=None, curves=None, modals=None, fig_title=None, subplot_titles=None
    ):
        img_idx = curve_idx = modal_idx = 0

        for plot in self.plots:
            if plot["type"] == "image" and images:
                data = images[img_idx]
                plot["im"].set_data(data)
                if plot["clim"] is not None:
                    plot["im"].set_clim(*plot["clim"])
                elif plot["dynamic_clim"]:
                    data_min = np.nanmin(data)
                    data_max = np.nanmax(data)
                    if np.isfinite(data_min) and np.isfinite(data_max):
                        if data_min == data_max:
                            delta = 1e-12 if data_min == 0 else abs(data_min) * 1e-3
                            plot["im"].set_clim(data_min - delta, data_max + delta)
                        else:
                            plot["im"].set_clim(data_min, data_max)
                img_idx += 1
            elif plot["type"] == "curve" and curves:
                for line, y in zip(plot["lines"], curves[curve_idx]):
                    line.set_ydata(y)
                all_y = np.concatenate([y for y in curves[curve_idx]])
                if all_y.size > 0:
                    plot["ax"].set_ylim(np.min(all_y), np.max(all_y))
                curve_idx += 1
            elif plot["type"] == "modal" and modals:
                for ml, y in zip(plot["modal_lines"], modals[modal_idx]):
                    x = np.arange(len(y))
                    segments = [((xi, 0), (xi, yi)) for xi, yi in zip(x, y)]
                    ml["lines"].set_segments(segments)
                    ml["scatter"].set_offsets(np.c_[x, y])
                max_val = max(np.max(np.abs(y)) for y in modals[modal_idx])
                if max_val > 0:
                    plot["ax"].set_ylim(-1.2 * max_val, 1.2 * max_val)
                modal_idx += 1

        # --- update subplot titles efficiently ---
        if subplot_titles:
            for plot, title in zip(self.plots, subplot_titles):
                plot["ax"].title.set_text(title)

        # --- update figure title efficiently ---
        if fig_title and self.fig._suptitle:
            self.fig._suptitle.set_text(fig_title)

        now = time.perf_counter()
        if now - self._last_draw_time < self.min_update_interval_s:
            return

        self.fig.canvas.draw_idle()
        try:
            self.fig.canvas.flush_events()
        except Exception:
            pass
        plt.pause(self.pause_s)
        self._last_draw_time = time.perf_counter()
