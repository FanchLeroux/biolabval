import math


def compact_square_layout(n_plots: int) -> tuple[int, int]:
    nrows = int(math.floor(math.sqrt(n_plots)))
    ncols = math.ceil(n_plots / nrows)
    return nrows, ncols
