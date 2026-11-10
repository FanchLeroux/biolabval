# %%

import numpy as np


def get_circular_pupil(npx):
    D = npx + 1
    x = np.linspace(-npx / 2, npx / 2, npx)
    xx, yy = np.meshgrid(x, x)
    circle = xx**2 + yy**2
    pupil = circle < (D / 2) ** 2

    return pupil


def get_tilt(shape, theta=0.0, pupil=None):
    theta = np.deg2rad(theta)

    x = np.arange(shape[1]) - (shape[1] - 1) / 2
    y = np.arange(shape[0]) - (shape[0] - 1) / 2
    X, Y = np.meshgrid(x, y)
    Y = np.flip(Y, axis=0)

    tilt = np.cos(theta) * X + np.sin(theta) * Y

    if pupil is not None:
        tilt *= pupil
        tilt[pupil] -= tilt[pupil].mean()
        tilt[pupil] /= tilt[pupil].std()
        return tilt

    else:
        tilt -= tilt.mean()
        tilt /= tilt.std()
        return tilt


def get_stripes(shape, stripe_width=1, stripe_orientation="vertical", pupil=None):
    if stripe_orientation == "vertical":
        stripes = (np.arange(shape[1]) // stripe_width) % 2
        stripes = np.tile(stripes, (shape[0], 1))
    elif stripe_orientation == "horizontal":
        stripes = (np.arange(shape[0]) // stripe_width) % 2
        stripes = np.tile(stripes, (shape[1], 1)).T
    else:
        raise ValueError(f"Invalid stripe orientation: {stripe_orientation}")
    if pupil is not None:
        stripes[pupil == False] = 0
    return stripes


def get_chess_board(shape):
    return np.indices(shape).sum(axis=0) % 2


# %%
