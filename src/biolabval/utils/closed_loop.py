from dataclasses import dataclass
from copy import deepcopy

from biolabval.utils.dft import pad_array

from tqdm import tqdm
import numpy as np
from time import sleep
from biolabval.utils.display import DisplayManager
from biolabval.OOPAO.Atmosphere import Atmosphere
from biolabval.OOPAO.Telescope import Telescope
from biolabval.OOPAO.OPD_map import OPD_map


@dataclass
class ClosedLoopData:

    loop_gain: float | None = None
    leaky_gain: float | None = None
    delay: int | None = None
    n_frames_avg: int | None = None
    reference_intensities: np.ndarray | None = None

    total: np.ndarray | None = None
    residual: np.ndarray | None = None
    strehl: np.ndarray | None = None
    wfs_frames: np.ndarray | None = None
    wfs_signals: np.ndarray | None = None
    wfs_measures: np.ndarray | None = None
    reconstructed_modes: np.ndarray | None = None
    reconstructed_phases: np.ndarray | None = None
    residual_phases: np.ndarray | None = None
    turbulent_phases_modal_decomposition: np.ndarray | None = None
    reconstructed_phases_modal_decomposition: np.ndarray | None = None
    residual_phases_modal_decomposition: np.ndarray | None = None
    focal_plane_images: np.ndarray | None = None
    reference_psf: np.ndarray | None = None
    open_loop_wfs_frames: np.ndarray | None = None
    open_loop_focal_plane_images: np.ndarray | None = None


def close_the_loop(
    src,
    tel: Telescope,
    wfc,
    wfs,
    modal_basis,  # [rad]
    turbulence_phase_screens,  # [rad]
    valid_pixels,
    reconstructor,
    loop_gain: float,
    reference_intensities,
    *,
    n_iter: int = 100,
    delay: int = 2,
    leaky_gain: float = 1.0,
    n_controlled_modes: int | None = None,
    n_frames_display: int = 5,
    save_telemetry=False,
    display: str = False,
    modal_projector=None,
    n_modes_display=50,
) -> ClosedLoopData:

    if modal_projector is None:
        # modal_projector = modal_basis[:, tel.pupil] / tel.pupil.sum() # Do not work I do not know why
        modal_projector = np.linalg.pinv(modal_basis[:, tel.pupil].T)

    if n_modes_display > modal_projector.shape[0]:
        n_modes_display = modal_projector.shape[0]

    # Memory allocation (mandatory)
    total = np.zeros(n_iter)  # turbulence phase std [rad]
    residual = np.zeros(n_iter)  # residual phase std [rad]
    strehl = np.zeros(n_iter)  # Strehl ratio
    buffer_wfs_measure = np.zeros((valid_pixels.sum(), delay))
    modal_coeficients = np.zeros(reconstructor.shape[0])
    reconstructed_phase = np.zeros(modal_basis.shape[1:])

    # Variable declaration (Optional telemetry)
    wfs_frames = None
    wfs_measures = None
    reconstructed_modes = None
    reconstructed_phases = None
    wfs_signals = None
    opd_screens_rad = None
    turbulent_phases_modal_decomposition = None
    reconstructed_phases_modal_decomposition = None
    residual_phases_modal_decomposition = None
    focal_plane_images: np.ndarray | None = None
    reference_psf: np.ndarray | None = None
    open_loop_wfs_frames: np.ndarray | None = None
    open_loop_focal_plane_images: np.ndarray | None = None

    wfc.coefs = 0
    src**tel * wfc * wfs  # reset OPD

    reference_psf = np.fft.fftshift(
        np.abs(np.fft.fft2(pad_array(tel.pupil.astype(float), factor=2))) ** 2
    )
    strehl_denominator = np.sum(np.real(np.fft.fft2(np.fft.fftshift(reference_psf))))

    # Memory allocation (Optional telemetry)
    if save_telemetry:
        wfs_frames = np.zeros((n_iter,) + wfs.cam.frame.shape)
        wfs_measures = np.zeros((n_iter, valid_pixels.sum()))
        wfs_signals = np.zeros((n_iter, valid_pixels.sum()))
        reconstructed_modes = np.zeros((n_iter, reconstructor.shape[0]))
        reconstructed_phases = np.zeros((n_iter,) + modal_basis.shape[1:])
        opd_screens_rad = np.zeros((n_iter,) + modal_basis.shape[1:])
        turbulent_phases_modal_decomposition = np.zeros(
            (n_iter, modal_projector.shape[0])
        )
        reconstructed_phases_modal_decomposition = np.zeros(
            (n_iter, modal_projector.shape[0])
        )
        residual_phases_modal_decomposition = np.zeros(
            (n_iter, modal_projector.shape[0])
        )
        focal_plane_images = np.zeros((n_iter,) + reference_psf.shape)
        open_loop_wfs_frames = np.zeros((n_iter,) + wfs.cam.frame.shape)
        open_loop_focal_plane_images = np.zeros((n_iter,) + reference_psf.shape)

    if display:
        grid_shape = (2, 3)
        plot_config = [
            {
                "type": "image",
                "pos": (0, 0),
                "title": "WFS Frame",
                "data_shape": wfs.cam.frame.shape,
            },
            {
                "type": "image",
                "pos": (0, 1),
                "title": "Turbulent Phase",
                "data_shape": turbulence_phase_screens.shape[1:],
            },
            {
                "type": "image",
                "pos": (0, 2),
                "title": "Residual Phase",
                "data_shape": turbulence_phase_screens.shape[1:],
            },
            {
                "type": "curve",
                "pos": (1, 0),
                "title": "Total & Residual Std [rad]",
                "xlabel": "n_iter",
                "ylabel": "std [rad]",
                "n_points": n_iter,
                "n_lines": 2,
                "colors": ["#1f77b4", "#ff7f0e"],
                "labels": ["Total", "Residual"],
            },
            {
                "type": "modal",
                "pos": (1, 1),
                "title": "Turbulent and Reconstructed modes",
                "xlabel": "# KL mode",
                "ylabel": "std [rad]",
                "n_modes": n_modes_display,
                "colors": ["#1f77b4", "#ff7f0e"],
                "labels": ["Turbulence", "Reconstructed"],
            },
            {
                "type": "modal",
                "pos": (1, 2),
                "title": "Residual Modes",
                "xlabel": "# KL mode",
                "ylabel": "std [rad]",
                "n_modes": n_modes_display,
                "colors": ["#2ca02c"],
                "labels": ["Residuals"],
            },
        ]

        display_manager = DisplayManager(
            grid_shape, plot_config, fig_title="AO Loop Visualization"
        )

    if n_controlled_modes is None:
        n_controlled_modes = reconstructor.shape[0]
    reconstructor[n_controlled_modes:, :] = 0
    modal_basis_reshaped = modal_basis.reshape(modal_basis.shape[0], -1)

    # initialization
    src**tel * wfc * wfs

    modal_basis_reshaped = modal_basis.reshape(modal_basis.shape[0], -1)
    turbulence = OPD_map(np.zeros(turbulence_phase_screens.shape[1:]))

    # close the loop
    for k in tqdm(range(n_iter)):

        total[k] = np.std(turbulence_phase_screens[k, tel.pupil])  # [rad]

        turbulence.OPD = (
            turbulence_phase_screens[k] * src.wavelength / (2 * np.pi)
        )  # [m]
        wfc.coefs = modal_coeficients

        src**tel * turbulence * wfc * wfs
        residual[k] = (2 * np.pi) / src.wavelength * tel.OPD[tel.pupil].std()  # [rad]
        psf = np.fft.fftshift(
            np.abs(
                np.fft.fft2(
                    pad_array(
                        tel.pupil.astype(float)
                        * np.exp(1j * (2 * np.pi) / src.wavelength * tel.OPD),
                        factor=2,
                    )
                )
            )
            ** 2
        )
        strehl[k] = (
            np.sum(np.real(np.fft.fft2(np.fft.ifftshift(psf)))) / strehl_denominator
        )

        buffer_wfs_measure = np.roll(buffer_wfs_measure, -1, axis=1)
        buffer_wfs_measure[:, -1] = wfs.signal

        modal_coeficients = (
            leaky_gain * modal_coeficients
            - loop_gain * reconstructor @ buffer_wfs_measure[:, 0]
        )

        reconstructed_phase = (
            (modal_coeficients @ modal_basis_reshaped).reshape(modal_basis.shape[1:])
            * src.wavelength
            / (2 * np.pi)
        )  # [m]

        opd_screen_rad = tel.OPD * 2 * np.pi / src.wavelength

        if display | save_telemetry:
            turbulent_phase_modal_decomposition = (
                modal_projector @ turbulence_phase_screens[k, tel.pupil]
            )
            reconstructed_phase_modal_decomposition = (
                (modal_projector @ reconstructed_phase[tel.pupil])
                * 2
                * np.pi
                / src.wavelength
            )
            residual_phase_modal_decomposition = (
                (2 * np.pi / src.wavelength) * modal_projector @ tel.OPD[tel.pupil]
            )

        if save_telemetry:
            wfs_frames[k] = wfs.cam.frame
            wfs_signals[k] = wfs.signal
            wfs_measures[k] = buffer_wfs_measure[:, 0]
            reconstructed_modes[k] = modal_coeficients
            reconstructed_phases[k] = reconstructed_phase
            opd_screens_rad[k] = opd_screen_rad
            turbulent_phases_modal_decomposition[k] = (
                turbulent_phase_modal_decomposition
            )
            reconstructed_phases_modal_decomposition[k] = (
                reconstructed_phase_modal_decomposition
            )
            residual_phases_modal_decomposition[k] = residual_phase_modal_decomposition
            focal_plane_images[k] = psf

        if display and (k % n_frames_display == 0):

            display_manager.update(
                images=[
                    wfs.cam.frame,
                    turbulence_phase_screens[k],
                    opd_screen_rad,
                ],
                curves=[[total, residual]],
                modals=[
                    [
                        turbulent_phase_modal_decomposition[:n_modes_display],
                        reconstructed_phase_modal_decomposition[:n_modes_display],
                    ],
                    [
                        turbulent_phase_modal_decomposition[:n_modes_display]
                        + reconstructed_phase_modal_decomposition[:n_modes_display]
                    ],
                ],
                fig_title=f"AO Loop Visualization - iter {k}/{n_iter}",
            )

        if k % n_frames_display == 0:
            print(f"total : {total[k]}      Residual {residual[k]}    [rad]")

    wfc.coefs = np.zeros_like(modal_coeficients)
    for k in tqdm(range(n_iter), desc="Recording open loop data"):

        turbulence.OPD = (
            turbulence_phase_screens[k] * src.wavelength / (2 * np.pi)
        )  # [m]

        src**tel * turbulence * wfc * wfs
        tel.computePSF()
        open_loop_focal_plane_images[k] = tel.PSF
        open_loop_wfs_frames[k] = wfs.cam.frame

    return ClosedLoopData(
        total=total,
        residual=residual,
        strehl=strehl,
        wfs_frames=wfs_frames,
        wfs_measures=wfs_measures,
        reconstructed_modes=reconstructed_modes,
        reconstructed_phases=reconstructed_phases,
        wfs_signals=wfs_signals,
        residual_phases=opd_screens_rad,
        turbulent_phases_modal_decomposition=turbulent_phases_modal_decomposition,
        reconstructed_phases_modal_decomposition=reconstructed_phases_modal_decomposition,
        residual_phases_modal_decomposition=residual_phases_modal_decomposition,
        focal_plane_images=focal_plane_images,
        reference_psf=reference_psf,
        open_loop_wfs_frames=open_loop_wfs_frames,
        open_loop_focal_plane_images=open_loop_focal_plane_images,
    )
