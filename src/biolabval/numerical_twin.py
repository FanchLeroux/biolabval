# %% Imports

import copy
from dataclasses import dataclass

import numpy as np

from OOPAO.Source import Source
from OOPAO.Telescope import Telescope
from OOPAO.DeformableMirror import DeformableMirror
from OOPAO.BioEdge import BioEdge
from OOPAO.calibration.InteractionMatrix import InteractionMatrix
from OOPAO.tools.interpolateGeometricalTransformation import interpolate_cube
from OOPAO.MisRegistration import MisRegistration
from OOPAO.OPD_map import OPD_map
from OOPAO.tools.interpolateGeometricalTransformation import interpolate_cube
from scipy import ndimage as ndi
from scipy.optimize import least_squares

# from OOPAO.tools.tools import centroid

# %% Function definitions


def centroid(image, threshold=0.01):
    if np.ndim(image) <= 2:
        im = np.reshape(image.copy(), (1, np.shape(image)[0], np.shape(image)[1]))
    else:
        im = np.atleast_3d(image.copy())
    im[im < (threshold * im.max())] = 0
    centroid_out = np.zeros([im.shape[0], 2])
    X_map, Y_map = np.meshgrid(np.arange(im.shape[1]), np.arange(im.shape[2]))
    X_coord_map = np.atleast_3d(X_map).T
    Y_coord_map = np.atleast_3d(Y_map).T
    norma = np.sum(np.sum(im, axis=1), axis=1)
    centroid_out[:, 0] = np.sum(np.sum(im * X_coord_map, axis=1), axis=1) / norma
    centroid_out[:, 1] = np.sum(np.sum(im * Y_coord_map, axis=1), axis=1) / norma
    return centroid_out


def check_wfs_pupils(valid_pixel_map, wfs, n_it=3, correct=False):

    xs = wfs.sx
    ys = wfs.sy
    if correct is False:
        for i in range(4):
            I = wfs.grabFullQuadrant(i + 1, valid_pixel_map)
            xc = I.shape[0] // 2
            [y, x] = np.asarray(centroid(I, threshold=0.3)[0])
            I_ = np.abs(wfs.grabFullQuadrant(i + 1))
            I_ /= I_.max()
            [y_, x_] = np.asarray(centroid(I_, threshold=0.3)[0])

    else:
        for i_it in range(n_it):
            wfs.apply_shift_wfs(sx=xs, sy=ys)

            for i in range(4):
                I = wfs.grabFullQuadrant(i + 1, valid_pixel_map)
                xc = I.shape[0] // 2
                [y, x] = np.asarray(centroid(I, threshold=0.3)[0])
                I_ = np.abs(wfs.grabFullQuadrant(i + 1))
                I_ /= I_.max()
                [y_, x_] = np.asarray(centroid(I_, threshold=0.3)[0])
                xs[i] += x - x_
                ys[i] += y_ - y
        return xs, ys


def mini_sprint(
    ngs,
    tel,
    wfs,
    modal_basis_exp,  # 3D cube of the modal basis, n_modes * n_res*n_res
    index_sprint,  # modes index considered for sprint (the more the better... but the slower)
    int_mat_exp,
    arg_mis_reg=None,
    starting_point=None,
    n_iteration=5,
    single_pass=True,
):  # reference experimental interaction matrix
    # mis-reg parameters to be considered (the order matters)
    if arg_mis_reg is None:
        arg_mis_reg = [
            "rotationAngle",
            "shiftX",
            "shiftY",
            "magnification" "radialScaling",
            "tangentialScaling",
        ]
    if np.isscalar(index_sprint):
        index_sprint = [index_sprint]
    mis_registration_dict = dict()
    mis_registration_dict["rotationAngle"] = dict()
    mis_registration_dict["rotationAngle"]["delta"] = 0.01
    mis_registration_dict["rotationAngle"]["units"] = "[deg]"

    mis_registration_dict["shiftX"] = dict()
    mis_registration_dict["shiftX"]["delta"] = 0.01 * tel.D / wfs.nSubap
    mis_registration_dict["shiftX"]["units"] = "[m]"

    mis_registration_dict["shiftY"] = dict()
    mis_registration_dict["shiftY"]["delta"] = 0.01 * tel.D / wfs.nSubap
    mis_registration_dict["shiftY"]["units"] = "[m]"

    mis_registration_dict["magnification"] = dict()
    mis_registration_dict["magnification"]["delta"] = 0.01
    mis_registration_dict["magnification"]["units"] = "[%]"

    mis_registration_dict["radialScaling"] = dict()
    mis_registration_dict["radialScaling"]["delta"] = 0.01
    mis_registration_dict["radialScaling"]["units"] = "[%]"

    mis_registration_dict["tangentialScaling"] = dict()
    mis_registration_dict["tangentialScaling"]["delta"] = 0.01
    mis_registration_dict["tangentialScaling"]["units"] = "[%]"

    int_mat_exp = np.squeeze(int_mat_exp[:, index_sprint])
    # considered modal basis and reference signal
    if starting_point is None:
        starting_point = MisRegistration()  # reference mis-registration (model)

    basis = modal_basis_exp[index_sprint, :, :]
    # % start mini-sprint
    misreg_id = MisRegistration()
    mis_reg_estimate_buffer = []
    for i_it in range(n_iteration):
        meta_mat = (
            []
        )  # list to stack all the sensitivity matrices (re-initialized at each iteration)
        m_ = interpolate_cube(
            basis,
            pixel_size_in=tel.D / tel.resolution,
            pixel_size_out=tel.D / tel.resolution,
            resolution_out=tel.resolution,
            shape_out=[tel.resolution, tel.resolution],
            mis_registration=starting_point,
            fliplr=False,
            flipud=False,
        )
        amp = 1e-12
        signal = []
        for i_mode in range(len(index_sprint)):
            opd = OPD_map(np.squeeze(m_[i_mode, :, :]) * tel.pupil * amp)
            ngs**tel * opd * wfs
            signal.append(wfs.signal / amp)
        ref_wfs_signal = np.squeeze(np.asarray(signal).T)
        if i_it == 0:
            ref_wfs_signal_0 = ref_wfs_signal.copy()
        # print(ref_wfs_signal.shape)

        for i in range(len(arg_mis_reg)):
            # initialize the delta-mis-reg corresponding to each mis-registration type
            delta_misreg = MisRegistration()
            if arg_mis_reg[i] == "magnification":
                setattr(
                    delta_misreg,
                    "tangentialScaling",
                    mis_registration_dict[arg_mis_reg[i]]["delta"],
                )
                setattr(
                    delta_misreg,
                    "radialScaling",
                    mis_registration_dict[arg_mis_reg[i]]["delta"],
                )
            else:
                setattr(
                    delta_misreg,
                    arg_mis_reg[i],
                    mis_registration_dict[arg_mis_reg[i]]["delta"],
                )
            # interpolate the modal basis of the model accordingly
            m_ = interpolate_cube(
                basis,
                pixel_size_in=tel.D / tel.resolution,
                pixel_size_out=tel.D / tel.resolution,
                resolution_out=tel.resolution,
                shape_out=[tel.resolution, tel.resolution],
                mis_registration=misreg_id + delta_misreg + starting_point,
                fliplr=False,
                flipud=False,
            )
            # print(m_.shape)
            signal = []
            for i_mode in range(len(index_sprint)):
                opd = OPD_map(np.squeeze(m_[i_mode, :, :]) * tel.pupil * amp)
                ngs**tel * opd * wfs
                signal.append(wfs.signal / amp)
            signal = np.hstack(signal)
            push = (signal) / mis_registration_dict[arg_mis_reg[i]]["delta"]

            if single_pass:
                pull = -push
            else:
                m_ = interpolate_cube(
                    basis,
                    pixel_size_in=tel.D / tel.resolution,
                    pixel_size_out=tel.D / tel.resolution,
                    resolution_out=tel.resolution,
                    shape_out=[tel.resolution, tel.resolution],
                    mis_registration=misreg_id - delta_misreg + starting_point,
                    fliplr=False,
                    flipud=False,
                )
                signal = []
                for i_mode in range(len(index_sprint)):
                    opd = OPD_map(np.squeeze(m_[i_mode, :, :]) * tel.pupil * amp)
                    ngs**tel * opd * wfs
                    signal.append(wfs.signal / amp)
                signal = np.hstack(signal)
                pull = (signal) / mis_registration_dict[arg_mis_reg[i]]["delta"]
            meta_mat.append(0.5 * (push - pull))
        # compute mis-reg reconstructor
        meta_mat = np.asarray(meta_mat).T

        meta_rec = np.linalg.pinv(meta_mat)

        # scaling_factor
        if len(index_sprint) == 1:
            scaling = np.sum(
                np.squeeze(ref_wfs_signal) * np.squeeze(int_mat_exp)
            ) / np.sum(np.squeeze(ref_wfs_signal) * np.squeeze(ref_wfs_signal))
            estimated_mis_reg = meta_rec @ (
                (int_mat_exp * (1 / scaling) - ref_wfs_signal)
            )
        else:
            scaling = np.diag(ref_wfs_signal.T @ int_mat_exp) / np.diag(
                ref_wfs_signal.T @ ref_wfs_signal
            )
            estimated_mis_reg = meta_rec @ (
                (int_mat_exp @ np.diag(1 / scaling) - ref_wfs_signal).flatten()
            )

        # overwrite the mis-reg
        for i in range(len(arg_mis_reg)):
            if arg_mis_reg[i] == "magnification":
                setattr(misreg_id, "tangentialScaling", estimated_mis_reg[i])
                setattr(misreg_id, "radialScaling", estimated_mis_reg[i])
            else:
                setattr(misreg_id, arg_mis_reg[i], estimated_mis_reg[i])
        # update working point
        starting_point = starting_point + misreg_id
        mis_reg_estimate_buffer.append(np.asarray(estimated_mis_reg))

    mis_reg_estimate_buffer = np.asarray(mis_reg_estimate_buffer)

    misreg_out = misreg_id + starting_point
    print(misreg_out)

    return misreg_out, mis_reg_estimate_buffer


# functions to align the mask in order to equalize the flux per pupil; as done on the bench


def apply_mask_translation(wfs: BioEdge, translation_offset):
    """Translate the WFS masks by (dy, dx) pixels (before zero-padding scaling)."""
    dy, dx = (
        np.asarray(translation_offset, dtype=float) * wfs.zeroPaddingFactor
    )  # new array, no in-place edit
    masks = wfs.mask
    shifted_masks = [
        ndi.shift(np.asarray(m, dtype=complex), shift=(dy, dx), order=1, mode="nearest")
        for m in masks
    ]
    wfs.mask = (
        np.stack(shifted_masks) if isinstance(masks, np.ndarray) else shifted_masks
    )
    wfs.modulation = (
        wfs.modulation
    )  # re-apply modulation to refresh the reference signal


def get_flux_per_pupil(wfs: BioEdge):
    """
    Get the flux per pupil from the WFS reference signal.

    Parameters
    ----------
    wfs : BioEdge
        The WFS object.

    Returns
    -------
    flux_per_pupil : float
        The flux per pupil.
    """

    labels, n = ndi.label(wfs.valid_signal_2D)
    if n != 4:
        print(f"Warning: found {n} pupils, expected 4")

    ids = np.arange(1, n + 1)
    flux = ndi.sum_labels(wfs.referenceSignal_2D, labels, ids)

    return flux


def find_equalizing_translation(
    wfs: BioEdge, x0=(0.0, 0.0), max_shift=5.0, verbose=True
):
    """
    Find the mask translation (dy, dx) that equalizes the flux of the four pupils.

    Returns
    -------
    offset : np.ndarray, shape (2,)
        (dy, dx) in the same units apply_mask_translation expects.
        The wfs is left in the translated state.
    """
    original_mask = copy.deepcopy(wfs.mask)

    def flux_at(offset):
        # Always translate from the ORIGINAL mask: apply_mask_translation is cumulative
        wfs.mask = copy.deepcopy(original_mask)
        apply_mask_translation(wfs, offset)
        return np.asarray(get_flux_per_pupil(wfs), dtype=float)

    def residuals(offset):
        flux = flux_at(offset)
        return (
            flux / flux.mean() - 1.0
        )  # relative deviation from the mean, one per pupil

    flux_before = flux_at(np.zeros(2))

    sol = least_squares(
        residuals,
        x0=np.asarray(x0, dtype=float),
        bounds=(-max_shift, max_shift),
        diff_step=1e-2,
    )

    flux_after = flux_at(sol.x)  # leaves wfs in the optimized state

    if verbose:
        print(f"offset (dy, dx) = {sol.x}")
        print(
            f"flux before: {flux_before}  (rel. std {flux_before.std() / flux_before.mean():.3%})"
        )
        print(
            f"flux after : {flux_after}  (rel. std {flux_after.std() / flux_after.mean():.3%})"
        )

    return sol.x


# %% Define a class to store the numerical twin


@dataclass
class Twin:
    src: Source
    tel: Telescope
    wfs: BioEdge
    dm: DeformableMirror
    interaction_matrix_sim: np.ndarray
    valid_pixels_sim: np.ndarray
    reference_intensities_sim: np.ndarray
    modal_basis_sim: np.ndarray
    mis_reg: MisRegistration = None


# %% Build numerical twin


def build_numerical_twin(
    grey_width,
    valid_pixels_exp,
    modal_basis_exp,
    interaction_matrix_exp,
    stroke_rad,
    polarization_leakage_factor,
    sim_zero_padding_factor,
    sim_tel_resolution_factor,
    optical_band="R",
    magnitude=0.0,
    modulation=0.0,
    compute_misreg=False,
    orthogonalize_modal_basis=False,
    equalize_flux=False,
):
    """
    Build a numerical twin of the bench setup.

    Parameters
    ----------
    grey_width : float
        Grey width of the WFS mask in lambda/D.
    valid_pixels_exp : np.ndarray
        Valid pixels of the WFS mask from the bench.
    modal_basis_exp : np.ndarray
        Modal basis from the bench.
    interaction_matrix_exp : np.ndarray
        Interaction matrix from the bench.
    stroke_rad : float
        Stroke of the calibration in radians.
    polarization_leakage_factor : float
        Polarization leakage factor of the Bi-O edge mask.
    sim_zero_padding_factor : int
        Zero padding factor for the simulation.
    sim_tel_resolution_factor : int
        Telescope resolution factor for the simulation.
    optical_band : str
        Optical band of the source.
    magnitude : float
        Magnitude of the source.
    modulation : float
        Modulation of the WFS.
    compute_misreg : bool
        Whether to compute the mis-registration.
    orthogonalize_modal_basis : bool
        Whether to orthogonalize the modal basis.
    equalize_flux : bool
        Whether to equalize the flux of the four pupils by translating the WFS masks.
    """

    ######### Adjust simulation parameters ############

    wfs_resolution = (
        valid_pixels_exp.sum(axis=0).max() // 2
    )  # 2 pupils along each axis of the 2D plane

    wfs_resolution += wfs_resolution % 2  # force to be even

    # force valid_pixels_exp resolution to be even along both axis
    if valid_pixels_exp.shape[0] % 2 != 0 or valid_pixels_exp.shape[1] % 2 != 0:
        valid_pixels_exp = np.pad(
            valid_pixels_exp,
            ((valid_pixels_exp.shape[0] % 2, 0), (0, valid_pixels_exp.shape[1] % 2)),
        )

    # resolution for the FFT
    wfs_sim_cam_resolution = wfs_resolution * sim_zero_padding_factor

    npx_x = (wfs_sim_cam_resolution - valid_pixels_exp.shape[0]) // 2
    npx_y = (wfs_sim_cam_resolution - valid_pixels_exp.shape[1]) // 2

    if npx_x > 0:
        # pad the valid pupils extracted from the bench
        valid_pixels_sim = np.pad(valid_pixels_exp, ((npx_x, npx_x), (0, 0)))
    elif npx_x < 0:
        # pad the valid pupils extracted from the bench
        valid_pixels_sim = valid_pixels_exp[-npx_x:npx_x, :]

    if npx_y > 0:
        # pad the valid pupils extracted from the bench
        valid_pixels_sim = np.pad(valid_pixels_sim, ((0, 0), (npx_y, npx_y)))
    elif npx_y < 0:
        # pad the valid pupils extracted from the bench
        valid_pixels_sim = valid_pixels_sim[:, -npx_y:npx_y]

    if np.sum(valid_pixels_sim) != np.sum(valid_pixels_exp):
        raise (ValueError("The experimental pupil do not fit in the desired array"))

    ######### Build numerical twin ############

    # Build source
    src = Source(optical_band, magnitude=magnitude)

    # Telescope
    # consider a slightly padded pupil (avoid edge effects for the numerical twin)
    n_pixel_padded = 2
    tel = Telescope(
        resolution=wfs_resolution * sim_tel_resolution_factor - 2 * n_pixel_padded,
        diameter=2,
    )  # n_subp per tel diameter
    if n_pixel_padded > 0:
        tel.pad(n_pixel_padded)
    src**tel

    # Modal basis sim
    modal_basis_sim = -interpolate_cube(
        np.rot90(
            modal_basis_exp, k=1, axes=(1, 2)
        ),  # rotate the modal basis to match the orientation of the bench
        pixel_size_in=tel.D / modal_basis_exp.shape[1],
        pixel_size_out=tel.D / tel.resolution,
        resolution_out=tel.resolution,
        shape_out=[tel.resolution, tel.resolution],
        mis_registration=None,
        fliplr=True,
        flipud=False,
    )

    wfs: BioEdge

    wfs = BioEdge(
        nSubap=wfs_resolution,
        telescope=tel,
        modulation=modulation,
        lightRatio=0.1,  # no need to use it since we input the valid pixel map
        n_pix_edge=(wfs_sim_cam_resolution - 2 * wfs_resolution) // 4
        - wfs_resolution // 2,
        n_pix_separation=0,
        postProcessing="fullFrame_sum_flux",  # normalisation based on the flux on the detector (same as what is done on the bench)
        grey_width=grey_width,
        userValidSignal=(
            valid_pixels_sim if compute_misreg else None
        ),  # valid_pixels_sim,  # input user-define valid pixel map
        quadrants_numbering=[
            0,
            1,
            3,
            2,
        ],  # re-order the quadrants to match the experimental display
        polarization_leakage_factor=polarization_leakage_factor,
    )

    if compute_misreg:

        xs, ys = check_wfs_pupils(valid_pixels_sim, wfs, correct=True, n_it=2)

        wfs.modulation = modulation

        index_sprint = np.min((30, modal_basis_exp.shape[0] - 2))
        mis_reg, buffer = mini_sprint(
            ngs=src,
            tel=tel,
            wfs=wfs,
            modal_basis_exp=modal_basis_sim,  # 3D cube of the modal basis, n_modes * n_res*n_res
            index_sprint=[
                index_sprint
            ],  # mode index considered for sprint (the more the better... but the slower)
            int_mat_exp=interaction_matrix_exp,  # reference experimental interaction matrix
            n_iteration=20,  # number of iteration of SPRINT
            single_pass=False,  # single pass flag for the sensitivity matrices (False recommended)
            starting_point=MisRegistration(),  # mis-registration from which to start (updated at each iteration)
            arg_mis_reg=[
                "rotationAngle",
                "shiftX",
                "shiftY",
            ],  # list of parameter to be identified
        )

        # second pass =>  estimate magnification
        mis_reg, buffer = mini_sprint(
            ngs=src,
            tel=tel,
            wfs=wfs,
            modal_basis_exp=modal_basis_sim,  # 3D cube of the modal basis, n_modes * n_res*n_res
            index_sprint=[
                index_sprint
            ],  # mode index considered for sprint (the more the better... but the slower)
            int_mat_exp=interaction_matrix_exp,  # reference experimental interaction matrix
            n_iteration=20,  # number of iteration of SPRINT
            single_pass=False,  # single pass flag for the sensitivity matrices (False recommended)
            starting_point=mis_reg,  # mis-registration from which to start (updated at each iteration)
            arg_mis_reg=["magnification"],  # list of parameter to be identified
        )

        # update  modal basis with proper mis_registration, orthonormalise it and create modal dm
        modal_basis_sim = interpolate_cube(
            modal_basis_sim,
            pixel_size_in=tel.D / tel.resolution,
            pixel_size_out=tel.D / tel.resolution,
            resolution_out=tel.resolution,
            shape_out=[tel.resolution, tel.resolution],
            mis_registration=mis_reg,
            fliplr=False,
            flipud=False,
        )

    if equalize_flux:
        _ = find_equalizing_translation(wfs)

    if orthogonalize_modal_basis:
        modal_basis_sim_ortho = np.zeros_like(modal_basis_sim)
        modal_basis_sim_ortho_flat = modal_basis_sim[:, tel.pupil].T
        modal_basis_sim_ortho_flat, _ = np.linalg.qr(modal_basis_sim_ortho_flat)
        modal_basis_sim_ortho_flat -= modal_basis_sim_ortho_flat.mean(
            axis=0
        )  # zero mean
        modal_basis_sim_ortho_flat /= modal_basis_sim_ortho_flat.std(
            axis=0
        )  # unitary std
        modal_basis_sim_ortho[:, tel.pupil] = modal_basis_sim_ortho_flat.T
        modal_basis_sim = modal_basis_sim_ortho

    # Deformable mirror
    dm = DeformableMirror(
        tel,
        nSubap=modal_basis_sim.shape[1],
        modes=modal_basis_sim.reshape((modal_basis_sim.shape[0], -1)).T,
    )

    # interaction matrix
    stroke_m = stroke_rad * src.wavelength / (2 * np.pi)  # convert to meters
    sim_calib = InteractionMatrix(
        ngs=src,
        tel=tel,
        wfs=wfs,
        dm=dm,
        M2C=np.identity(
            dm.coefs.shape[0]
        ),  # full M2C => identity matrix since we consider a modal DM
        stroke=stroke_m,
        invert=False,
        display=True,
        single_pass=False,
    )
    interaction_matrix_sim = sim_calib.D * src.wavelength / (2 * np.pi)

    return Twin(
        src=src,
        tel=tel,
        wfs=wfs,
        dm=dm,
        interaction_matrix_sim=interaction_matrix_sim,
        valid_pixels_sim=valid_pixels_sim,
        reference_intensities_sim=wfs.referenceSignal,
        modal_basis_sim=modal_basis_sim,
        mis_reg=mis_reg if compute_misreg else None,
    )
