# This file contains a bunch of functions from Benjamin's lab2im package
# (it's just much lighter to import...)
import os,sys

import nibabel as nib
import numpy as np
import argparse

from scipy.interpolate import RegularGridInterpolator as rgi
from scipy.ndimage import gaussian_filter as gauss_filt
from scipy.ndimage import zoom


def load_volume(path_volume, im_only=True, squeeze=True, dtype=None, aff_ref=None):
    """
    Load volume file.
    :param path_volume: path of the volume to load. Can either be a nii, nii.gz, mgz, or npz format.
    If npz format, 1) the variable name is assumed to be 'vol_data',
    2) the volume is associated with a identity affine matrix and blank header.
    :param im_only: (optional) if False, the function also returns the affine matrix and header of the volume.
    :param squeeze: (optional) whether to squeeze the volume when loading.
    :param dtype: (optional) if not None, convert the loaded volume to this numpy dtype.
    :param aff_ref: (optional) If not None, the loaded volume is aligned to this affine matrix.
    The returned affine matrix is also given in this new space. Must be a numpy array of dimension 4x4.
    :return: the volume, with corresponding affine matrix and header if im_only is False.
    """
    assert path_volume.endswith(('.nii', '.nii.gz', '.mgz', '.npz')), 'Unknown data file: %s' % path_volume

    if path_volume.endswith(('.nii', '.nii.gz', '.mgz')):
        x = nib.load(path_volume)
        if squeeze:
            volume = np.squeeze(x.get_data())
        else:
            volume = x.get_data()
        aff = x.affine
        header = x.header
    else:  # npz
        volume = np.load(path_volume)['vol_data']
        if squeeze:
            volume = np.squeeze(volume)
        aff = np.eye(4)
        header = nib.Nifti1Header()
    if dtype is not None:
        volume = volume.astype(dtype=dtype)

    # align image to reference affine matrix
    if aff_ref is not None:
        from . import edit_volumes  # the import is done here to avoid import loops
        n_dims, _ = get_dims(list(volume.shape), max_channels=10)
        volume, aff = edit_volumes.align_volume_to_ref(volume, aff, aff_ref=aff_ref, return_aff=True, n_dims=n_dims)

    if im_only:
        return volume
    else:
        return volume, aff, header


def save_volume(volume, aff, header, path, res=None, dtype=None, n_dims=3):
    """
    Save a volume.
    :param volume: volume to save
    :param aff: affine matrix of the volume to save. If aff is None, the volume is saved with an identity affine matrix.
    aff can also be set to 'FS', in which case the volume is saved with the affine matrix of FreeSurfer outputs.
    :param header: header of the volume to save. If None, the volume is saved with a blank header.
    :param path: path where to save the volume.
    :param res: (optional) update the resolution in the header before saving the volume.
    :param dtype: (optional) numpy dtype for the saved volume.
    :param n_dims: (optional) number of dimensions, to avoid confusion in multi-channel case. Default is None, where
    n_dims is automatically inferred.
    """
    if dtype is not None:
        volume = volume.astype(dtype=dtype)
    if '.npz' in path:
        np.savez_compressed(path, vol_data=volume)
    else:
        if header is None:
            header = nib.Nifti1Header()
        if isinstance(aff, str):
            if aff == 'FS':
                aff = np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
        elif aff is None:
            aff = np.eye(4)
        nifty = nib.Nifti1Image(volume, aff, header)
        if res is not None:
            if n_dims is None:
                n_dims, _ = get_dims(volume.shape)
            res = reformat_to_list(res, length=n_dims, dtype=None)
            nifty.header.set_zooms(res)
        nib.save(nifty, path)

def get_dims(shape, max_channels=10):
    """Get the number of dimensions and channels from the shape of an array.
    The number of dimensions is assumed to be the length of the shape, as long as the shape of the last dimension is
    inferior or equal to max_channels (default 3).
    :param shape: shape of an array. Can be a sequence or a 1d numpy array.
    :param max_channels: maximum possible number of channels.
    :return: the number of dimensions and channels associated with the provided shape.
    example 1: get_dims([150, 150, 150], max_channels=10) = (3, 1)
    example 2: get_dims([150, 150, 150, 3], max_channels=10) = (3, 3)
    example 3: get_dims([150, 150, 150, 15], max_channels=10) = (4, 1), because 5>3"""
    if shape[-1] <= max_channels:
        n_dims = len(shape) - 1
        n_channels = shape[-1]
    else:
        n_dims = len(shape)
        n_channels = 1
    return n_dims, n_channels

def reformat_to_list(var, length=None, load_as_numpy=False, dtype=None):
    """This function takes a variable and reformat it into a list of desired
    length and type (int, float, bool, str).
    If variable is a string, and load_as_numpy is True, it will be loaded as a numpy array.
    If variable is None, this funtion returns None.
    :param var: a str, int, float, list, tuple, or numpy array
    :param length: (optional) if var is a single item, it will be replicated to a list of this length
    :param load_as_numpy: (optional) whether var is the path to a numpy array
    :param dtype: (optional) convert all item to this type. Can be 'int', 'float', 'bool', or 'str'
    :return: reformated list
    """

    # convert to list
    if var is None:
        return None
    var = load_array_if_path(var, load_as_numpy=load_as_numpy)
    if isinstance(var, (int, float, np.int, np.int32, np.int64, np.float, np.float32, np.float64)):
        var = [var]
    elif isinstance(var, tuple):
        var = list(var)
    elif isinstance(var, np.ndarray):
        var = np.squeeze(var).tolist()
    elif isinstance(var, str):
        var = [var]
    elif isinstance(var, bool):
        var = [var]
    if isinstance(var, list):
        if length is not None:
            if len(var) == 1:
                var = var * length
            elif len(var) != length:
                raise ValueError('if var is a list/tuple/numpy array, it should be of length 1 or {0}, '
                                 'had {1}'.format(length, var))
    else:
        raise TypeError('var should be an int, float, tuple, list, numpy array, or path to numpy array')

    # convert items type
    if dtype is not None:
        if dtype == 'int':
            var = [int(v) for v in var]
        elif dtype == 'float':
            var = [float(v) for v in var]
        elif dtype == 'bool':
            var = [bool(v) for v in var]
        elif dtype == 'str':
            var = [str(v) for v in var]
        else:
            raise ValueError("dtype should be 'str', 'float', 'int', or 'bool'; had {}".format(dtype))
    return var

def print_no_newline(string):
    sys.stdout.write(string)
    sys.stdout.flush()

def parse_args_unet_predict():

    parser = argparse.ArgumentParser(description="Runs U-net white matter bundle prediction for brainstem WM")

    parser.add_argument('-m','--model_file', help="UNet model file path", type=str, required=True)
    parser.add_argument('-o','--output_path', help="Output directory for saving segmentations", type=str, required=True)
    parser.add_argument('-lowb','--lowb_file', help="Low-b volume file path", type=str, required=True)
    parser.add_argument('-fa','--fa_file', help="FA volume file path", type=str, required=True)
    parser.add_argument('-tract','--tract_file', help="RGB tract file path", type=str, required=True)
    parser.add_argument('-llp','--label_list_path', help="Path to numpy array containing label values", type=str, required=True)
    parser.add_argument('-r','--resolution', help="Resolution of the input segmentations", type=float, default=1.0, required=False)
    parser.add_argument('-gm','--generator_mode', help="Which generator to use", type=str, default='rgb', required=False)
    parser.add_argument('-fc','--unet_feat_count', help="Number of features in UNet base level", type=int, default=24, required=False)
    parser.add_argument('-nl','--n_levels', help="Number of levels", type=int, default=5, required=False)
    parser.add_argument('-cs','--conv_size', help="Size of convolution kernel", type=int, default=3, required=False)
    parser.add_argument('-fm','--feat_multiplier', help="Feature multiplier per level", type=int, default=2, required=False)
    parser.add_argument('-cpl','--nb_conv_per_level', help="Number of convolutions per unet layer level", type=int, default=2, required=False)
    parser.add_argument('-a','--activation', help="Type of activation function, can be [elu] or [relu]", type=str, default='elu', required=False)
    parser.add_argument('-bb','--bounding_box_width', help="Bounding box width around the brainstem", type=int, default=64, required=False)
    parser.add_argument('-ar','--aff_ref', help="Feature multiplier per level", type=int, default=np.eye(4), required=False)

    return parser.parse_args()


def load_array_if_path(var, load_as_numpy=True):
    """If var is a string and load_as_numpy is True, this function loads the array writen at the path indicated by var.
    Otherwise it simply returns var as it is."""
    if (isinstance(var, str)) & load_as_numpy:
        assert os.path.isfile(var), 'No such path: %s' % var
        var = np.load(var)
    return var

def align_volume_to_ref(volume, aff, aff_ref=None, return_aff=False, n_dims=None):
    """This function aligns a volume to a reference orientation (axis and direction) specified by an affine matrix.
    :param volume: a numpy array
    :param aff: affine matrix of the floating volume
    :param aff_ref: (optional) affine matrix of the target orientation. Default is identity matrix.
    :param return_aff: (optional) whether to return the affine matrix of the aligned volume
    :param n_dims: number of dimensions (excluding channels) of the volume corresponding to the provided affine matrix.
    :return: aligned volume, with corresponding affine matrix if return_aff is True.
    """

    # work on copy
    aff_flo = aff.copy()

    # default value for aff_ref
    if aff_ref is None:
        aff_ref = np.eye(4)

    # extract ras axes
    if n_dims is None:
        n_dims, _ = get_dims(volume.shape)
    ras_axes_ref = get_ras_axes(aff_ref, n_dims=n_dims)
    ras_axes_flo = get_ras_axes(aff_flo, n_dims=n_dims)

    # align axes
    aff_flo[:, ras_axes_ref] = aff_flo[:, ras_axes_flo]
    for i in range(n_dims):
        if ras_axes_flo[i] != ras_axes_ref[i]:
            volume = np.swapaxes(volume, ras_axes_flo[i], ras_axes_ref[i])
            swapped_axis_idx = np.where(ras_axes_flo == ras_axes_ref[i])
            ras_axes_flo[swapped_axis_idx], ras_axes_flo[i] = ras_axes_flo[i], ras_axes_flo[swapped_axis_idx]

    # align directions
    dot_products = np.sum(aff_flo[:3, :3] * aff_ref[:3, :3], axis=0)
    for i in range(n_dims):
        if dot_products[i] < 0:
            volume = np.flip(volume, axis=i)
            aff_flo[:, i] = - aff_flo[:, i]
            aff_flo[:3, 3] = aff_flo[:3, 3] - aff_flo[:3, i] * (volume.shape[i] - 1)

    if return_aff:
        return volume, aff_flo
    else:
        return volume

def get_ras_axes(aff, n_dims=3):
    """This function finds the RAS axes corresponding to each dimension of a volume, based on its affine matrix.
    :param aff: affine matrix Can be a 2d numpy array of size n_dims*n_dims, n_dims+1*n_dims+1, or n_dims*n_dims+1.
    :param n_dims: number of dimensions (excluding channels) of the volume corresponding to the provided affine matrix.
    :return: two numpy 1d arrays of lengtn n_dims, one with the axes corresponding to RAS orientations,
    and one with their corresponding direction.
    """
    aff_inverted = np.linalg.inv(aff)
    img_ras_axes = np.argmax(np.absolute(aff_inverted[0:n_dims, 0:n_dims]), axis=0)
    return img_ras_axes




def get_dims(shape, max_channels=10):
    """Get the number of dimensions and channels from the shape of an array.
    The number of dimensions is assumed to be the length of the shape, as long as the shape of the last dimension is
    inferior or equal to max_channels (default 3).
    :param shape: shape of an array. Can be a sequence or a 1d numpy array.
    :param max_channels: maximum possible number of channels.
    :return: the number of dimensions and channels associated with the provided shape.
    example 1: get_dims([150, 150, 150], max_channels=10) = (3, 1)
    example 2: get_dims([150, 150, 150, 3], max_channels=10) = (3, 3)
    example 3: get_dims([150, 150, 150, 15], max_channels=10) = (4, 1), because 5>3"""
    if shape[-1] <= max_channels:
        n_dims = len(shape) - 1
        n_channels = shape[-1]
    else:
        n_dims = len(shape)
        n_channels = 1
    return n_dims, n_channels

def resample_like(vol_ref, aff_ref, vol_flo, aff_flo, method='linear'):
    """This function reslices a floating image to the space of a reference image
    :param vol_res: a numpy array with the reference volume
    :param aff_ref: affine matrix of the reference volume
    :param vol_flo: a numpy array with the floating volume
    :param aff_flo: affine matrix of the floating volume
    :param method: linear or nearest
    :return: resliced volume
    """

    T = np.matmul(np.linalg.inv(aff_flo), aff_ref)

    xf = np.arange(0, vol_flo.shape[0])
    yf = np.arange(0, vol_flo.shape[1])
    zf = np.arange(0, vol_flo.shape[2])

    my_interpolating_function = rgi((xf, yf, zf), vol_flo, method=method, bounds_error=False, fill_value=0.0)

    xr = np.arange(0, vol_ref.shape[0])
    yr = np.arange(0, vol_ref.shape[1])
    zr = np.arange(0, vol_ref.shape[2])

    xrg, yrg, zrg = np.meshgrid(xr, yr, zr, indexing='ij', sparse=False)
    n = xrg.size
    xrg = xrg.reshape([n])
    yrg = yrg.reshape([n])
    zrg = zrg.reshape([n])
    bottom = np.ones_like(xrg)
    coords = np.stack([xrg, yrg, zrg, bottom])
    coords_new = np.matmul(T, coords)[:-1, :]
    result = my_interpolating_function((coords_new[0, :], coords_new[1, :], coords_new[2, :]))

    if vol_ref.size == result.size:
        return result.reshape(vol_ref.shape)
    else:
        return result.reshape([*vol_ref.shape, vol_flo.shape[-1]])



# Creates a 3x3 rotation matrix from a vector with 3 rotations about x, y, and z
def make_rotation_matrix(rot):
    Rx = np.array([[1, 0, 0], [0, np.cos(rot[0]), -np.sin(rot[0])], [0, np.sin(rot[0]), np.cos(rot[0])]])
    Ry = np.array([[np.cos(rot[1]), 0, np.sin(rot[1])], [0, 1, 0], [-np.sin(rot[1]), 0, np.cos(rot[1])]])
    Rz = np.array([[np.cos(rot[2]), -np.sin(rot[2]), 0], [np.sin(rot[2]), np.cos(rot[2]), 0], [0, 0, 1]])
    R = np.matmul(np.matmul(Rx, Ry), Rz)
    return R

# Computer linear (flattened) indices for linear interpolation of a volume of size nx x ny x nz at locations xx, yy, zz
# (as well as a boolean vector 'ok' telling which indices are inbounds)
# Note that it doesn't support sparse xx, yy, zz
def nn_interpolator_indices(xx, yy, zz, nx, ny, nz):
    xx2r = np.round(xx).astype(int)
    yy2r = np.round(yy).astype(int)
    zz2r = np.round(zz).astype(int)
    ok = (xx2r >= 0) & (yy2r >= 0) & (zz2r >= 0) & (xx2r <= nx - 1) & (yy2r <= ny - 1) & (zz2r <= nz - 1)
    idx = xx2r[ok] + nx * yy2r[ok] + nx * ny * zz2r[ok]
    return idx, ok

# Similar to nn_interpolator_indices but does not check out of bounds.
# Note that it *does* support sparse xx, yy, zz
def nn_interpolator_indices_nocheck(xx, yy, zz, nx, ny, nz):
    xx2r = np.round(xx).astype(int)
    yy2r = np.round(yy).astype(int)
    zz2r = np.round(zz).astype(int)
    idx = xx2r + nx * yy2r + nx * ny * zz2r
    return idx

# Subsamples a volume by a given ration in each dimension.
# It carefully accounts for origin shifts
def subsample(X, ratio, size, method='linear', return_locations=False):
    xi = np.arange(0.5 * (ratio[0] - 1.0), size[0] - 1 + 1e-6, ratio[0])
    yi = np.arange(0.5 * (ratio[1] - 1.0), size[1] - 1 + 1e-6, ratio[1])
    zi = np.arange(0.5 * (ratio[2] - 1.0), size[2] - 1 + 1e-6, ratio[2])
    xig, yig, zig = np.meshgrid(xi, yi, zi, indexing='ij', sparse=True)
    interpolator = rgi((range(size[0]), range(size[1]), range(size[2])), X, method=method)
    Y = interpolator((xig, yig, zig))
    if return_locations:
        return Y, xig, yig, zig
    else:
        return Y


def upsample(X, ratio, size, method='linear', return_locations=False):
    start = (1.0 - ratio[0]) / (2.0 * ratio[0])
    inc = 1.0 / ratio[0]
    end = start + inc * size[0] - 1e-6
    xi = np.arange(start, end, inc)
    xi[xi < 0] = 0
    xi[xi > X.shape[0] - 1] = X.shape[0] - 1

    start = (1.0 - ratio[1]) / (2.0 * ratio[1])
    inc = 1.0 / ratio[1]
    end = start + inc * size[1] - 1e-6
    yi = np.arange(start, end, inc)
    yi[yi < 0] = 0
    yi[yi > X.shape[1] - 1] = X.shape[1] - 1

    start = (1.0 - ratio[2]) / (2.0 * ratio[2])
    inc = 1.0 / ratio[2]
    end = start + inc * size[2] - 1e-6
    zi = np.arange(start, end, inc)
    zi[zi < 0] = 0
    zi[zi > X.shape[2] - 1] = X.shape[2] - 1

    xig, yig, zig = np.meshgrid(xi, yi, zi, indexing='ij', sparse=True)
    interpolator = rgi((range(X.shape[0]), range(X.shape[1]), range(X.shape[2])), X, method=method)
    Y = interpolator((xig, yig, zig))

    if return_locations:
        return Y, xig, yig, zig
    else:
        return Y

# Augmentation of FA with gaussian noise and gamma transform
def augment_fa(X, gamma_std, max_noise_std_fa):
    gamma_fa = np.exp(gamma_std * np.random.randn(1)[0])
    noise_std = max_noise_std_fa * np.random.rand(1)[0]
    Y = X + noise_std * np.random.randn(*X.shape)
    Y[Y < 0] = 0
    Y[Y > 1] = 1
    Y = Y ** gamma_fa
    return Y

# Augmentation of T1 intensities with random contrast, brightness, gamma, and gaussian noise
def augment_t1(X, gamma_std, contrast_std, brightness_std, max_noise_std):
    # TODO: maybe add bias field? If we're working with FreeSurfer processed images maybe it's not too important
    gamma_t1 = np.exp(gamma_std * np.random.randn(1)[0])  # TODO: maybe make it spatially variable?
    contrast = np.min((1.4, np.max((0.6, 1.0 + contrast_std * np.random.randn(1)[0]))))
    brightness = np.min((0.4, np.max((-0.4, brightness_std * np.random.randn(1)[0]))))
    noise_std = max_noise_std * np.random.rand(1)[0]
    Y = ((X - 0.5) * contrast + (0.5 + brightness)) + noise_std * np.random.randn(*X.shape)
    Y[Y < 0] = 0
    Y[Y > 1] = 1
    Y = Y ** gamma_t1
    return Y


def rescale_voxel_size(volume, aff, new_vox_size):
    """This function resizes the voxels of a volume to a new provided size, while adjusting the header to keep the RAS
    :param volume: a numpy array
    :param aff: affine matrix of the volume
    :param new_vox_size: new voxel size (3 - element numpy vector) in mm
    :return: new volume and affine matrix
    """

    pixdim = np.sqrt(np.sum(aff * aff, axis=0))[:-1]
    new_vox_size = np.array(new_vox_size)
    factor = pixdim / new_vox_size
    sigmas = 0.25 / factor
    sigmas[factor > 1] = 0  # don't blur if upsampling

    volume_filt = gauss_filt(volume, sigmas)

    # volume2 = zoom(volume_filt, factor, order=1, mode='reflect', prefilter=False)
    x = np.arange(0, volume_filt.shape[0])
    y = np.arange(0, volume_filt.shape[1])
    z = np.arange(0, volume_filt.shape[2])

    my_interpolating_function = rgi((x, y, z), volume_filt)

    start = - (factor - 1) / (2 * factor)
    step = 1.0 / factor
    stop = start + step * np.ceil(volume_filt.shape * factor)

    xi = np.arange(start=start[0], stop=stop[0], step=step[0])
    yi = np.arange(start=start[1], stop=stop[1], step=step[1])
    zi = np.arange(start=start[2], stop=stop[2], step=step[2])
    xi[xi < 0] = 0
    yi[yi < 0] = 0
    zi[zi < 0] = 0
    xi[xi > (volume_filt.shape[0] - 1)] = volume_filt.shape[0] - 1
    yi[yi > (volume_filt.shape[1] - 1)] = volume_filt.shape[1] - 1
    zi[zi > (volume_filt.shape[2] - 1)] = volume_filt.shape[2] - 1

    xig, yig, zig = np.meshgrid(xi, yi, zi, indexing='ij', sparse=True)
    volume2 = my_interpolating_function((xig, yig, zig))

    aff2 = aff.copy()
    for c in range(3):
        aff2[:-1, c] = aff2[:-1, c] / factor[c]
    aff2[:-1, -1] = aff2[:-1, -1] - np.matmul(aff2[:-1, :-1], 0.5 * (factor - 1))

    return volume2, aff2


def downsample_upsample_scipy(volume, downsampling_factor, original_shape):
    # Downsample
    downsampled_volume = zoom(volume, zoom=downsampling_factor, order=1, prefilter=False)
    # Upsample
    upsampled_volume = zoom(downsampled_volume, zoom=[o/d for o, d in zip(original_shape, downsampled_volume.shape)], order=1, prefilter=False)

    return upsampled_volume
