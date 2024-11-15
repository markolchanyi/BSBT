import os
import json
import numpy as np
import nibabel as nib
from scipy import ndimage


def count_shells(dwi_json_path):
    '''
    Count the number of shells in the DWI data
    '''
    f = open(dwi_json_path)
    dw_head = json.load(f)

    shell_list = []
    for enc in dw_head["keyval"]["dw_scheme"]:
        shell_mag_val_true = enc[3]
        ## shells in some data are +- 50 of ideal shell value..this is just a dirty way to
        ## round them off
        shell_mag_val_rounded = int(round(shell_mag_val_true/500)*500)
        shell_list.append(shell_mag_val_rounded)
    n_shells = len((set(shell_list)))
    print("Diffusion data contains " + str(n_shells) + " unique shell values...shell vals are: " + str(list(set(shell_list))))
    return n_shells


def get_header_resolution(dwi_json_path):
    '''
    Extract the spatial resolution of the DWI data from its associated JSON
    '''
    f = open(dwi_json_path)
    dw_head = json.load(f)
    return dw_head['spacing'][0]


def write_volinfo(json_file_path, volinfo_path):
    vox_resolution = get_header_resolution(json_file_path)
    shell_count = count_shells(json_file_path)
    single_shell = shell_count <= 2

    # Write results to volinfo.txt
    with open(volinfo_path, "w") as volinfo:
        volinfo.write(f"resolution: {vox_resolution} mm\n")
        volinfo.write(f"number of shells: {shell_count-1}\n")
        volinfo.write(f"single_shell_mode: {single_shell}\n")


def extract_bin_roi(input_file, labels, output_file):
    '''
    Extract an ROI from a segmentation volume.
    '''

    img = nib.load(input_file)
    data = img.get_fdata()

    labels_set = set(labels)  # Convert to set for faster lookup
    binary_mask = np.isin(data, labels_set).astype(np.uint8)

    binary_img = nib.Nifti1Image(binary_mask, affine=img.affine, header=img.header)
    nib.save(binary_img, output_file)



def crop_around_centroid(vol_path,mask_vol_path,output_path,crop_size=64):
    '''
    Crop a volume around the centroid of a binary lask
    '''
    vol_img = nib.load(vol_path)
    vol = vol_img.get_fdata()
    affine = vol_img.affine
    mask_vol_img = nib.load(mask_vol_path)
    mask_vol = mask_vol_img.get_fdata()

    COM = ndimage.measurements.center_of_mass(mask_vol)
    COM = np.round(COM).astype(int)

    # Define the cube dimensions
    cube_size = int(crop_size)
    half_size = int(cube_size/2)
    half_size = (np.ones_like(COM)*half_size).astype(int)

    # Calculate the bounds for cropping
    start = np.maximum(COM - half_size, np.zeros(3, dtype=int))
    end = np.minimum(start + cube_size, np.array(mask_vol.shape, dtype=int))

    start = end - cube_size
    if vol.ndim == 3:
        cropped_vol = vol[start[0]:end[0],start[1]:end[1],start[2]:end[2]]
    else:
        cropped_vol = vol[start[0]:end[0],start[1]:end[1],start[2]:end[2],:]

    # Adjust the affine matrix to retain same RAS coordinates
    new_affine = affine.copy()
    new_affine[:3, 3] += np.dot(affine[:3, :3], start)

    output_img = nib.Nifti1Image(cropped_vol, new_affine, vol_img.header)
    nib.save(output_img, output_path)


def tractography_mask(template_vol_path,output_path):
    '''
    Create a tractography mask around the brainstem for PFM creation
    '''
    template_vol_img = nib.load(template_vol_path)
    template_vol = template_vol_img.get_fdata()
    aff = template_vol_img.affine

    template_vol = np.asarray(template_vol, dtype=np.int32)
    mask_vol = np.zeros_like(template_vol,dtype=np.int32)

    label_coords=np.argwhere(template_vol==1)

    min_point=np.min(label_coords[:],axis=0)
    max_point=np.max(label_coords[:],axis=0)

    mask_vol[min_point[0]:max_point[0],min_point[1]:max_point[1],min_point[2]:max_point[2]] = 1

    output_img = nib.Nifti1Image(mask_vol, aff, template_vol_img.header)
    nib.save(output_img, output_path)
