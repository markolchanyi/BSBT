import os
import json

'''
Count the number of shells in the DWI data
'''
def count_shells(dwi_json_path):
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

'''
Extract the spatial resolution of the DWI data from its associated JSON
'''
def get_header_resolution(dwi_json_path):
    f = open(dwi_json_path)
    dw_head = json.load(f)
    return dw_head['spacing'][0]


def write_volinfo(json_file_path, output_dir):
    vox_resolution = get_header_resolution(json_file_path)
    shell_count = count_shells(header_json_path)
    single_shell = shell_count <= 2

    # Write results to volinfo.txt
    volinfo_path = os.path.join(output_dir, "volinfo.txt")
    with open(volinfo_path, "w") as volinfo:
        volinfo.write(f"resolution: {vox_resolution} mm\n")
        volinfo.write(f"number of shells: {shell_count} mm\n")
        volinfo.write(f"single_shell_mode: {single_shell}\n")
