#!/bin/bash

# Description:
# This script extracts the mean low-b from the input DWI and performed SynthSeg
# and brainstem-subregion ROI extraction for tractography seeding.
#
# Usage:
#   ./freesurfer_preprocess.sh <input_dir>
#
# Positional args:
#   1. <input_dir>   - Directory containing `input_dwi.nii.gz` (required)
#   2. <log_file>    - log file (optional)
#   3. <threads>     - Number of threads to use for MrTrix (optional; default: 1)

run_cmd() {
  local cmd="$1"
  echo "$cmd" | tee -a "$log_file"
  eval "$cmd" &>> "$log_file"
}

if [ -z "$1" ]; then
  echo "Error: Input directory is required."
  echo "Usage: ./freesurfer_preprocess.sh <input_dir>"
  exit 1
fi

log_file="${2:-$volume_directory/preprocessing_log_$(date +%Y%m%d%H%M%S).log}"
threads="${3:-1}"  # Default to singgle thread

# Ensure FREESURFER and MrTrix libs are found
if [ -z "$FREESURFER_HOME" ]; then
    echo "Error: FREESURFER_HOME is not set. Please make sure to source Freesurfer."
    exit 1
fi

if ! command -v mrconvert &> /dev/null; then
    echo "Error: MRtrix not found. Please ensure it is installed."
    exit 1
fi

input_directory="$1"
# Check if input dir and log dirs exist. If log dir doesnt exist, then make it..
if [ ! -d "$input_directory" ]; then
  echo "Error: Directory $input_directory does not exist."
  exit 1
fi

inp="$input_directory/input_dwi.nii.gz"
fbval="$input_directory/bvals.txt"
fbvec="$input_directory/bvecs.txt"

header_json="$input_directory/header.json"
volinfo_txt="$input_directory/volinfo.txt"
run_cmd "touch ${volinfo_txt}"

if [ ! -f "$inp" ]; then
  echo "Error: $inp not found. Please ensure it is in the specified directory."
  exit 1
fi

if [ ! -f "$fbval" ]; then
  echo "Error: $fbval not found. Please ensure it is in the specified directory."
  exit 1
fi

if [ ! -f "$fbvec" ]; then
  echo "Error: $fbvec not found. Please ensure it is in the specified directory."
  exit 1
fi

# same for internal bs_trackgen commands
output_dir="$input_directory"

###############################################################################
#*************            PREPROCESSING COMMANDS                **************#
###############################################################################

echo "Preprocessing DWI in dir: $input_directory" | tee -a "$log_file"

# create DWI info files
run_cmd "touch \"$header_json\" \"$volinfo_txt\""

inpmif="$input_directory/input_dwi.mif"
run_cmd "mrconvert ${inp} ${inpmif} -fslgrad ${fbvec} ${fbval} -nthreads ${threads} -force"

# Run mrinfo to populate header.json with MRtrix header information
run_cmd "mrinfo -json_all ${header_json} ${inpmif} -force"

# populate volinfo.txt
run_cmd "python -c \"from utils import write_volinfo; write_volinfo('${header_json}','${volinfo_txt}')\""


# MRtrix processing steps
echo "Regridding to 1mm..."

echo "done preprocessing!!"
echo " "
echo " "
