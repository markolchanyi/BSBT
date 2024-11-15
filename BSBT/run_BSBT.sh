#!/bin/bash

# Description:
# This is the main run script for BSBT pipeline
# It combines all necessary preprocessing, ROI extraction,
#
# Usage:
#   ./run_bsbt.sh <dwi> <bvals> <bvecs> <threads>
#
# Positional args:
#   1. <dwi>         - Directory containing `input_dwi.nii.gz` (required)
#   2. <bvals>       - Directory containing `input_dwi.nii.gz` (required)
#   3. <bvecs>       - Directory containing `input_dwi.nii.gz` (required)
#   4. <out>         - Path for intermed (optional; default: current working dir)
#   5. <threads>     - Number of threads to use for MrTrix (optional; default: 1)

if [ -f "$1" ] && [ -r "$1" ]; then
  echo "Error: Input DWI does not exist or is not readable."
  echo "Usage: ./run_bsbt.sh <dwi> <bvals> <bvecs> <threads>"
  exit 1
fi
if [ -f "$2" ] && [ -r "$2" ]; then
  echo "Error: bvals does not exist or is not readable."
  echo "Usage: ./run_bsbt.sh <dwi> <bvals> <bvecs> <threads>"
  exit 1
fi
if [ -f "$3" ] && [ -r "$3" ]; then
  echo "Error: bvecs does not exist or is not readable."
  echo "Usage: ./run_bsbt.sh <dwi> <bvals> <bvecs> <threads>"
  exit 1
fi
if [ -z "$4" ]; then
  OUTPUT_DIR=$(pwd)
else
  OUTPUT_DIR="$4"
fi

# Ensure FS and MrTrix DEPENDENCIES found
if [ -z "$FREESURFER_HOME" ]; then
    echo "Error: FREESURFER_HOME is not set. Please make sure to source Freesurfer."
    exit 1
fi
if ! command -v mrconvert &> /dev/null; then
    echo "Error: MRtrix not found. Please ensure it is installed."
    exit 1
fi

finp="$1"
fbval="$2"
fbvec="$3"
THREADS="${5:-1}"  # Default to singgle thread

# Dir for all intermediate files. To be deleted
INTERDIR=$(mktemp -d $OUTPUT_DIR/XXXXXXX)
echo "Created intermediate dir: ${INTERDIR}"
echo "Moving relevent files to: ${INTERDIR}"

rsync -av ${finp} "${INTERDIR}/input_dwi.nii.gz"
rsync -av ${fbval} "${INTERDIR}/bvals.txt"
rsync -av ${fbvec} "${INTERDIR}/bvecs.txt"

# ALL PREPROCESSING FOR TRACTOGRAPHY
LOGFILE="${INTERDIR}/log.txt"
./trackgen/fs_preprocc_extract.sh ${INTERDIR} ${LOGFILE} ${THREADS}

echo "All done!!"
