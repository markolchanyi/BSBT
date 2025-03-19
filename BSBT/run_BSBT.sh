#!/bin/bash

# Description:
# This is the main run script for BSBT pipeline
# It combines all necessary preprocessing, ROI extraction, etc.

# Usage:
#   ./run_bsbt.sh --dwi <dwi.nii.gz> --bvals <bvals.txt> --bvecs <bvecs.txt> [--out <outdir>] [--threads <num_threads>]

# Flags:
#   --dwi <file>       - Path to input DWI file (required)
#   --bvals <file>     - Path to bvals file (required)
#   --bvecs <file>     - Path to bvecs file (required)
#   --out <directory>  - Output directory (optional; default: current working dir)
#   --threads <int>    - Number of threads to use for MrTrix (optional; default: 1)

# Parse arguments
DWI=""
BVALS=""
BVECS=""
OUTPUT_DIR=""
THREADS=""

while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --dwi)
            DWI="$2"
            shift # past argument
            shift # past value
            ;;
        --bvals)
            BVALS="$2"
            shift
            shift
            ;;
        --bvecs)
            BVECS="$2"
            shift
            shift
            ;;
        --out)
            OUTPUT_DIR="$2"
            shift
            shift
            ;;
        --threads)
            THREADS="$2"
            shift
            shift
            ;;
        *)    # unknown option
            echo "Unknown flag: $1"
            echo "Usage: ./run_bsbt.sh --dwi <dwi> --bvals <bvals> --bvecs <bvecs> [--out <outdir>] [--threads <num_threads>]"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "$DWI" ] || [ -z "$BVALS" ] || [ -z "$BVECS" ]; then
    echo "Error: Missing required arguments."
    echo "Usage: ./run_bsbt.sh --dwi <dwi> --bvals <bvals> --bvecs <bvecs> [--out <outdir>] [--threads <num_threads>]"
    exit 1
fi

# Validate that input files exist and are readable
if [ ! -f "$DWI" ] || [ ! -r "$DWI" ]; then
    echo "Error: DWI file does not exist or is not readable."
    exit 1
fi
if [ ! -f "$BVALS" ] || [ ! -r "$BVALS" ]; then
    echo "Error: bvals file does not exist or is not readable."
    exit 1
fi
if [ ! -f "$BVECS" ] || [ ! -r "$BVECS" ]; then
    echo "Error: bvecs file does not exist or is not readable."
    exit 1
fi

# Set defaults for optional arguments
if [ -z "$OUTPUT_DIR" ]; then
    OUTPUT_DIR=$(pwd)
fi
if [ -z "$THREADS" ]; then
    THREADS=1
fi

echo "Saving results to: ${OUTPUT_DIR}"

# Ensure FS and MrTrix DEPENDENCIES found
if [ -z "$FREESURFER_HOME" ]; then
    echo "Error: FREESURFER_HOME is not set. Please make sure to source Freesurfer."
    exit 1
fi
if ! command -v mrconvert &> /dev/null; then
    echo "Error: MRtrix not found. Please ensure it is installed."
    exit 1
fi

finp="$DWI"
fbval="$BVALS"
fbvec="$BVECS"

# Create intermediate directory
TEMPNAME="temp_$(uuidgen | tr -d '-' | head -c 7)"
INTERDIR=${OUTPUT_DIR}/${TEMPNAME}
mkdir -p ${INTERDIR}
echo "Created intermediate dir: ${INTERDIR}"
echo "Moving relevant files to: ${INTERDIR}"

rsync -av ${finp} "${INTERDIR}/input_dwi.nii.gz"
rsync -av ${fbval} "${INTERDIR}/bvals.txt"
rsync -av ${fbvec} "${INTERDIR}/bvecs.txt"

# add temp file name for ref
LOGFILE="${OUTPUT_DIR}/log.txt"
echo ${TEMPNAME} >> ${LOGFILE}
echo "^^^ log for intermediate files ^^^"

# FSL/Mrtrix preprocess
./trackgen/fs_preprocc_extract.sh ${INTERDIR} ${LOGFILE} ${THREADS}


# Generate the PFM
./trackgen/pfmgen.sh ${INTERDIR} ${LOGFILE} ${OUTPUT_DIR} ${THREADS}


# Run the unet + CRF wrapper
MODELFILE="../model_weights/dice_480.h5"
LABLIST="../model_weights/brainstem_wm_label_list.npy"
OUTPUTPATH=${OUTPUT_DIR}
LOWBFILE=${OUTPUT_DIR}/lowb_1mm_cropped_norm.nii.gz
FAFILE=${OUTPUT_DIR}/fa_1mm_cropped_norm.nii.gz
PFMFILE=${OUTPUT_DIR}/tracts_concatenated_1mm_cropped_norm.nii.gz


if [ ! -f "$LOWBFILE" ] || [ ! -s "$LOWBFILE" ]; then
    echo "${LOWBFILE} either doesn't exist or is empty."
    exit 1
fi

if [ ! -f "$FAFILE" ] || [ ! -s "$FAFILE" ]; then
    echo "${FAFILE} either doesn't exist or is empty."
    exit 1
fi

if [ ! -f "$PFMFILE" ] || [ ! -s "$PFMFILE" ]; then
    echo "${PFMFILE} either doesn't exist or is empty."
    exit 1
fi


python ./unet_scripts/unet_wm_predict.py \
        --model_file ${MODELFILE} \
        --output_path ${OUTPUTPATH} \
        --lowb_file ${LOWBFILE} \
        --fa_file ${FAFILE} \
        --tract_file ${PFMFILE} \
        --label_list_path ${LABLIST}


echo "All done!!"

