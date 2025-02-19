#!/bin/bash

# Description:
# This script generates the RGB-encoded Probabilistic Fiber Map (PFM)
# from brainstem-adjacent ROIs extracted from fs_preprocc_extract.sh
#
# Usage:
#   ./trackgen/pfmgen.sh <INTERDIR> <LOGFILE> <OUTPUT_DIR> <THREADS>
#
# Positional args:
#   1. <INTERDIR>    - Directory containing preprocessed intermediates (required)
#   2. <LOGFILE>     - log file (optional)
#   2. <OUTPUT_DIR>  - Output directory for PFM volumes
#   3. <threads>     - Number of threads to use for MrTrix (optional; default: 1)

run_cmd() {
  local cmd="$1"
  echo "$cmd" | tee -a "$log_file"
  eval "$cmd" &>> "$log_file"
  echo " "
}


if [ -z "$1" ]; then
  echo "Error: Intermediate directory is required."
  echo "Usage: ./trackgen/pfmgen.sh <INTERDIR> <LOGFILE> <OUTPUT_DIR> <THREADS>"
  exit 1
fi

intermediate_directory="$1"

log_file="${2:-$./preprocessing_log_$(date +%Y%m%d%H%M%S).log}"

output_directory="$3"

threads="${4:-1}"  # Default to singgle thread

# Ensure FREESURFER and MrTrix libs are found
if [ -z "$FREESURFER_HOME" ]; then
    echo "Error: FREESURFER_HOME is not set. Please make sure to source Freesurfer."
    exit 1
fi

if ! command -v mrconvert &> /dev/null; then
    echo "Error: MRtrix not found. Please ensure it is installed."
    exit 1
fi

# Check if input dir and log dirs exist. If log dir doesnt exist, then make it..
if [ ! -d "$intermediate_directory" ]; then
  echo "Error: Directory $input_directory does not exist."
  exit 1
fi


###############################################################################
#*************            PREPROCESSING COMMANDS                **************#
###############################################################################

inpmif_resampled="${intermediate_directory}/input_dwi_1mm_resampled.mif"
brainmask="${intermediate_directory}/brain_mask.nii.gz"

fDCmorphed="${intermediate_directory}/DC_morphed.nii.gz"
fcortmorphed="${intermediate_directory}/cort_morphed.nii.gz"
fthalmorphed="${intermediate_directory}/thal_morphed.nii.gz"
fCBmorphed="${intermediate_directory}/CB_morphed.nii.gz"
fmedullamorphed="${intermediate_directory}/medulla_morphed.nii.gz"
fponsmorphed="${intermediate_directory}/pons_morphed.nii.gz"
fDCboundary="${intermediate_directory}/DC_boundary.nii.gz"

volinfofile="${intermediate_directory}/volinfo.txt"

SINGLE_SHELL_MODE=$(grep "^single_shell_mode:" "$volinfofile" | awk '{print $2}')
echo "SINGLE_SHELL_MODE is set to: $SINGLE_SHELL_MODE"

# Response function estimation dependent on single vs. multishell input
wmtxt="${intermediate_directory}/wm.txt"
gmtxt="${intermediate_directory}/gm.txt"
csftxt="${intermediate_directory}/csf.txt"
voxfile="${intermediate_directory}/voxels.mif"
wmfod="${intermediate_directory}/wmfod.mif"
gmfod="${intermediate_directory}/gmfod.mif"
csffod="${intermediate_directory}/csffod.mif"
wmfodnorm="${intermediate_directory}/wmfod_norm.mif"
gmfodnorm="${intermediate_directory}/gmfod_norm.mif"
csffodnorm="${intermediate_directory}/csffod_norm.mif"
tractmask="${intermediate_directory}/tractography_mask.nii.gz"

echo "Estimating response function(s) and calculating ODFs..."
if [ "$SINGLE_SHELL_MODE" = "True" ]; then
    run_cmd "dwi2response fa ${inpmif_resampled} ${wmtxt} -voxels ${voxfile} -threshold 0.3 -nthreads ${threads} -force"
    run_cmd "dwi2fod csd ${inpmif_resampled} ${wmtxt} ${wmfod} -mask ${brainmask} -lmax 6 -nthreads ${threads} -force"
else
    run_cmd "dwi2response dhollander ${inpmif_resampled} ${wmtxt} ${gmtxt} ${csftxt} -nthreads ${threads} -force"
    run_cmd "dwi2fod msmt_csd ${inpmif_resampled} -mask ${brainmask} ${wmtxt} ${wmfod} ${gmtxt} ${gmfod} ${csftxt} ${csffod} -nthreads ${threads} -force"
fi
echo "done"

echo "Normalizing ODFs..."
if [ "$SINGLE_SHELL_MODE" = "True" ]; then
    run_cmd "mtnormalise ${wmfod} ${wmfodnorm} -mask ${brainmask} -force"
else
    run_cmd "mtnormalise ${wmfod} ${wmfodnorm} ${gmfod} ${gmfodnorm} ${csffod} ${csffodnorm} -mask ${brainmask} -force"
fi
echo "done"

# PFM tractography
ftractsTHAL="${intermediate_directory}/tracts_thal.tck"
ftractsDC="${intermediate_directory}/tracts_DC.tck"
ftractsCB="${intermediate_directory}/tracts_CB.tck"

tractOPTS_pref="-algorithm iFOD2 -angle 50 -select 100000"
tractOPTS_suf="-max_attempts_per_seed 750 -trials 750 -cutoff 0.1 -maxlength 200 -step 0.5"

echo "Starting R tracking..."
run_cmd "tckgen ${tractOPTS_pref} -seed_image ${fthalmorphed} -include ${fmedullamorphed} ${wmfodnorm} ${ftractsTHAL} -mask ${tractmask} ${tractOPTS_suf} -nthreads ${threads} -force"

echo "Starting G tracking..."
run_cmd "tckgen ${tractOPTS_pref} -seed_image ${fDCboundary} -include ${fmedullamorphed} -exclude ${fCBmorphed} ${wmfodnorm} ${ftractsDC} -mask ${tractmask} ${tractOPTS_suf} -nthreads ${threads} -force"

echo "Starting B tracking..."
run_cmd "tckgen ${tractOPTS_pref} -seed_image ${fDCboundary} -include ${fCBmorphed} -exclude ${fponsmorphed} -exclude ${fmedullamorphed} ${wmfodnorm} ${ftractsCB} -mask ${tractmask} ${tractOPTS_suf} -nthreads ${threads} -force"

meanlowbnifti="${intermediate_directory}/mean_b0.nii.gz"
ftractsTHALimg="${intermediate_directory}/tracts_thal.nii.gz"
ftractsDCimg="${intermediate_directory}/tracts_DC.nii.gz"
ftractsCBimg="${intermediate_directory}/tracts_CB.nii.gz"

echo "Mapping streamlines to intensities..."
run_cmd "tckmap ${ftractsTHAL} -template ${meanlowbnifti} -contrast tdi ${ftractsTHALimg} -force"
run_cmd "tckmap ${ftractsDC} -template ${meanlowbnifti} -contrast tdi ${ftractsDCimg} -force"
run_cmd "tckmap ${ftractsCB} -template ${meanlowbnifti} -contrast tdi ${ftractsCBimg} -force"

ftractsDCmatched="${intermediate_directory}/tracts_DC_matched.nii.gz"
ftractsCBmatched="${intermediate_directory}/tracts_CB_matched.nii.gz"

echo "Histogram matching G and B channels to R channel..."
run_cmd "mrhistmatch linear ${ftractsDCimg} ${ftractsTHALimg} ${ftractsDCmatched} -force"
run_cmd "mrhistmatch linear ${ftractsCBimg} ${ftractsTHALimg} ${ftractsCBmatched} -force"

echo "re-scaling intensities..."
run_cmd "mrcalc ${ftractsDCmatched} 5 -mult ${ftractsDCmatched} -force"
run_cmd "mrcalc ${ftractsCBmatched} 8 -mult ${ftractsCBmatched} -force"

# Concatenation into PFM
ftractsconcat="${intermediate_directory}/tracts_concatenated.nii.gz"
ftractsconcat1mm="${intermediate_directory}/tracts_concatenated_1mm.nii.gz"

echo "Transforming into color map..."
run_cmd "mrcat ${ftractsTHALimg} ${ftractsCBmatched} ${ftractsDCmatched} ${ftractsconcat} -force"
run_cmd "mrgrid ${ftractsconcat} regrid -voxel 1.0 ${ftractsconcat1mm} -force"

fpons="${intermediate_directory}/pons.nii.gz"
ftractsconcatCROPPED="${intermediate_directory}/tracts_concatenated_1mm_cropped.nii.gz"

echo "Cropping PFM"
run_cmd "python -c \"from trackgen.utils import crop_around_centroid; crop_around_centroid('${ftractsconcat1mm}','${fpons}','${ftractsconcatCROPPED}')\""

lowbcropped="${intermediate_directory}/lowb_1mm_cropped.nii.gz"
lowbcroppedNORM="${output_directory}/lowb_1mm_cropped_norm.nii.gz"
facropped="${intermediate_directory}/fa_1mm_cropped.nii.gz"
facroppedNORM="${output_directory}/fa_1mm_cropped_norm.nii.gz"
tractscroppedNORM="${output_directory}/tracts_concatenated_1mm_cropped_norm.nii.gz"

echo "Rescaling intensities..."
run_cmd "python -c \"from trackgen.utils import rescale_intensities; rescale_intensities('${ftractsconcatCROPPED}','${lowbcropped}','${tractscroppedNORM}',factor=20)\""
run_cmd "python -c \"from trackgen.utils import rescale_intensities; rescale_intensities('${lowbcropped}','${lowbcropped}','${lowbcroppedNORM}',factor=5)\""
run_cmd "python -c \"from trackgen.utils import rescale_intensities; rescale_intensities('${facropped}','${lowbcropped}','${facroppedNORM}',factor=5)\""
