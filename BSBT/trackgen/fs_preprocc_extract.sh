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
  echo " "
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

# convert to MrTrix .mif format
inpmif="${input_directory}/input_dwi.mif"
run_cmd "mrconvert ${inp} ${inpmif} -fslgrad ${fbvec} ${fbval} -nthreads ${threads} -force"

# Run mrinfo to populate header.json with MRtrix header information
# create DWI info files
header_json="${input_directory}/header.json"
volinfo_txt="${input_directory}/volinfo.txt"
run_cmd "touch ${header_json}"
run_cmd "touch ${volinfo_txt}"
run_cmd "mrinfo -json_all ${header_json} ${inpmif} -force"

# populate volinfo.txt
echo "Extracting header info..."
run_cmd "python -c \"from trackgen.utils import write_volinfo; write_volinfo('${header_json}','${volinfo_txt}')\""

# resample to 1mm isotropic
inpmif_resampled="${input_directory}/input_dwi_1mm_resampled.mif"

echo "Regridding to 1mm..."
run_cmd "mrgrid ${inpmif} regrid -vox 1.0 ${inpmif_resampled} -nthreads ${threads} -force"

# extract mean b0
meanlowbmif="${input_directory}/mean_b0.mif"
meanlowbnifti="${input_directory}/mean_b0.nii.gz"

echo "Extracting mean b0..."
run_cmd "dwiextract ${inpmif_resampled} - -bzero | mrmath - mean ${meanlowbmif} -axis 3 -force"
run_cmd "mrconvert ${meanlowbmif} ${meanlowbnifti} -datatype float32 -force"

# Extract brain mask with FS SynthStrip
meanlowbstripped="${input_directory}/mean_b0_masked.nii.gz"
brainmask="${input_directory}/brain_mask.nii.gz"

echo "Extracting brain mask..."
run_cmd "mri_synthstrip -i ${meanlowbnifti} -o ${meanlowbstripped} -m ${brainmask}"

# Fit tensor model to the DWI to extract relevent metrics
echo "Tensor fitting and extracting FA/MD/V1..."
tensorimg="${input_directory}/dwi_dt_1mm.mif"
faimg="${input_directory}/fa_1mm.nii.gz"
mdimg="${input_directory}/md_1mm.nii.gz"
v1img="${input_directory}/v1_1mm.nii.gz"

run_cmd "dwi2tensor ${inpmif_resampled} ${tensorimg} -nthreads ${threads} -force"
run_cmd "tensor2metric ${tensorimg} -fa ${faimg} -adc ${mdimg} -vector ${v1img} -force"

# intermediate cleanup
rm ${tensorimg}
rm ${inpmif}
rm ${meanlowbmif}

# run synthseg
synthsegpth="${input_directory}/synthseg_output"

echo "Running SynthSeg on the mean_b0..."
run_cmd "mri_synthseg --i ${meanlowbnifti} --o ${synthsegpth} --threads ${threads}"
synthsegvol="${input_directory}/synthseg_output/mean_b0_synthseg.nii.gz"

# Run SynthSR to generate MPRage for brainstem subfield segmentation
echo "Generating SynthSR MPRage..."
run_cmd "mri_synthsr --i ${meanlowbnifti} --o ${input_directory} --threads ${threads}"
lowbsynthsr="${input_directory}/mean_b0_synthsr.nii.gz" #SynthSR convention

# Run brainstem subfield segmentation
# this requires making a "fake" FS_subject directory with an "mri" folder
# and T1.mgz, norm.mgz and aseg.mgz vols
fsmridir="${input_directory}/fs_subj_dir/mri"
bssubfieldseg="${input_directory}/brainstem_subfields/brainstemSsLabels.FSvoxelSpace.mgz"

echo "Prepping FS directory for brainstem subfield segmentation..."
mkdir -p $fsmridir
mkdir -p "${input_directory}/brainstem_subfields"
run_cmd "mri_convert ${lowbsynthsr} ${fsmridir}/T1.mgz -rl ${meanlowbnifti}"
run_cmd "mri_convert ${synthsegvol} ${fsmridir}/aseg.mgz"
cp ${fsmridir}/T1.mgz ${fsmridir}/norm.mgz

echo "Running brainstem subfield segmentation..."
run_cmd "segment_subregions brainstem --out-dir=${input_directory}/brainstem_subfields --cross ${input_directory}/fs_subj_dir --threads ${threads}"
run_cmd "mri_convert ${bssubfieldseg} ${bssubfieldseg} -rl ${meanlowbnifti} -rt nearest -odt float"

# Extract SynthSeg ROIs to use for brainstem tract generation
thallab="[10,49]"
DClab="[28,60]"
cortlab="[18,54]"
CBlab="[7,46]"
midbrainlab="[173]"
ponslab="[174]"
medullalab="[175]"
brainstemlab="[16]"
synthsegalllabs="[10,49,28,60,7,46,16]" # all SynthSeg extra-brainstem labels together

fthal="${input_directory}/thal.nii.gz"
fDC="${input_directory}/DC.nii.gz"
fcort="${input_directory}/cort.nii.gz"
fCB="${input_directory}/CB.nii.gz"
fmidbrain="${input_directory}/midbrain.nii.gz"
fpons="${input_directory}/pons.nii.gz"
fmedulla="${input_directory}/medulla.nii.gz"
fbrainstem="${input_directory}/brainstem.nii.gz"
fsynthsegalllabs="${input_directory}/all_labels.nii.gz"

echo "Extracting SynthSeg and Brainstem Subfield ROIs..."
run_cmd "python -c \"from trackgen.utils import extract_bin_roi; extract_bin_roi('${synthsegvol}',${thallab},'${fthal}')\""
run_cmd "python -c \"from trackgen.utils import extract_bin_roi; extract_bin_roi('${synthsegvol}',${DClab},'${fDC}')\""
run_cmd "python -c \"from trackgen.utils import extract_bin_roi; extract_bin_roi('${synthsegvol}',${cortlab},'${fcort}')\""
run_cmd "python -c \"from trackgen.utils import extract_bin_roi; extract_bin_roi('${synthsegvol}',${CBlab},'${fCB}')\""
run_cmd "python -c \"from trackgen.utils import extract_bin_roi; extract_bin_roi('${synthsegvol}',${brainstemlab},'${fbrainstem}')\""
run_cmd "python -c \"from trackgen.utils import extract_bin_roi; extract_bin_roi('${synthsegvol}',${synthsegalllabs},'${fsynthsegalllabs}')\""
run_cmd "python -c \"from trackgen.utils import extract_bin_roi; extract_bin_roi('${bssubfieldseg}',${midbrainlab},'${fmidbrain}')\""
run_cmd "python -c \"from trackgen.utils import extract_bin_roi; extract_bin_roi('${bssubfieldseg}',${ponslab},'${fpons}')\""
run_cmd "python -c \"from trackgen.utils import extract_bin_roi; extract_bin_roi('${bssubfieldseg}',${medullalab},'${fmedulla}')\""

# Get centroid voxel coordinates of union of thalamic and brainstem masks to obtain bounding box location
# Crop all relevent DWI metrics (FOR CNN PREP)
lowbcropped="${input_directory}/lowb_1mm_cropped.nii.gz"
facropped="${input_directory}/fa_1mm_cropped.nii.gz"
v1cropped="${input_directory}/v1_1mm_cropped_norm.nii.gz"
cropsize=64


echo "Cropping relevent DWI volumes..."
run_cmd "mrcentroid -voxelspace ${fpons} > ${input_directory}/pontine_cntr_coords.txt"
run_cmd "python -c \"from trackgen.utils import crop_around_centroid; crop_around_centroid('${meanlowbnifti}','${fpons}','${lowbcropped}')\""
run_cmd "python -c \"from trackgen.utils import crop_around_centroid; crop_around_centroid('${faimg}','${fpons}','${facropped}')\""
run_cmd "python -c \"from trackgen.utils import crop_around_centroid; crop_around_centroid('${v1img}','${fpons}','${v1cropped}')\""

# Apply dilation/erosion for peri-brainstem vols for better streamline seeding for PFM
fDCmorphed="${input_directory}/DC_morphed.nii.gz"
fcortmorphed="${input_directory}/cort_morphed.nii.gz"
fthalmorphed="${input_directory}/thal_morphed.nii.gz"
fCBmorphed="${input_directory}/CB_morphed.nii.gz"
fmedullamorphed="${input_directory}/medulla_morphed.nii.gz"
fponsmorphed="${input_directory}/pons_morphed.nii.gz"
fDCboundary="${input_directory}/DC_boundary.nii.gz"
morpho=3

echo "Morphing peri-brainstem ROIs..."
run_cmd "maskfilter ${fDC} dilate -npass 4 ${fDCmorphed} -force"
run_cmd "maskfilter ${fcort} dilate -npass 5 ${fcortmorphed} -force"
run_cmd "maskfilter ${fthal} dilate -npass 3 ${fthalmorphed} -force"
run_cmd "maskfilter ${fCB} erode -npass 3 ${fCBmorphed} -force"
run_cmd "maskfilter ${fmedulla} dilate -npass 2 ${fmedullamorphed} -force"
run_cmd "maskfilter ${fpons} erode -npass 4 ${fponsmorphed} -force"
run_cmd "mrcalc ${fDCmorphed} ${fcortmorphed} -mult ${fDCboundary} -force"

echo "done preprocessing!!"
echo " "
echo " "
