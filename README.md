# BSBT
BrainStem Bundle Tool

### Dependencies

BSBT uses wrapped MRtrix3 and Freesurfer (>7.3) commands for tract generation. To run these command, please ensure that:

1. **MRtrix3** is installed and available in `PATH`.
   - Verify MRtrix installation by running: `mrconvert -help`.

2. **Freesurfer** is installed, and `FREESURFER_HOME` env variable is set.
   - Dont forget to source the setup script by either adding the below line to your .bashrc/.bash_profile or running explicitely in terminal:
     ```bash
     source /path/to/freesurfer/SetUpFreeSurfer.sh
     ```
   - Verify the installation by running: `recon-all -version`.


### Reassembling Split Model Weights (NOT USABLE FOR NOW)

The CNN model weights file is too large to include as a single file, so it is split into two parts. First, navigate to the ./scripts folder in Terminal, then run the following shell script to reassemble the weights file:

```bash
sh ./reassemble_weights.sh
```

### Running BSBT

Once all dependencies are set up, you can run BSBT through the main shell script:

```bash
cd BSBT
sh ./run_BSBT.sh 
   --dwi </path/to/dwi_volume>
   --bvals </path/to/bvals_file>
   --bvecs </path/to/bvecs_file>
   --out </path/to/output_directory>
   --threads <number of threads>
```
