import os
import math
import numpy as np
from scipy import ndimage

import tensorflow as tf
import utils as utils
import models
from dcrf_gradient import dense_crf_inference
from scipy.ndimage import gaussian_filter
from utils import downsample_upsample_scipy 
import nibabel as nib

def predict(subject_list,
                fs_subject_dir,
                dataset,
                path_label_list,
                model_file,
                resolution_model_file=1.0,
                generator_mode='rgb',
                unet_feat_count=24,
                n_levels=5,
                conv_size=3,
                feat_multiplier=2,
                nb_conv_per_level=2,
                activation='elu',
                bounding_box_width=64,
                aff_ref=np.eye(4),
                shell_flag=None,
                attention=True,
                ablate_rgb=False,
                use_v1=False,
                outputpath="results",
                ds_factor=None):



    assert (generator_mode == 'fa_v1') | (generator_mode == 'rgb'), \
        'generator mode must be fa_v1 or rgb'

    assert dataset in ('HCP','ADNI','template','validate','DRC') #will do for now

    # Load label list
    label_list = np.load(path_label_list)

    # Build Unet
    if ablate_rgb:
        unet_input_shape = [bounding_box_width, bounding_box_width, bounding_box_width, 2]
    else:
        unet_input_shape = [bounding_box_width, bounding_box_width, bounding_box_width, 5]
    n_labels = len(label_list)

    unet_model = models.unet(nb_features=unet_feat_count,
                             input_shape=unet_input_shape,
                             nb_levels=n_levels,
                             conv_size=conv_size,
                             nb_labels=n_labels,
                             feat_mult=feat_multiplier,
                             nb_conv_per_level=nb_conv_per_level,
                             conv_dropout=0,
                             batch_norm=-1,
                             activation=activation,
                             attention_gating=True,
                             input_model=None)

    unet_model.load_weights(model_file, by_name=True)
    print("loaded: ", model_file)

    results_base_dir = outputpath
    ### iteratre over subjects
    for subject in subject_list:

        if not os.path.exists(os.path.join(fs_subject_dir, subject, results_base_dir)):
            os.mkdir(os.path.join(fs_subject_dir, subject, results_base_dir))
        output_seg_file = os.path.join(fs_subject_dir, subject, results_base_dir, 'bsNet.seg.mgz')
        output_crf_file = os.path.join(fs_subject_dir, subject, results_base_dir, 'bsNet.crfposterior.mgz')
        output_crf_seg_file = os.path.join(fs_subject_dir, subject, results_base_dir, 'bsNet.crfseg.mgz')
        output_posteriors_file = os.path.join(fs_subject_dir, subject, results_base_dir, 'bsNet.posteriors.mgz')
        output_vol_file = os.path.join(fs_subject_dir, subject, results_base_dir, 'bsNet.vol.npy')

        # File names
        if dataset=='HCP':
            t1_file = os.path.join(fs_subject_dir, subject, 'mri', 'T1w_hires.masked.norm.mgz')
            aseg_file = os.path.join(fs_subject_dir, subject, 'mri', 'aseg.mgz')
            fa_file = os.path.join(fs_subject_dir, subject, 'dmri', 'dtifit.1+2+3K_FA.nii.gz')
            v1_file = os.path.join(fs_subject_dir, subject, 'dmri', 'dtifit.1+2+3K_V1.nii.gz')


        if dataset == 'template':
            t1_file = os.path.join(fs_subject_dir, subject, 'dmri', 'lowb.nii.gz')
            fa_file = os.path.join(fs_subject_dir, subject, 'dmri', 'FA.nii.gz')
            if use_v1:
                v1_file = os.path.join(fs_subject_dir, subject, 'dmri', 'v1.nii.gz')
            else:
                v1_file = os.path.join(fs_subject_dir, subject, 'dmri', 'tracts.nii.gz')


        # Read in and reorient T1
        t1, aff, _ = utils.load_volume(t1_file, im_only=False)
        t1, aff2 = utils.align_volume_to_ref(t1, aff, aff_ref=aff_ref, return_aff=True, n_dims=3)

        # If the resolution is not the one the model expected, we need to upsample!
        if any(abs(np.diag(aff2)[:-1] - resolution_model_file) > 0.1):
            print('Warning: t1 does not have the resolution that the CNN expects; we need to resample')
            t1, aff2 = utils.rescale_voxel_size(t1, aff2, [resolution_model_file, resolution_model_file, resolution_model_file])


        # Now the diffusion data
        # We only resample in the cropped region
        # TODO: we'll want to do this in the log-tensor domain
        if generator_mode=='fa_v1':
            fa, aff, _ = utils.load_volume(fa_file, im_only=False)
            fa = utils.resample_like(t1, aff2, fa, aff)
            v1, aff, _ = utils.load_volume(v1_file, im_only=False)
            v1 = np.nan_to_num(v1)
            v1_copy = v1.copy()
            v1 = np.zeros([*t1.shape, 3])
            v1[:, :, :, 0] = - utils.resample_like(t1, aff2, v1_copy[:, :, :, 0], aff, method='nearest') # minus as in generators.py
            v1[:, :, :, 1] = utils.resample_like(t1, aff2, v1_copy[:, :, :, 1], aff, method='nearest')
            v1[:, :, :, 2] = utils.resample_like(t1, aff2, v1_copy[:, :, :, 2], aff, method='nearest')
            #dti = np.abs(v1 * fa[..., np.newaxis])
        else:
            fa, aff, _ = utils.load_volume(fa_file, im_only=False)
            v1 = utils.load_volume(v1_file, im_only=True)
            v1 = np.nan_to_num(v1)
            #dti = np.abs(v1 * fa[..., np.newaxis])
            fa = utils.resample_like(t1, aff2, fa, aff)
            dti = utils.resample_like(t1, aff2, v1, aff)
            #dti = v1

        if ds_factor is not None:
            #gauss_sigma = ds_factor/math.sqrt(12)
            gauss_sigma = ds_factor/(math.sqrt(2*math.log(2)))
            print("Downsample factor: ",ds_factor, "Downsampling with sigma = ",str(gauss_sigma))
            t1 = gaussian_filter(t1, sigma=gauss_sigma)
            t1 = downsample_upsample_scipy(t1,downsampling_factor=1/(ds_factor),original_shape=t1.shape)
            fa = gaussian_filter(fa, sigma=gauss_sigma)
            fa = downsample_upsample_scipy(fa,downsampling_factor=1/(ds_factor),original_shape=fa.shape)

            # for SIM volume
            for channel in range(dti.shape[3]):
                # Apply Gaussian filter to each channel in-place
                dti[:,:,:,channel] = gaussian_filter(dti[:,:,:,channel], sigma=gauss_sigma)
                dti[:,:,:,channel] = downsample_upsample_scipy(dti[:,:,:,channel],downsampling_factor=1/(ds_factor),original_shape=dti[:,:,:,channel].shape)

        # Predict with left-right flipping augmentation
        if ablate_rgb:
            input = np.concatenate((t1[..., np.newaxis], fa[..., np.newaxis]), axis=-1)[np.newaxis,...]
        else:
            input = np.concatenate((t1[..., np.newaxis], fa[..., np.newaxis], dti), axis=-1)[np.newaxis,...]

        posteriors = np.squeeze(unet_model.predict(input))
        posteriors_flipped = np.squeeze(unet_model.predict(input[:,::-1,:,:,:]))
        #print("shape of posteriors is: ", posteriors.shape)
        #print("shape of flipped posteriors is: ", posteriors_flipped.shape)
        nlab = int(( len(label_list) - 1 ) / 2)
        posteriors[:,:,:,0] = 0.5 * posteriors[:,:,:,0] + 0.5 *  posteriors_flipped[::-1,:,:,0] #Background
        posteriors[:,:,:,1:nlab+1] = 0.5 * posteriors[:,:,:,1:nlab+1] + 0.5 *  posteriors_flipped[::-1,:,:,nlab+1:]
        posteriors[:,:,:,nlab+1:] = 0.5 * posteriors[:,:,:,nlab+1:] + 0.5 *  posteriors_flipped[::-1,:,:,1:nlab+1]

        # extract attention layer output
        #print(unet_model.summary())


        ###########
        attention_layer_output = unet_model.get_layer('attn_coeffs').output
        get_attention_output = tf.keras.backend.function([unet_model.input], [attention_layer_output])
        attention_output = get_attention_output([input])[0]

        attention_nifti = nib.Nifti1Image(attention_output[0,:,:,:,0], aff2)
        nib.save(attention_nifti,os.path.join(fs_subject_dir, subject, results_base_dir,"attn_coeffs.nii.gz"))
        ##########

        ''' 
        ###########
        attention_layer_output = unet_model.get_layer('attn_wx').output
        get_attention_output = tf.keras.backend.function([unet_model.input], [attention_layer_output])
        attention_output = get_attention_output([input])[0]

        attention_nifti = nib.Nifti1Image(attention_output[0,...], aff2)
        nib.save(attention_nifti,os.path.join(fs_subject_dir, subject, results_base_dir,"attn_wx.nii.gz"))
        ##########


        ###########
        attention_layer_output = unet_model.get_layer('attn_add').output
        get_attention_output = tf.keras.backend.function([unet_model.input], [attention_layer_output])
        attention_output = get_attention_output([input])[0]

        attention_nifti = nib.Nifti1Image(attention_output[0,...], aff2)
        nib.save(attention_nifti,os.path.join(fs_subject_dir, subject, results_base_dir,"attn_add.nii.gz"))
        ##########

        ###########
        attention_layer_output = unet_model.get_layer('attn_out').output
        get_attention_output = tf.keras.backend.function([unet_model.input], [attention_layer_output])
        attention_output = get_attention_output([input])[0]

        attention_nifti = nib.Nifti1Image(attention_output[0,...], aff2)
        nib.save(attention_nifti,os.path.join(fs_subject_dir, subject, results_base_dir,"attn_out.nii.gz"))
        ##########


        ###########
        attention_layer_output = unet_model.get_layer('unet_conv_downarm_0_1').output
        get_attention_output = tf.keras.backend.function([unet_model.input], [attention_layer_output])
        attention_output = get_attention_output([input])[0]

        attention_nifti = nib.Nifti1Image(attention_output[0,...], aff2)
        nib.save(attention_nifti,os.path.join(fs_subject_dir, subject, results_base_dir,"unet_conv_downarm_0_1.nii.gz"))
        ##########

        ###########
        attention_layer_output = unet_model.get_layer('attn_g_0').output
        get_attention_output = tf.keras.backend.function([unet_model.input], [attention_layer_output])
        attention_output = get_attention_output([input])[0]

        attention_nifti = nib.Nifti1Image(attention_output[0,...], aff2)
        nib.save(attention_nifti,os.path.join(fs_subject_dir, subject, results_base_dir,"attn_g_0.nii.gz"))
        ##########


        ###########
        attention_layer_output = unet_model.get_layer('attn_g_1').output
        get_attention_output = tf.keras.backend.function([unet_model.input], [attention_layer_output])
        attention_output = get_attention_output([input])[0]

        attention_nifti = nib.Nifti1Image(attention_output[0,...], aff2)
        nib.save(attention_nifti,os.path.join(fs_subject_dir, subject, results_base_dir,"attn_g_1.nii.gz"))
        ##########


        ###########
        #attention_layer_output = unet_model.get_layer('attn_g_2').output
        #get_attention_output = tf.keras.backend.function([unet_model.input], [attention_layer_output])
        #attention_output = get_attention_output([input])[0]

        #attention_nifti = nib.Nifti1Image(attention_output[0,...], aff2)
        #nib.save(attention_nifti,os.path.join(fs_subject_dir, subject, results_base_dir,"attn_g_2.nii.gz"))
        ##########
        '''

        # Compute volumes (skip background)
        voxel_vol_mm3 = np.linalg.det(aff2)
        vols_in_mm3 = np.sum(posteriors, axis=(0,1,2))[1:] * voxel_vol_mm3

        print(" ")
        print(" ")
        print("Volumes are: ", vols_in_mm3)
        print(" ")
        print(" ")
        # Compute segmentations
        #posteriors[posteriors>0.4] = 1 
        seg = label_list[np.argmax(posteriors, axis=-1)]

        crf_output = dense_crf_inference(posteriors)
        crf_segs = label_list[np.argmax(crf_output, axis=-1)]

        utils.save_volume(crf_output, aff2, None, output_crf_file)
        utils.save_volume(crf_segs, aff2, None, output_crf_seg_file)
        # Write to disk and we're done!
        utils.save_volume(seg.astype(int), aff2, None, output_seg_file)
        np.save(output_vol_file, vols_in_mm3)
        utils.save_volume(posteriors, aff2, None, output_posteriors_file)
        #print('freeview ' + t1_file + ' ' + fa_file + ' '  + output_seg_file)

        print('All done!')
