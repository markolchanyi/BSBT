import os, sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
from scipy import ndimage
import unet.utils as utils
import unet.models as models
from unet.crf import dense_crf_inference
from unet.utils import parse_args_unet_predict, print_no_newline


def unet_predict(output_path,
            lowb_file,
            fa_file,
            tract_file,
            path_label_list,
            model_file,
            resolution=1.0,
            generator_mode='rgb',
            unet_feat_count=24,
            n_levels=5,
            conv_size=3,
            feat_multiplier=2,
            nb_conv_per_level=2,
            activation='elu',
            bounding_box_width=64,
            aff_ref=np.eye(4)):

    assert (generator_mode == 'fa_v1') | (generator_mode == 'rgb'), \
        'generator mode must be fa_v1 or rgb'

    print_no_newline("Starting UNet WM predictions... ")
    # Load label list
    label_list = np.load(path_label_list)

    # Build Unet
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


    if not os.path.exists(os.path.join(output_path, 'unet_results')):
        os.makedirs(os.path.join(output_path, 'unet_results'))
    output_seg_file = os.path.join(output_path, 'unet_results', 'wmunet.seg.mgz')
    output_posteriors_file = os.path.join(output_path, 'unet_results', 'wmunet.posteriors.mgz')
    output_vol_file = os.path.join(output_path, 'unet_results', 'wmunet.vol.npy')
    output_crf_file = os.path.join(output_path, 'unet_results', 'wmunet.crfposteriors.mgz')
    output_crf_seg_file = os.path.join(output_path, 'unet_results', 'wmunet.crfseg.mgz')

    # Read in and reorient lowb
    lowb, aff, _ = utils.load_volume(lowb_file, im_only=False)
    lowb, aff2 = utils.align_volume_to_ref(lowb, aff, aff_ref=aff_ref, return_aff=True, n_dims=3)

    # If the resolution is not the one the model expected, we need to upsample!
    if any(abs(np.diag(aff2)[:-1] - resolution) > 0.1):
        print('Warning: lowb does not have the resolution that the CNN expects (found resolution is ' + str(abs(np.diag(aff2)[:-1])) + '); we need to resample')
        #lowb, aff2 = utils.rescale_voxel_size(lowb, aff2, [resolution, resolution, resolution])


    # We only resample in the cropped region
    # TODO: we'll want to do this in the log-tensor domain
    if generator_mode=='fa_v1':
        fa, aff, _ = utils.load_volume(fa_file, im_only=False)
        fa = utils.resample_like(lowb, aff2, fa, aff)
        v1, aff, _ = utils.load_volume(tract_file, im_only=False)
        v1_copy = v1.copy()
        v1 = np.zeros([*t1.shape, 3])
        v1[:, :, :, 0] = - utils.resample_like(lowb, aff2, v1_copy[:, :, :, 0], aff, method='nearest') # minus as in generators.py
        v1[:, :, :, 1] = utils.resample_like(lowb, aff2, v1_copy[:, :, :, 1], aff, method='nearest')
        v1[:, :, :, 2] = utils.resample_like(lowb, aff2, v1_copy[:, :, :, 2], aff, method='nearest')
        #dti = np.abs(v1 * fa[..., np.newaxis])
    else:
        fa, aff, _ = utils.load_volume(fa_file, im_only=False)
        v1 = utils.load_volume(tract_file, im_only=True)
        #dti = np.abs(v1 * fa[..., np.newaxis])
        fa = utils.resample_like(lowb, aff2, fa, aff)
        dti = utils.resample_like(lowb, aff2, v1, aff)
        #dti = v1

    # Predict with left-right flipping augmentation
    input = np.concatenate((lowb[..., np.newaxis], fa[..., np.newaxis], dti), axis=-1)[np.newaxis,...]
    print("running rsUNET...")
    posteriors = np.squeeze(unet_model.predict(input))
    print("done")
    posteriors_flipped = np.squeeze(unet_model.predict(input[:,::-1,:,:,:]))
    nlab = int((len(label_list) - 1)/2)
    posteriors[:,:,:,0] = 0.5 * posteriors[:,:,:,0] + 0.5 *  posteriors_flipped[::-1,:,:,0]
    posteriors[:,:,:,1:nlab+1] = 0.5 * posteriors[:,:,:,1:nlab+1] + 0.5 *  posteriors_flipped[::-1,:,:,nlab+1:]
    posteriors[:,:,:,nlab+1:] = 0.5 * posteriors[:,:,:,nlab+1:] + 0.5 *  posteriors_flipped[::-1,:,:,1:nlab+1]


    # Compute volumes (skip background)
    voxel_vol_mm3 = np.linalg.det(aff2)
    vols_in_mm3 = np.sum(posteriors, axis=(0,1,2))[1:] * voxel_vol_mm3
    print("total posterior volume: ", vols_in_mm3)

    # Compute segmentations
    seg = label_list[np.argmax(posteriors, axis=-1)]

    # Compute conditional random field segmentations
    crf_output = dense_crf_inference(posteriors)
    crf_segs = label_list[np.argmax(crf_output, axis=-1)]

    print("CRF seg volumes: ")
    values, counts = np.unique(crf_segs, return_counts=True)
    for value, count in zip(values, counts):
        print(f"Intensity Value: {value}, Total Voxels: {count}")

    # Write to disk and we're done!
    utils.save_volume(crf_output, aff2, None, output_crf_file)
    utils.save_volume(crf_segs, aff2, None, output_crf_seg_file)

    utils.save_volume(seg.astype(int), aff2, None, output_seg_file)
    np.save(output_vol_file, vols_in_mm3)
    utils.save_volume(posteriors, aff2, None, output_posteriors_file)
    print('finished!')





args = parse_args_unet_predict()

model_file = args.model_file
output_path = args.output_path
lowb_file = args.lowb_file
fa_file = args.fa_file
tract_file = args.tract_file
path_label_list = args.label_list_path

unet_predict(output_path,
            lowb_file,
            fa_file,
            tract_file,
            path_label_list,
            model_file,
            resolution=args.resolution,
            generator_mode=args.generator_mode,
            unet_feat_count=args.unet_feat_count,
            n_levels=args.n_levels,
            conv_size=args.conv_size,
            feat_multiplier=args.feat_multiplier,
            nb_conv_per_level=args.nb_conv_per_level,
            activation=args.activation,
            bounding_box_width=args.bounding_box_width,
            aff_ref=np.eye(4))
