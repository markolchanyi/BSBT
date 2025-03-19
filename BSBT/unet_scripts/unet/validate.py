import os

import numpy as np
import tensorflow as tf
import torch
import glob

import utils
import models
import metrics

from generators import encode_onehot

from tensorflow.keras.utils import Progbar


def validate_dti_segs(subject_list,
                path_label_list,
                path_group_list,
                model_file,
                unet_feat_count=24,
                n_levels=5,
                conv_size=3,
                feat_multiplier=2,
                nb_conv_per_level=2,
                activation='elu',
                bounding_box_width=128,
                seg_selection='grouped',
                dice_version='grouped'):


    # check type of one-hot encoding
    assert (seg_selection == 'single') or (seg_selection == 'combined') \
            or (seg_selection == 'mode') or (seg_selection == 'grouped') ,\
            'seg_selection must be single, combined, mode or grouped'

    assert (dice_version == "grouped") | (dice_version == "individual"), \
        'dice version must be grouped or individual'

    # Load label list
    label_list = np.load(path_label_list).astype(int)
    mapping = np.zeros(1 + label_list[-1], dtype='int')
    mapping[label_list] = np.arange(len(label_list))
    mapping = torch.tensor(mapping, device='cpu').long()

    # Load groups if grouping enabled in validation run
    if path_group_list is not None:
        group_seg = np.load(path_group_list)
        n_groups = group_seg.max() + 1
    else:
        group_seg = None
        n_groups = None

    if seg_selection == 'grouped':
        grp_mat = torch.zeros(group_seg.shape[0], group_seg.max() + 1, dtype=torch.float64)
        for il in range(0, group_seg.shape[0]):
            grp_mat[il, group_seg[il]] = 1
    else:
        grp_mat = None

    # Get loss calculator
    if dice_version == 'individual':
        loss_calculator = metrics.DiceLoss()
    else:
        loss_calculator = metrics.DiceLossGrouped(group_seg, n_groups)

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
                             input_model=None)

    unet_model.load_weights(model_file, by_name=True)

    # keep track of losses
    subj_no = 0
    loss_vector = np.zeros((len(subject_list),1))

    progressbar = Progbar(len(subject_list))

    ### iteratre over subjects
    for subject in subject_list:

        subject_name = os.path.split(subject)[1]
        # get filepaths for data
        t1_file = os.path.join(subject, subject_name[:-3] + '.t1.nii.gz')
        fa_file = os.path.join(subject, 'dmri', subject_name + '_fa.nii.gz')
        v1_file = os.path.join(subject, 'dmri', subject_name + '_v1.nii.gz')

        # get list of segmentations
        seg_list = sorted(glob.glob(subject + '/segs/*nii.gz'))
        
        # Read in and reorient T1
        t1, aff, _ = utils.load_volume(t1_file, im_only=False)
        t1[t1 < 0] = 0
        t1[t1 > 1] = 1

        nx,ny,nz = t1.shape

        # Find the center of the thalamus and crop a volumes around it
        i1 = (np.round(0.5 * (nx - bounding_box_width))).astype(int)
        j1 = (np.round(0.5 * (ny - bounding_box_width))).astype(int)
        k1 = (np.round(0.5 * (nz - bounding_box_width))).astype(int)
        i2 = i1 + bounding_box_width
        j2 = j1 + bounding_box_width
        k2 = k1 + bounding_box_width

        t1 = t1[i1:i2, j1:j2, k1:k2]
        aff[:-1, -1] = aff[:-1, -1] + np.matmul(aff[:-1, :-1], np.array([i1, j1, k1])) # preserve the RAS coordinates

        # Now the diffusion data
        fa = utils.load_volume(fa_file, im_only=True)
        v1 = utils.load_volume(v1_file, im_only=True)

        fa = fa[i1:i2, j1:j2, k1:k2]
        v1 = v1[i1:i2, j1:j2, k1:k2, :]

        dti = np.abs(v1 * fa[..., np.newaxis])


        # Predict with left-right flipping augmentation
        input = np.concatenate((t1[..., np.newaxis], fa[..., np.newaxis], dti ), axis=-1)[np.newaxis,...]
        posteriors = unet_model.predict(input)


        # read in source segs and calculate loss
        seg = utils.load_volume(seg_list[0])
        seg = torch.tensor(seg, device='cpu').long()
        seg = seg[..., None]
        for il in range(1, len(seg_list)):
            np_seg = utils.load_volume(seg_list[il])
            seg = torch.concat((seg, torch.tensor(np_seg[..., None], device='cpu')), dim=3)

        seg = seg[i1:i2, j1:j2, k1:k2, :]

        if seg_selection == 'single':
            seg_loss_vector = np.zeros((len(seg_list),1))
            for il in range(0,len(seg_list)):
                target = encode_onehot(mapping, torch.squeeze(seg[...,il]), label_list, seg_selection, grp_mat).float().detach().numpy()
                seg_loss_vector[il] = loss_calculator.loss(target[None,...],posteriors)

            subj_loss = seg_loss_vector.mean()
        else:
            target = encode_onehot(mapping, seg, label_list, seg_selection, grp_mat).float().detach().numpy()
            subj_loss = loss_calculator.loss(target[None,...],posteriors)

            subj_loss = tf.math.reduce_mean(subj_loss)

        loss_vector[subj_no] = subj_loss

        progressbar.add(1, values=[("validation loss", subj_loss)])

        subj_no += 1

    mean_val_loss = loss_vector.mean()
    return  mean_val_loss, loss_vector

