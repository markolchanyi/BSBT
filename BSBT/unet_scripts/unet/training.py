import os
import glob
import keras
import keras.callbacks as KC
import numpy as np
import tensorflow as tf
from keras import models as KM
from keras.optimizers import Adam
import metrics
import models
import nibabel as nib
from generators import image_seg_generator_rgb, image_seg_generator_rgb_validation
from tensorflow.keras.callbacks import Callback as tf_callback


class SaveAttentionLayerOutput(tf_callback):
    def __init__(self, model, layer_name, save_path):
        super().__init__()
        self.model = model
        self.layer_name = layer_name
        self.save_path = save_path
        self.attention_layer = model.get_layer(layer_name).output
        # Create a function to fetch the output of the attention layer
        self.get_attention_output = tf.keras.backend.function([model.input], [self.attention_layer])

    def on_epoch_end(self, epoch, logs=None):
        # Generate some input data or use validation data here
        # input_data = ...

        # Get the output of the attention layer
        attention_output = self.get_attention_output([input_data])[0]

        # Convert the output to a Numpy array
        attention_output_np = np.array(attention_output)

        # Save the numpy array as a NIfTI file
        affine = np.eye(4)  # Replace with the correct affine matrix for your data
        attention_nifti = nib.Nifti1Image(attention_output_np, affine)
        nib.save(attention_nifti, f'{self.save_path}/attention_output_epoch_{epoch}.nii.gz')
        print(f'Attention layer output saved for epoch {epoch}')



def train(training_dir,
             path_label_list,
             model_dir,
             path_group_list=None,
             validation_dir=None,
             batchsize=1,
             crop_size=128,
             scaling_bounds=0.15,
             rotation_bounds=15,
             nonlinear_rotation=False,
             max_noise_std=0.1,
             max_noise_std_fa=0.03,
             gamma_std=0.1,
             contrast_std=0.1,
             brightness_std=0.1,
             randomize_resolution=False,
             generator_mode='rgb',
             diffusion_resolution=None,
             speckle_frac_selected=1e-4,
             seg_selection='grouped',
             flag_deformation=True,
             deformation_max=5.0,
             n_levels=5,
             nb_conv_per_level=2,
             conv_size=3,
             unet_feat_count=24,
             feat_multiplier=2,
             dropout=0,
             attention_gating=True,
             ablate_dti=False,
             activation='elu',
             lr=1e-4,
             lr_decay=0,
             wl2_epochs=5,
             dice_epochs=200,
             steps_per_epoch=1000,
             checkpoint=None,
             checkpoint_l2=None,
             checkpoint_dice=None,
             dice_version="grouped",
             checkpoint_frequency=1):

    # check epochs
    assert (wl2_epochs > 0) | (dice_epochs > 0), \
        'either wl2_epochs or dice_epochs must be positive, had {0} and {1}'.format(wl2_epochs, dice_epochs)
    # check generator mode
    assert (generator_mode == 'fa_v1') | (generator_mode == 'rgb'), \
        'generator mode must be fa_v1 or rgb'

    assert (dice_version == "grouped") | (dice_version == "individual"), \
        'dice version must be grouped or individual'

    if diffusion_resolution is not None:
        if type(diffusion_resolution) == int:
            diffusion_resolution = [diffusion_resolution] * 3

    if checkpoint is None :
        if (checkpoint_dice is None) & (checkpoint_l2 is None):
            checkpoint_list=sorted(glob.glob(model_dir + '/dice_???.h5'))
            if len(checkpoint_list)!=0 :
                checkpoint_dice = checkpoint_list[-1]
            else :
                checkpoint_list=sorted(glob.glob(model_dir + '/wl2_???.h5'))
                if len(checkpoint_list)!=0 :
                    checkpoint_l2 = checkpoint_list[-1]

        if checkpoint_l2 is not None:
            checkpoint = checkpoint_l2


    if generator_mode == 'rgb':
        generator = image_seg_generator_rgb(training_dir,
                                path_label_list,
                                path_group_list,
                                batchsize=batchsize,
                                scaling_bounds=scaling_bounds,
                                rotation_bounds=rotation_bounds,
                                nonlinear_rotation=nonlinear_rotation,
                                max_noise_std=max_noise_std,
                                max_noise_std_fa=max_noise_std_fa,
                                gamma_std=gamma_std,
                                contrast_std=contrast_std,
                                brightness_std=brightness_std,
                                crop_size=crop_size,
                                randomize_resolution=randomize_resolution,
                                diffusion_resolution=diffusion_resolution,
                                speckle_frac_selected=speckle_frac_selected,
                                seg_selection=seg_selection,
                                flag_deformation=flag_deformation,
                                deformation_max=deformation_max,
                                ablate_dti=ablate_dti)
    else:
        print("only an RGB generator is inlcuded in this training script. Sorry!")

    label_list = np.sort(np.load(path_label_list)).astype(int)
    n_labels = np.size(label_list)

    if validation_dir is not None:
        validation_generator = image_seg_generator_rgb_validation(validation_dir,
                                path_label_list,
                                batchsize=batchsize,
                                scaling_bounds=scaling_bounds,
                                rotation_bounds=rotation_bounds,
                                max_noise_std=max_noise_std,
                                max_noise_std_fa=max_noise_std_fa,
                                gamma_std=gamma_std,
                                contrast_std=contrast_std,
                                brightness_std=brightness_std,
                                crop_size=crop_size,
                                randomize_resolution=randomize_resolution,
                                diffusion_resolution=diffusion_resolution)
    else:
        validation_generator = None

    if path_group_list is not None:
        group_seg = np.load(path_group_list)
        n_groups = group_seg.max() + 1
    else:
        group_seg = None
        n_groups = None


    if crop_size is None:
        aux = next(generator)
        crop_size = aux[0].shape[1:-1]
    if type(crop_size) == int:
        crop_size = [crop_size] * 3

    # only for the ablation step where if true, training occurs for lowB and FA only!
    if not ablate_dti:
        unet_input_shape = [*crop_size, 5]
    else:
        unet_input_shape = [*crop_size, 2]

    unet_model = models.unet(nb_features=unet_feat_count,
                                 input_shape=unet_input_shape,
                                 nb_levels=n_levels,
                                 conv_size=conv_size,
                                 nb_labels=n_labels,
                                 feat_mult=feat_multiplier,
                                 nb_conv_per_level=nb_conv_per_level,
                                 conv_dropout=dropout,
                                 batch_norm=-1,
                                 activation=activation,
                                 attention_gating=attention_gating,
                                 input_model=None)

    # pre-training with weighted L2, input is fit to the softmax rather than the probabilities
    if (wl2_epochs > 0) & (checkpoint_dice is None):
        wl2_model = models.Model(unet_model.inputs, [unet_model.get_layer('unet_likelihood').output])
        train_model(wl2_model, generator, lr, lr_decay, wl2_epochs, steps_per_epoch, model_dir, 'wl2', n_labels, checkpoint)
        checkpoint_dice = os.path.join(model_dir, 'wl2_%03d.h5' % wl2_epochs)

    # fine-tuning with dice metric
    train_model(unet_model, generator, lr, lr_decay, dice_epochs, steps_per_epoch, model_dir, 'dice', n_labels,
                group_seg, n_groups, checkpoint_dice, validation_generator=validation_generator, dice_version=dice_version,
                checkpoint_frequency=checkpoint_frequency)

    print('All done!')


def train_model(model,
                generator,
                learning_rate,
                lr_decay,
                n_epochs,
                n_steps,
                model_dir,
                metric_type,
                n_labels,
                group_seg=None,
                n_groups=None,
                path_checkpoint=None,
                validation_generator=None,
                dice_version="grouped",
                checkpoint_frequency=1,
                reinitialise_momentum=False):

    # prepare log folder
    log_dir = os.path.join(model_dir, 'logs')
    if os.path.exists(model_dir) is False:
        os.mkdir(model_dir)
    if os.path.exists(log_dir) is False:
        os.mkdir(log_dir)

    # model saving callback
    save_file_name = os.path.join(model_dir, '%s_{epoch:03d}.h5' % metric_type)
    callbacks = [KC.ModelCheckpoint(save_file_name, verbose=1, period=checkpoint_frequency)]

    # TensorBoard callback
    if metric_type == 'dice':
        callbacks.append(KC.TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=False))

    # Attention gating weight saving callback
    #save_attention_output_callback = SaveAttentionLayerOutput(model, 'attn_coeffs', '/autofs/space/nicc_003/users/olchanyi/BSB/unet_scripts/scripts')

    compile_model = True
    init_epoch = 0
    if path_checkpoint is not None:
        print('Checkpoint found:')
        print(path_checkpoint)
        if metric_type in path_checkpoint:
            init_epoch = int(os.path.basename(path_checkpoint).split(metric_type)[1][1:-3])
        if (not reinitialise_momentum) & (metric_type in path_checkpoint):
            if metric_type == 'dice':
                if dice_version == "grouped":
                    custom_objects = {'tf': tf, 'keras': keras, 'loss': metrics.DiceLossGrouped(group_seg, n_groups).loss}
                else :
                    custom_objects = {'tf': tf, 'keras': keras, 'loss': metrics.DiceLoss().loss}
            else :
                custom_objects = {'tf': tf, 'keras': keras, 'loss': metrics.WL2Loss(3.0, n_labels, background_weight=0.01).loss}
            model = KM.load_model(path_checkpoint, custom_objects=custom_objects)
            compile_model = False
        else:
            model.load_weights(path_checkpoint, by_name=True)


    # compile
    if compile_model :
        if metric_type == 'dice':
            if dice_version=="grouped":
                assert (group_seg is not None) & (n_groups is not None), \
                    "grouped Dice requires thalamic nuclei be grouped in a file provided by group_seg"

                model.compile(optimizer=Adam(lr=learning_rate, decay=lr_decay),
                          loss=metrics.DiceLossGrouped(group_seg, n_groups).loss,
                          loss_weights=[1.0])

            else:
                model.compile(optimizer=Adam(lr=learning_rate, decay=lr_decay),
                              loss=metrics.DiceLoss().loss,
                              loss_weights=[1.0])
        else:
            model.compile(optimizer=Adam(lr=learning_rate, decay=lr_decay),
                          loss=metrics.WL2Loss(3.0, n_labels, background_weight=0.01).loss, # since we crop, 0.01 is ok
                          loss_weights=[1.0])

    # fit
    if validation_generator is not None:
        validation_freq = list(range(1,6))

        for il in range(10, n_epochs, 5):
            validation_freq.append(il)

        validation_freq.append(n_epochs)

        model.fit(generator,epochs=n_epochs,
                            steps_per_epoch=n_steps,
                            callbacks=callbacks,
                            initial_epoch=init_epoch,
                            use_multiprocessing=False,
                            validation_data=tuple(validation_generator),
                            validation_steps=1,
                            validation_freq=validation_freq)
    else:
        model.fit(generator,epochs=n_epochs,
                            steps_per_epoch=n_steps,
                            callbacks=callbacks,
                            initial_epoch=init_epoch,
                            use_multiprocessing=True)
