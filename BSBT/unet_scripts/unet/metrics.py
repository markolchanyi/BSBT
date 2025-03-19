import keras.backend as K
import tensorflow as tf


class WL2Loss(object):

    def __init__(self, target_value, n_labels, background_weight=1e-4, **kwargs):
        self.target_value = target_value
        self.n_labels = n_labels
        self.background_weight = background_weight

    def loss(self, gt, pred):
        weights = tf.expand_dims(1 - gt[..., 0] + self.background_weight, -1)
        loss = K.sum(weights * K.square(pred - self.target_value * (2 * gt - 1))) / (K.sum(weights) * self.n_labels)
        tf.debugging.check_numerics(loss, 'Loss not finite')
        return loss


class DiceLoss(object):

    def __init__(self, **kwargs):
        pass

    def loss(self, y, x):
        x = K.clip(x / tf.math.reduce_sum(x, axis=-1, keepdims=True), 0, 1)
        y = K.clip(y / tf.math.reduce_sum(y, axis=-1, keepdims=True), 0, 1)
        # compute dice loss for each label
        top = tf.math.reduce_sum(2 * x * y + tf.keras.backend.epsilon(), axis=list(range(1, 4)))
        bottom = tf.math.square(x) + tf.math.square(y) + tf.keras.backend.epsilon()
        bottom = tf.math.reduce_sum(bottom, axis=list(range(1, 4)))
        dice = top / bottom
        loss = 1 - dice
        tf.debugging.check_numerics(loss, 'Loss not finite')
        return loss


class DiceLossLabels(object):

    def __init__(self, **kwargs):
        pass

    def loss(self, y, x):
        x = K.clip(x / tf.math.reduce_sum(x, axis=-1, keepdims=True), 0, 1)
        y = K.clip(y / tf.math.reduce_sum(y, axis=-1, keepdims=True), 0, 1)
        # compute dice loss for each label
        top = tf.math.reduce_sum(2 * x * y + tf.keras.backend.epsilon(), axis=list(range(1, 4)))
        bottom = tf.math.square(x) + tf.math.square(y) + tf.keras.backend.epsilon()
        bottom = tf.math.reduce_sum(bottom, axis=list(range(1, 4)))
        dice = top / bottom
        loss = K.mean(1 - dice)

        x = tf.math.reduce_sum(x[..., 1:], axis=-1)
        y = tf.math.reduce_sum(y[..., 1:], axis=-1)

        top2 = tf.math.reduce_sum(2 * x * y + tf.keras.backend.epsilon(), axis=list(range(1, 4)))
        bottom2 = tf.math.reduce_sum(tf.math.square(x) + tf.math.square(y) + tf.keras.backend.epsilon(), axis=list(range(1, 4)))

        dice2 = top2/bottom2

        loss = (loss + 1 - dice2)/2

        tf.debugging.check_numerics(loss, 'Loss not finite')
        return loss


class DiceLossGrouped(object):

    def __init__(self, group_seg, n_groups, **kwargs):
        self.group_seg = group_seg
        self.n_groups = n_groups
        pass

    def loss(self, y, x):
        x = K.clip(x / tf.math.reduce_sum(x, axis=-1, keepdims=True), 0, 1)
        y = K.clip(y / tf.math.reduce_sum(y, axis=-1, keepdims=True), 0, 1)
        # compute dice loss for each label
        top = tf.math.reduce_sum(2 * x[..., 1:] * y[..., 1:] + tf.keras.backend.epsilon(), axis=list(range(1, 4)))
        bottom = tf.math.square(x[..., 1:]) + tf.math.square(y[..., 1:]) + tf.keras.backend.epsilon()
        bottom = tf.math.reduce_sum(bottom, axis=list(range(1, 4)))
        dice = top / bottom
        loss = K.mean(1 - dice)

        # group together the labels
        x = tf.transpose(tf.math.unsorted_segment_sum(tf.transpose(x), self.group_seg, num_segments=self.n_groups ))
        y = tf.transpose(tf.math.unsorted_segment_sum(tf.transpose(y), self.group_seg, num_segments=self.n_groups ))
        # compute dice loss for each group
        top = tf.math.reduce_sum(2 * x[..., 1:] * y[..., 1:] + tf.keras.backend.epsilon(), axis=list(range(1, 4)))
        bottom = tf.math.square(x[..., 1:]) + tf.math.square(y[..., 1:]) + tf.keras.backend.epsilon()
        bottom = tf.math.reduce_sum(bottom, axis=list(range(1, 4)))

        dice = top/bottom

        loss = loss + K.mean(1 - dice)

        # whole thalamus
        x = tf.math.reduce_sum(x[..., 1:], axis=-1)
        y = tf.math.reduce_sum(y[..., 1:], axis=-1)

        top = tf.math.reduce_sum(2 * x * y + tf.keras.backend.epsilon(), axis=list(range(1, 4)))
        bottom = tf.math.reduce_sum(tf.math.square(x) + tf.math.square(y) + tf.keras.backend.epsilon(),
                                     axis=list(range(1, 4)))

        dice = top / bottom

        loss = (loss + K.mean(1 - dice))/3

        tf.debugging.check_numerics(loss, 'Loss not finite')
        return loss
