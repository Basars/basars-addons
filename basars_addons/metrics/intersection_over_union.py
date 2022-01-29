import tensorflow as tf

from tensorflow.keras.metrics import MeanIoU


class ThresholdBinaryIoU(MeanIoU):

    def __init__(self, num_classes, threshold=0.5, name=None, dtype=None):
        super(ThresholdBinaryIoU, self).__init__(num_classes=num_classes, name=name, dtype=dtype)

        self.threshold = threshold

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.cast(y_pred >= self.threshold, self._dtype)

        super(ThresholdBinaryIoU, self).update_state(y_true, y_pred, sample_weight=sample_weight)
