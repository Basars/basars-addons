import tensorflow as tf

from tensorflow.keras.metrics import Precision, Recall


class ThresholdPrecision(Precision):

    def __init__(self,
                 threshold=0.5, thresholds=None, top_k=None, class_id=None, from_logits=False,
                 name=None, dtype=None):
        super(ThresholdPrecision, self).__init__(thresholds, top_k, class_id, name, dtype)

        self.threshold = threshold
        self.from_logits = from_logits

    def update_state(self, y_true, y_pred, sample_weight=None):
        if self.from_logits:
            y_pred = tf.nn.softmax(y_pred, axis=-1)

        y_pred = tf.cast(y_pred >= self.threshold, self._dtype)

        super(ThresholdPrecision, self).update_state(y_true, y_pred, sample_weight=sample_weight)


class ThresholdRecall(Recall):

    def __init__(self,
                 threshold=0.5, thresholds=None, top_k=None, class_id=None, from_logits=False,
                 name=None, dtype=None):
        super(ThresholdRecall, self).__init__(thresholds, top_k, class_id, name, dtype)

        self.threshold = threshold
        self.from_logits = from_logits

    def update_state(self, y_true, y_pred, sample_weight=None):
        if self.from_logits:
            y_pred = tf.nn.softmax(y_pred, axis=-1)

        y_pred = tf.cast(y_pred >= self.threshold, self._dtype)

        super(ThresholdRecall, self).update_state(y_true, y_pred, sample_weight=sample_weight)
