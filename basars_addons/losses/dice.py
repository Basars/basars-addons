import tensorflow as tf

from tensorflow.keras.losses import Loss, Reduction


class Dice(Loss):

    def __init__(self,
                 num_classes=1,
                 epsilon=1e-5,
                 reduction: str = Reduction.AUTO,
                 from_logits=False,
                 name=None):
        super(Dice, self).__init__(reduction, name)

        self.num_classes = num_classes
        self.epsilon = epsilon
        self.from_logits = from_logits

    def dice_coefficient(self, y_true, y_pred):
        intersection = tf.reduce_sum(y_true * y_pred, axis=1)
        y_sum = tf.reduce_sum(y_true * y_true, axis=1)
        z_sum = tf.reduce_sum(y_pred * y_pred, axis=1)
        return 1 - (2 * intersection + self.epsilon) / (z_sum + y_sum + self.epsilon)

    def call(self, y_true, y_pred):
        if self.from_logits:
            y_pred = tf.nn.softmax(y_pred)

        batch_size = tf.shape(y_pred)[0]

        y_true = tf.reshape(y_true, (batch_size, -1, self.num_classes))
        y_pred = tf.reshape(y_pred, (batch_size, -1, self.num_classes))

        return self.dice_coefficient(y_true, y_pred)