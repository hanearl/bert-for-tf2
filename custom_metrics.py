import tensorflow as tf
from tensorflow import keras


class MultiLabelAccuracy(keras.metrics.Metric):
    def __init__(self, name='multi_label_accuracy', **kwargs):
        super(MultiLabelAccuracy, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name='tp', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        # y_true = tf.cast(y_true, dtype=tf.float32)
        correct = tf.reduce_sum(tf.cast((y_pred > 0.5) & (y_true == 1), dtype=tf.float32))
        self.true_positives.assign(tf.reduce_sum(correct)/tf.reduce_sum(y_true))

    def result(self):
        return self.true_positives

    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self.true_positives.assign(0.)