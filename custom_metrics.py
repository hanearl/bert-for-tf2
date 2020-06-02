import tensorflow as tf
from tensorflow import keras


class MultiLabelAccuracy(keras.metrics.Metric):
    def __init__(self, name='multi_label_accuracy', **kwargs):
        super(MultiLabelAccuracy, self).__init__(name=name, **kwargs)
        self.acc = self.add_weight(name='tp', initializer='zeros')
        self.num_acc = self.add_weight(name='tp', initializer='zeros')


    def update_state(self, y_true, y_pred, sample_weight=None):
        # y_true = tf.cast(y_true, dtype=tf.float32)
        correct = tf.reduce_sum(tf.cast((y_pred > 0.5) & (y_true == 1), dtype=tf.float32))
        self.acc.assign_add(tf.reduce_sum(correct)/tf.reduce_sum(y_true))
        self.num_acc.assign_add(1.0)

    def result(self):
        return self.acc/self.num_acc

    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self.acc.assign(0.)
        self.num_acc.assign(0.)