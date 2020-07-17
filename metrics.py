import tensorflow as tf
from tensorflow import keras


def get_hamming_loss(y_true, y_pred, batch_size):
    y_pred = tf.sigmoid(y_pred)
    hamming = tf.math.logical_xor((y_pred >= 0.5), (y_true == 1))
    hamming = tf.reduce_sum(tf.cast(hamming, dtype=tf.float32))
    return hamming / (batch_size * y_true.shape[1])


class MultiLabelAccuracy(keras.metrics.Metric):
    def __init__(self, name='multi_label_accuracy', **kwargs):
        super(MultiLabelAccuracy, self).__init__(name=name, **kwargs)
        self.inter = self.add_weight(name='tp', initializer='zeros')
        self.union = self.add_weight(name='tp', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.sigmoid(y_pred)
        inter = tf.reduce_sum(tf.cast((y_pred >= 0.5) & (y_true == 1), dtype=tf.float32))
        union = tf.reduce_sum(tf.cast((y_pred >= 0.5) | (y_true == 1), dtype=tf.float32))
        self.inter.assign_add(inter)
        self.union.assign_add(union)

    def result(self):
        return self.inter/(self.union + 1e-8)

    def reset_states(self):
        self.inter.assign(0.)
        self.union.assign(0.)


class MultiLabelPrecision(keras.metrics.Metric):
    def __init__(self, name='multi_label_precision', **kwargs):
        super(MultiLabelPrecision, self).__init__(name=name, **kwargs)
        self.inter = self.add_weight(name='tp', initializer='zeros')
        self.pred_sum = self.add_weight(name='tp', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):

        y_pred = tf.sigmoid(y_pred)
        inter = tf.reduce_sum(tf.cast((y_pred >= 0.5) & (y_true == 1), dtype=tf.float32))
        pred_sum = tf.reduce_sum(tf.cast(y_pred >= 0.5, dtype=tf.float32))
        self.inter.assign_add(inter)
        self.pred_sum.assign_add(pred_sum)

    def result(self):
        return self.inter / (self.pred_sum + 1e-8)

    def reset_states(self):
        self.inter.assign(0.)
        self.pred_sum.assign(0.)


class MultiLabelRecall(keras.metrics.Metric):
    def __init__(self, name='multi_label_recall', **kwargs):
        super(MultiLabelRecall, self).__init__(name=name, **kwargs)
        self.inter = self.add_weight(name='tp', initializer='zeros')
        self.true_sum = self.add_weight(name='tp', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        inter = tf.reduce_sum(tf.cast((y_pred >= 0.5) & (y_true == 1), dtype=tf.float32))
        true_sum = tf.reduce_sum(tf.cast(y_true == 1, dtype=tf.float32))
        self.inter.assign_add(inter)
        self.true_sum.assign_add(true_sum)

    def result(self):
        return self.inter/(self.true_sum+1e-8)

    def reset_states(self):
        self.inter.assign(0.)
        self.true_sum.assign(0.)


class MultiLabelF1(keras.metrics.Metric):
    def __init__(self, name='multi_label_f1_score', **kwargs):
        super(MultiLabelF1, self).__init__(name=name, **kwargs)
        self.inter = self.add_weight(name='tp', initializer='zeros')
        self.true_sum = self.add_weight(name='tp', initializer='zeros')
        self.pred_sum = self.add_weight(name='tp', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.sigmoid(y_pred)
        inter = tf.reduce_sum(tf.cast((y_pred >= 0.5) & (y_true == 1), dtype=tf.float32))
        true_sum = tf.reduce_sum(tf.cast(y_true == 1, dtype=tf.float32))
        pred_sum = tf.reduce_sum(tf.cast(y_pred >= 0.5, dtype=tf.float32))
        self.inter.assign_add(inter)
        self.true_sum.assign_add(true_sum)
        self.pred_sum.assign_add(pred_sum)

    def result(self):
        return (2 * self.inter) / (self.true_sum + self.true_sum + 1e-10)

    def reset_states(self):
        self.inter.assign(0.)
        self.true_sum.assign(0.)
        self.pred_sum.assign(0.)


class HammingLoss(keras.metrics.Metric):
    def __init__(self, name='hamming_loss', **kwargs):
        super(HammingLoss, self).__init__(name=name, **kwargs)
        self.hamming_loss = self.add_weight(name='tp', initializer='zeros')
        self.num_iter = self.add_weight(name='tp', initializer='zeros')
        self.batch_size = kwargs['batch_size']

    def update_state(self, y_true, y_pred, sample_weight=None):
        hamming_loss = get_hamming_loss(y_true, y_pred, self.batch_size)
        self.hamming_loss.assign_add(hamming_loss)
        self.num_iter.assign_add(1.0)

    def result(self):
        return self.hamming_loss/self.num_iter

    def reset_states(self):
        self.hamming_loss.assign(0.)
        self.num_iter.assign(0.)