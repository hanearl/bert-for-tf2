import tensorflow as tf
from tensorflow import keras


class MultiLabelAccuracy(keras.metrics.Metric):
    def __init__(self, name='multi_label_accuracy', **kwargs):
        super(MultiLabelAccuracy, self).__init__(name=name, **kwargs)
        self.accuracy = self.add_weight(name='tp', initializer='zeros')
        self.num_iter = self.add_weight(name='tp', initializer='zeros')
        self.batch_size = kwargs['batch_size']

    def update_state(self, y_true, y_pred, sample_weight=None):
        inter = tf.reduce_sum(tf.cast((y_pred > 0.5) & (y_true == 1), dtype=tf.float32))
        union = tf.reduce_sum(tf.cast((y_pred > 0.5) | (y_true == 1), dtype=tf.float32))
        union += 1e-8
        accuracy = tf.reduce_sum(inter/union) / self.batch_size
        self.accuracy.assign_add(accuracy)
        self.num_iter.assign_add(1.0)

    def result(self):
        return self.accuracy/self.num_iter

    def reset_states(self):
        self.accuracy.assign(0.)
        self.num_iter.assign(0.)


class MultiLabelPrecision(keras.metrics.Metric):
    def __init__(self, name='multi_label_precision', **kwargs):
        super(MultiLabelPrecision, self).__init__(name=name, **kwargs)
        self.precision = self.add_weight(name='tp', initializer='zeros')
        self.num_iter = self.add_weight(name='tp', initializer='zeros')
        self.batch_size = kwargs['batch_size']

    def update_state(self, y_true, y_pred, sample_weight=None):
        inter = tf.reduce_sum(tf.cast((y_pred > 0.5) & (y_true == 1), dtype=tf.float32), axis=-1)
        pred_sum = tf.reduce_sum(tf.cast(y_pred > 0.5, dtype=tf.float32), axis=-1)
        pred_sum += 1e-8
        precision = tf.reduce_sum(inter/pred_sum) / self.batch_size
        self.precision.assign_add(precision)
        self.num_iter.assign_add(1.0)

    def result(self):
        return self.precision/self.num_iter

    def reset_states(self):
        self.precision.assign(0.)
        self.num_iter.assign(0.)


class MultiLabelRecall(keras.metrics.Metric):
    def __init__(self, name='multi_label_recall', **kwargs):
        super(MultiLabelRecall, self).__init__(name=name, **kwargs)
        self.recall = self.add_weight(name='tp', initializer='zeros')
        self.num_iter = self.add_weight(name='tp', initializer='zeros')
        self.batch_size = kwargs['batch_size']

    def update_state(self, y_true, y_pred, sample_weight=None):
        inter = tf.reduce_sum(tf.cast((y_pred > 0.5) & (y_true == 1), dtype=tf.float32), axis=-1)
        true_sum = tf.reduce_sum(tf.cast(y_true == 1, dtype=tf.float32), axis=-1)
        true_sum += 1e-8
        recall = tf.reduce_sum(inter/true_sum) / self.batch_size
        self.recall.assign_add(recall)
        self.num_iter.assign_add(1.0)

    def result(self):
        return self.recall/self.num_iter

    def reset_states(self):
        self.recall.assign(0.)
        self.num_iter.assign(0.)


class MultiLabelF1(keras.metrics.Metric):
    def __init__(self, name='multi_label_f1_score', **kwargs):
        super(MultiLabelF1, self).__init__(name=name, **kwargs)
        self.f1_score = self.add_weight(name='tp', initializer='zeros')
        self.num_iter = self.add_weight(name='tp', initializer='zeros')
        self.batch_size = kwargs['batch_size']

    def update_state(self, y_true, y_pred, sample_weight=None):
        inter = tf.reduce_sum(tf.cast((y_pred > 0.5) & (y_true == 1), dtype=tf.float32), axis=-1)
        true_sum = tf.reduce_sum(tf.cast(y_true == 1, dtype=tf.float32), axis=-1)
        pred_sum = tf.reduce_sum(tf.cast(y_pred > 0.5, dtype=tf.float32), axis=-1)
        f1_score = tf.reduce_sum((2 * inter) / (true_sum + pred_sum + 1e-10))
        self.f1_score.assign_add(f1_score / self.batch_size)
        self.num_iter.assign_add(1.0)

    def result(self):
        return self.f1_score/self.num_iter

    def reset_states(self):
        self.f1_score.assign(0.)
        self.num_iter.assign(0.)


class HammingLoss(keras.metrics.Metric):
    def __init__(self, name='hamming_loss', **kwargs):
        super(HammingLoss, self).__init__(name=name, **kwargs)
        self.hamming_loss = self.add_weight(name='tp', initializer='zeros')
        self.num_iter = self.add_weight(name='tp', initializer='zeros')
        self.batch_size = kwargs['batch_size']

    def update_state(self, y_true, y_pred, sample_weight=None):
        hamming = tf.math.logical_xor((y_pred > 0.5), (y_true == 1))
        hamming = tf.reduce_sum(tf.cast(hamming, dtype=tf.float32))
        self.hamming_loss.assign_add(hamming / (self.batch_size * y_true.shape[1]))
        self.num_iter.assign_add(1.0)

    def result(self):
        return self.hamming_loss/self.num_iter

    def reset_states(self):
        self.hamming_loss.assign(0.)
        self.num_iter.assign(0.)