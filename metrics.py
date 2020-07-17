import tensorflow as tf
from tensorflow import keras


def get_accuracy(y_true, y_pred, batch_size):
    y_pred = tf.sigmoid(y_pred)
    inter = tf.reduce_sum(tf.cast((y_pred >= 0.5) & (y_true == 1), dtype=tf.float32))
    union = tf.reduce_sum(tf.cast((y_pred >= 0.5) | (y_true == 1), dtype=tf.float32))
    union += 1e-8
    return inter / union


def get_precision(y_true, y_pred, batch_size):
    y_pred = tf.sigmoid(y_pred)
    inter = tf.reduce_sum(tf.cast((y_pred >= 0.5) & (y_true == 1), dtype=tf.float32))
    pred_sum = tf.reduce_sum(tf.cast(y_pred >= 0.5, dtype=tf.float32))
    pred_sum += 1e-8
    return inter / pred_sum


def get_recall(y_true, y_pred, batch_size):
    y_pred = tf.sigmoid(y_pred)
    inter = tf.reduce_sum(tf.cast((y_pred >= 0.5) & (y_true == 1), dtype=tf.float32))
    true_sum = tf.reduce_sum(tf.cast(y_true == 1, dtype=tf.float32))
    true_sum += 1e-8
    return inter/true_sum


def get_f1_score(y_true, y_pred, batch_size):
    y_pred = tf.sigmoid(y_pred)
    inter = tf.reduce_sum(tf.cast((y_pred >= 0.5) & (y_true == 1), dtype=tf.float32))
    true_sum = tf.reduce_sum(tf.cast(y_true == 1, dtype=tf.float32))
    pred_sum = tf.reduce_sum(tf.cast(y_pred >= 0.5, dtype=tf.float32))
    return (2 * inter) / (true_sum + pred_sum + 1e-10)


def get_hamming_loss(y_true, y_pred, batch_size):
    y_pred = tf.sigmoid(y_pred)
    hamming = tf.math.logical_xor((y_pred >= 0.5), (y_true == 1))
    hamming = tf.reduce_sum(tf.cast(hamming, dtype=tf.float32))
    return hamming / (batch_size * y_true.shape[1])


class MultiLabelAccuracy(keras.metrics.Metric):
    def __init__(self, name='multi_label_accuracy', **kwargs):
        super(MultiLabelAccuracy, self).__init__(name=name, **kwargs)
        self.accuracy = self.add_weight(name='tp', initializer='zeros')
        self.num_iter = self.add_weight(name='tp', initializer='zeros')
        self.batch_size = kwargs['batch_size']

    def update_state(self, y_true, y_pred, sample_weight=None):
        accuracy = get_accuracy(y_true, y_pred, self.batch_size)
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
        precision = get_precision(y_true, y_pred, self.batch_size)
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
        recall = get_recall(y_true, y_pred, self.batch_size)
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
        f1_score = get_f1_score(y_true, y_pred, self.batch_size)
        self.f1_score.assign_add(f1_score)
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
        hamming_loss = get_hamming_loss(y_true, y_pred, self.batch_size)
        self.hamming_loss.assign_add(hamming_loss)
        self.num_iter.assign_add(1.0)

    def result(self):
        return self.hamming_loss/self.num_iter

    def reset_states(self):
        self.hamming_loss.assign(0.)
        self.num_iter.assign(0.)