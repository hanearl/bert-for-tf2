import os
import math

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras

from bert import BertModelLayer
from bert.loader import StockBertConfig, map_stock_config_to_params, load_stock_weights

from metrics import MultiLabelAccuracy, MultiLabelPrecision,\
                            MultiLabelRecall, MultiLabelF1, HammingLoss


def flatten_layers(root_layer):
    if isinstance(root_layer, keras.layers.Layer):
        yield root_layer
    for layer in root_layer._layers:
        for sub_layer in flatten_layers(layer):
            yield sub_layer


def freeze_bert_layers(l_bert):
    """
    Freezes all but LayerNorm and adapter layers - see arXiv:1902.00751.
    """
    for layer in flatten_layers(l_bert):
        if layer.name in ["LayerNorm", "adapter-down", "adapter-up"]:
            layer.trainable = True
        elif len(layer._layers) == 0:
            layer.trainable = False
        l_bert.embeddings_layer.trainable = False


def create_learning_rate_scheduler(max_learn_rate=5e-5,
                                   end_learn_rate=1e-7,
                                   warmup_epoch_count=10,
                                   total_epoch_count=90):

    def lr_scheduler(epoch):
        if epoch < warmup_epoch_count:
            res = (max_learn_rate/warmup_epoch_count) * (epoch + 1)
        else:
            res = max_learn_rate*math.exp(
                math.log(end_learn_rate/max_learn_rate)*(epoch-warmup_epoch_count+1)/(total_epoch_count-warmup_epoch_count+1))
        return float(res)
    learning_rate_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=1)

    return learning_rate_scheduler


def get_cls_weight():
    return np.array([0.94829875, 0.9763115 , 0.98265562, 0.8767022 , 0.99662406,
       0.99833099, 0.97284072, 0.9967663 , 0.97057429, 0.97911846,
       0.96726473, 0.98885749, 0.97259417, 0.97139931, 0.94361416,
       0.97229071, 0.95213936, 0.89593369, 0.99282138, 0.98731176,
       0.94005804, 0.96403103, 0.92873535, 0.96460001, 0.96383189,
       0.99580852, 0.99823616, 0.9925843 , 0.99441452, 0.99229981,
       0.9662785 , 0.97193036, 0.99304897, 0.9916929 ])

def create_model(config, adapter_size=64):
    """Creates a classification model."""

    # create the bert layer
    with tf.io.gfile.GFile(config.bert_config_file, "r") as reader:
        bc = StockBertConfig.from_json_string(reader.read())
        bert_params = map_stock_config_to_params(bc)
        bert_params.adapter_size = adapter_size
        bert = BertModelLayer.from_params(bert_params, name="bert")

    input_ids = keras.layers.Input(shape=(config.max_seq_len,), dtype='int32', name="input_ids")
    output = bert(input_ids)

    matmul_qk = tf.matmul(output, output, transpose_b=True)
    attention_weights = tf.nn.softmax(matmul_qk, axis=-1)
    logits = tf.matmul(attention_weights, output)
    logits = tf.reduce_sum(logits, axis=1) * config.attn_weight

    cls_out = keras.layers.Lambda(lambda seq: seq[:, 0, :])(output) * config.cls_weight
    logits = cls_out + logits

    logits = keras.layers.Dropout(0.5)(logits)
    logits = keras.layers.LayerNormalization()(logits)
    logits = keras.layers.Dense(units=len(config.classes))(logits)

    model = keras.Model(inputs=input_ids, outputs=logits)
    model.build(input_shape=(None, config.max_seq_len))

    # load the pre-trained model weights
    load_stock_weights(bert, config.bert_ckpt_file)

    # freeze weights if adapter-BERT is used
    # if adapter_size is not None:
    #     freeze_bert_layers(bert)

    sigmoid_cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True,
                                                               label_smoothing=config.label_smoothing)

    def tfa_focal_loss(y_true, y_pred):
        cls_weight = None
        loss = tfa.losses.SigmoidFocalCrossEntropy(alpha=config.focal_alpha,
                                                         gamma=config.focal_gamma,
                                                         from_logits=True)
        y_true = (1 - config.label_smoothing) * y_true + config.label_smoothing / len(config.classes)
        print('y_true', y_true.shape)
        x = loss(y_true, y_pred)
        print('x', x.shape)
        x *= cls_weight
        print('x2', x.shape)
        return x

    loss_func_list = {
        "sigmoid_cross_entropy_loss": sigmoid_cross_entropy,
        "focal_loss": tfa_focal_loss
    }

    model.compile(optimizer=keras.optimizers.Adam(),
                  loss=loss_func_list[config.loss_func],
                  metrics=[MultiLabelAccuracy(batch_size=config.batch_size),
                           MultiLabelPrecision(batch_size=config.batch_size),
                           MultiLabelRecall(batch_size=config.batch_size),
                           MultiLabelF1(batch_size=config.batch_size),
                           HammingLoss(batch_size=config.batch_size)])

    model.summary()

    return model
