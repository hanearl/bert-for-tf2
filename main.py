import os
import sys
import datetime
import pickle
import math
import json

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

import bert
from bert import BertModelLayer
from bert.loader import StockBertConfig, map_stock_config_to_params, load_stock_weights
from bert.tokenization.bert_tokenization import FullTokenizer

from custom_metrics import MultiLabelAccuracy, MultiLabelPrecision,\
                            MultiLabelRecall, MultiLabelF1, HammingLoss
from sentiments_data import SentimentsData

with open('./config.json', 'r') as f:
    config = json.load(f)

drive_path = "/home/hanearl/Desktop"
train_name = config["train_name"]

project_path = os.path.join(drive_path, "bert_sentiment")
bert_model_path = os.path.join(project_path, "bert_model")
data_path = os.path.join(project_path, "data")
epoch_log_path = os.path.join(project_path, "epoch_logs", train_name)
epoch_model_path = os.path.join(project_path, "epoch_models", train_name)
tb_path = os.path.join(project_path, "logs", train_name)

if not os.path.isdir(epoch_log_path):
    os.mkdir(epoch_log_path)

if not os.path.isdir(epoch_model_path):
    os.mkdir(epoch_model_path)

if not os.path.isdir(tb_path):
    os.mkdir(tb_path)

model_name = config['model_name']
model_dir = bert.fetch_google_bert_model(model_name, ".model")
model_ckpt = os.path.join(bert_model_path, model_dir, "bert_model.ckpt")

# Tokenize
do_lower_case = not (model_name.find("cased") == 0 or model_name.find("multi_cased") == 0)
bert.bert_tokenization.validate_case_matches_checkpoint(do_lower_case, model_ckpt)
vocab_file = os.path.join(model_dir, "vocab.txt")
tokenizer = bert.bert_tokenization.FullTokenizer(vocab_file, do_lower_case)

# bert ckpt path
bert_ckpt_dir = os.path.join(bert_model_path, "multi_cased_L-12_H-768_A-12")
bert_ckpt_file = os.path.join(bert_ckpt_dir, "bert_model.ckpt")
bert_config_file = os.path.join(bert_ckpt_dir, "bert_config.json")

with open(os.path.join(data_path, "sentiments.pkl"), "rb") as f:
    data = pickle.load(f)
df = pd.read_csv(os.path.join(data_path, 'sentiments.csv'))

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
            res = max_learn_rate*math.exp(math.log(end_learn_rate/max_learn_rate)*(epoch-warmup_epoch_count+1)/(total_epoch_count-warmup_epoch_count+1))
        return float(res)
    learning_rate_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=1)

    return learning_rate_scheduler

class MyCustomCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        self.pred_sentences = [df.sentence[i] for i in range(10, 20)]
        self.pred_sentiments = [df.sentiments[i] for i in range(10, 20)]

    def on_epoch_end(self, epoch, logs=None):
    #def on_batch_end(self, epoch, logs=None):
        epoch = epoch + 1
        if epoch % config['save_model_period'] == 0:
            self.model.save_weights(os.path.join(epoch_model_path, 'sentiments.h5'), overwrite=True)
        pred_sentences = self.pred_sentences
        pred_tokens = map(tokenizer.tokenize, pred_sentences)
        pred_tokens = map(lambda tok: ["[CLS]"] + tok + ["[SEP]"], pred_tokens)
        pred_token_ids = list(map(tokenizer.convert_tokens_to_ids, pred_tokens))

        pred_token_ids = map(lambda tids: tids + [0] * (128 - len(tids)), pred_token_ids)
        pred_token_ids = np.array(list(pred_token_ids))

        res = self.model.predict(pred_token_ids)
        res = tf.sigmoid(res)
        res = tf.cast(res > 0.5, dtype=tf.int32).numpy()

        res_string = ''
        res_string += 'epoch: {}\n'.format(epoch)
        res_string += 'acc: {}, precision: {}, recall: {}, f1_score: {}, hamming_loss: {}\n'\
                        .format(logs['multi_label_accuracy'],
                                logs['multi_label_precision'],
                                logs['multi_label_recall'],
                                logs['multi_label_f1_score'],
                                logs['hamming_loss'])
        for text, label, sentiment in zip(pred_sentences, self.pred_sentiments, res):
            pred_sentiments = [data.code_to_senti[s-1] for s in sentiment * np.arange(1, 35) if s != 0]
            res_string += "text: {}\nlabels: {}\nres: {}\n\n".format(text, label, pred_sentiments)

        with open(os.path.join(epoch_log_path, 'epoch_res_{}.txt'.format(epoch)), 'w') as f:
            f.write(res_string)

log_dir = os.path.join(tb_path, datetime.datetime.now().strftime("%Y%m%d-%H%M%s"))
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir)

from focal_loss import BinaryFocalLoss

def create_model(max_seq_len, adapter_size=64):
    """Creates a classification model."""

    # adapter_size = 64  # see - arXiv:1902.00751

    # create the bert layer
    with tf.io.gfile.GFile(bert_config_file, "r") as reader:
        bc = StockBertConfig.from_json_string(reader.read())
        bert_params = map_stock_config_to_params(bc)
        bert_params.adapter_size = adapter_size
        bert = BertModelLayer.from_params(bert_params, name="bert")

    input_ids = keras.layers.Input(shape=(max_seq_len,), dtype='int32', name="input_ids")
    # token_type_ids = keras.layers.Input(shape=(max_seq_len,), dtype='int32', name="token_type_ids")
    # output         = bert([input_ids, token_type_ids])
    output = bert(input_ids)

    print("bert shape", output.shape)
    cls_out = keras.layers.Lambda(lambda seq: seq[:, 0, :])(output)
    cls_out = keras.layers.Dropout(0.5)(cls_out)
    logits = keras.layers.Dense(units=768, activation="tanh")(cls_out)
    logits = keras.layers.Dropout(0.5)(logits)
    logits = keras.layers.Dense(units=34)(logits)

    # model = keras.Model(inputs=[input_ids, token_type_ids], outputs=logits)
    # model.build(input_shape=[(None, max_seq_len), (None, max_seq_len)])
    model = keras.Model(inputs=input_ids, outputs=logits)
    model.build(input_shape=(None, max_seq_len))

    # load the pre-trained model weights
    load_stock_weights(bert, bert_ckpt_file)

    # freeze weights if adapter-BERT is used
    if adapter_size is not None:
        freeze_bert_layers(bert)



    def sigmoid_cross_entropy_loss(true, pred):
        loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=pred, labels=true)
        loss = tf.reduce_mean(tf.reduce_sum(loss))
        return loss
    focal_loss = BinaryFocalLoss(gamma=1, from_logits=True)

    loss_func_list = {
        "sigmoid_cross_entropy_loss": sigmoid_cross_entropy_loss,
        "focal_loss": focal_loss
    }

    model.compile(optimizer=keras.optimizers.Adam(),
                  loss=loss_func_list[config['loss_func']],
                  metrics=[MultiLabelAccuracy(batch_size=config['batch_size']),
                           MultiLabelPrecision(batch_size=config['batch_size']),
                           MultiLabelRecall(batch_size=config['batch_size']),
                           MultiLabelF1(batch_size=config['batch_size']),
                           HammingLoss(batch_size=config['batch_size'])])

    model.summary()

    return model

adapter_size = None # use None to fine-tune all of BERT
model = create_model(data.max_seq_len, adapter_size=adapter_size)
total_epoch_count = config['num_epochs']

model.fit(x=data.train_x, y=data.train_y,
          validation_split=0.1,
          batch_size=config['batch_size'],
          shuffle=True,
          epochs=total_epoch_count,
          initial_epoch=0,
          callbacks=[create_learning_rate_scheduler(max_learn_rate=1e-5,
                                                    end_learn_rate=1e-7,
                                                    warmup_epoch_count=config['warmup_epoch_count'],
                                                    total_epoch_count=total_epoch_count),
                     keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True),

                     tensorboard_callback, MyCustomCallback()])
model.save_weights(os.path.join(epoch_model_path, 'sentiments_fin.h5'), overwrite=True)

adapter_size = None # use None to fine-tune all of BERT
model = create_model(data.max_seq_len, adapter_size=adapter_size)

model.load_weights(os.path.join(epoch_model_path, 'sentiments_fin.h5'))
