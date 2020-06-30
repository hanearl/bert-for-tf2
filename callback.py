import os
import pickle
import json

import pandas as pd
import tensorflow as tf
import numpy as np

from config import Config
from tokenizer import get_tokenizer
from alarm_bot import ExamAlarmBot


class MyCustomCallback(tf.keras.callbacks.Callback):
    def __init__(self, config):
        super(MyCustomCallback, self).__init__()
        self.config = config
        df = pd.read_csv(os.path.join(self.config.data_path, 'sentiments.csv'))
        self.bot = ExamAlarmBot()

        self.pred_sentences = [df.sentence[i] for i in range(10, 20)]
        self.pred_sentiments = [df.sentiments[i] for i in range(10, 20)]

        self.tokenizer = get_tokenizer(self.config)

    def on_epoch_end(self, epoch, logs=None):
        epoch = epoch + 1
        if epoch % 1 == 0:
            self.bot.send_msg('epoch {} done'.format(epoch))

        if epoch % self.config.save_model_period == 0:
            self.model.save_weights(os.path.join(self.config.epoch_model_path, 'sentiments_{}.h5'.format(epoch)), overwrite=True)
        pred_sentences = self.pred_sentences

        pred_tokens = map(self.tokenizer.tokenize, pred_sentences)
        pred_tokens = map(lambda tok: ["[CLS]"] + tok + ["[SEP]"], pred_tokens)
        pred_token_ids = list(map(self.tokenizer.convert_tokens_to_ids, pred_tokens))

        pred_token_ids = map(lambda tids: tids + [0] * (128 - len(tids)), pred_token_ids)
        pred_token_ids = np.array(list(pred_token_ids))

        res = self.model.predict(pred_token_ids)
        res = tf.sigmoid(res)
        res = tf.cast(res >= 0.5, dtype=tf.int32).numpy()

        res_string = ''
        res_string += 'epoch: {}\n'.format(epoch)
        res_string += 'acc: {}, precision: {}, recall: {}, f1_score: {}, hamming_loss: {}\n'\
                        .format(logs['val_multi_label_accuracy'],
                                logs['val_multi_label_precision'],
                                logs['val_multi_label_recall'],
                                logs['val_multi_label_f1_score'],
                                logs['val_hamming_loss'])
        for text, label, sentiment in zip(pred_sentences, self.pred_sentiments, res):
            pred_sentiments = [self.config.classes[str(s-1)] \
                               for s in sentiment * np.arange(1, len(self.config.classes) + 1) if s != 0]
            res_string += "text: {}\nlabels: {}\nres: {}\n\n".format(text, label, pred_sentiments)

        with open(os.path.join(self.config.epoch_log_path, 'epoch_res_{}.txt'.format(epoch)), 'w') as f:
            f.write(res_string)
