import os
import pandas as pd
import tensorflow as tf
import numpy as np
import json
from config import Config
import pickle
import bert


class MyCustomCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super(MyCustomCallback, self).__init__()
        with open('./config.json', 'r') as f:
            json_config = json.load(f)

        self.config = Config(json_config)

        with open(os.path.join(self.config.data_path, "sentiments.pkl"), "rb") as f:
            self.data = pickle.load(f)
        df = pd.read_csv(os.path.join(self.config.data_path, 'sentiments.csv'))

        self.pred_sentences = [df.sentence[i] for i in range(10, 20)]
        self.pred_sentiments = [df.sentiments[i] for i in range(10, 20)]

        # Tokenize
        do_lower_case = not (self.config.model_name.find("cased") == 0 or self.config.model_name.find("multi_cased") == 0)
        bert.bert_tokenization.validate_case_matches_checkpoint(do_lower_case, self.config.model_ckpt)
        vocab_file = os.path.join(self.config.model_dir, "vocab.txt")
        self.tokenizer = bert.bert_tokenization.FullTokenizer(vocab_file, do_lower_case)

    def on_epoch_end(self, epoch, logs=None):
        epoch = epoch + 1
        if epoch % self.config.save_model_period == 0:
            self.model.save_weights(os.path.join(self.config.epoch_model_path, 'sentiments.h5'), overwrite=True)
        pred_sentences = self.pred_sentences
        pred_tokens = map(self.tokenizer.tokenize, pred_sentences)
        pred_tokens = map(lambda tok: ["[CLS]"] + tok + ["[SEP]"], pred_tokens)
        pred_token_ids = list(map(self.tokenizer.convert_tokens_to_ids, pred_tokens))

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
            pred_sentiments = [self.data.code_to_senti[s-1] for s in sentiment * np.arange(1, 35) if s != 0]
            res_string += "text: {}\nlabels: {}\nres: {}\n\n".format(text, label, pred_sentiments)

        with open(os.path.join(self.config.epoch_log_path, 'epoch_res_{}.txt'.format(epoch)), 'w') as f:
            f.write(res_string)
