import os
import pickle
import numpy as np
import tensorflow as tf

from create_model import create_model
from tokenizer import get_tokenizer


class Eval:
    def __init__(self, config, model_name="sentiments.h5"):
        self.config = config

        self.model_name = model_name
        self.tokenizer = get_tokenizer(self.config)
        with open(os.path.join(self.config.data_path, 'test_set.pkl'), 'rb') as f:
            (self.test_x, self.test_y) = pickle.load(f)

        adapter_size = None # use None to fine-tune all of BERT
        self.model = create_model(self.config.max_seq_len, adapter_size=adapter_size)
        self.model.load_weights(os.path.join(self.config.epoch_model_path, model_name))

    def eval(self):
        result = self.model.evaluate(x=self.test_x[:int(self.config.train_len * 0.2)], y=self.test_y[:int(self.config.train_len * 0.2)], batch_size=self.config.batch_size)
        with open(os.path.join(self.config.epoch_log_path, 'eval.txt'), 'w') as f:
            f.write(str(result))


    def predict(self, test_set):
        sentences = self.tokenize(test_set)

        y_pred = self.model.predict(sentences)
        y_pred = tf.sigmoid(y_pred)
        y_pred = tf.cast(y_pred >= 0.5, dtype=tf.int32).numpy()

        res_string = ''
        for text, sentiment in zip(test_set, y_pred):
            pred_sentiments = [self.config.classes[str(s-1)] for s in sentiment * np.arange(1, 35) if s != 0]
            res_string += "sentence: {} \npred_sentiments: {}\n\n".format(text, pred_sentiments)
        print(res_string)

    def tokenize(self, sentences):
        pred_tokens = map(self.tokenizer.tokenize, sentences)
        pred_tokens = map(lambda tok: ["[CLS]"] + tok + ["[SEP]"], pred_tokens)
        pred_token_ids = list(map(self.tokenizer.convert_tokens_to_ids, pred_tokens))

        pred_token_ids = map(lambda tids: tids + [0] * (128 - len(tids)), pred_token_ids)
        return np.array(list(pred_token_ids))
