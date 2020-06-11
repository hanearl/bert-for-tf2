import numpy as np
import tensorflow as tf


class SentimentsData:
    def __init__(self, df, tokenizer, max_seq_len):
        self.max_seq_len = max_seq_len

        if tokenizer:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = self.init_tokenizer()

        self.df = df[['sentence', 'sentiments']].dropna()
        self.get_train_data()
        self.get_label()

    def get_train_data(self):
        def processing(sequence):
            ops = [self._tokenize, self._add_special_token, self._pad_and_trim_sequence,
                   self._convert_tokens_to_ids]
            for op in ops:
                sequence = op(sequence)
            return sequence

        self.df.sentence = self.df.sentence.map(processing)
        self.train_x = np.array([np.array(x) for x in self.df.sentence.tolist()])


    def get_label(self):
        sentiment_list = set()

        def parse(sentiments):
            sentiments = str(sentiments)[1:-1].split(",")
            sentiments = [senti.replace('\'', '').strip() for senti in sentiments]
            sentiment_list.update(set(sentiments))
            return sentiments

        self.df.sentiments = self.df.sentiments.map(parse)
        self.senti_to_code = {senti: idx for idx, senti in enumerate(sentiment_list)}
        self.code_to_senti = {code: senti for senti, code in self.senti_to_code.items()}

        def mapping(sentiments):
            return [self.senti_to_code.get(x, 'unk') for x in sentiments]
        self.df.label = self.df.sentiments.map(mapping)

        num_class = 34
        self.train_y = []
        for y in self.df.label.tolist():
            res = [0] * num_class
            for num in y:
                res[num] = 1
            self.train_y.append(np.array(res))
        self.train_y = tf.cast(np.array(self.train_y), dtype=tf.float32)

    def _tokenize(self, string):
        return self.tokenizer.tokenize(string)

    def _pad_and_trim_sequence(self, tokens):
        max_seq_len = self.max_seq_len
        def pad():
            nonlocal tokens
            while len(tokens) != max_seq_len:
                tokens.append("[PAD]")

        if len(tokens) < max_seq_len:
            pad()
        elif len(tokens) > max_seq_len:
            tokens = tokens[:max_seq_len]

        return tokens

    def _add_special_token(self, tokens):
        return ["[CLS]"] + tokens + ["[SEP]"]

    def _convert_tokens_to_ids(self, tokens):
        return self.tokenizer.convert_tokens_to_ids(tokens)
