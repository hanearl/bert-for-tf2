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
            ops = [self._tokenize, self._pad_and_trim_sequence, self._add_special_token,
                   self._convert_tokens_to_ids]
            for op in ops:
                sequence = op(sequence)
            return sequence

        self.df.sentence = self.df.sentence.map(processing)

    def get_label(self):
        sentiment_list = set()

        def parse(sentiments):
            """
            side-effect : update sentiment_list
            """
            sentiments = str(sentiments)[1:-1].split(",")
            sentiments = [senti.replace('\'', '').strip() for senti in sentiments]
            sentiment_list.update(set(sentiments))
            return sentiments

        self.df.sentiments = self.df.sentiments.map(parse)
        senti_to_code = {senti: idx for idx, senti in enumerate(sentiment_list)}

        def mapping(sentiments):
            return [senti_to_code.get(x, 'unk') for x in sentiments]
        self.df.label = self.df.sentiments.map(mapping)

    def _tokenize(self, string):
        return self.tokenizer.tokenize(string)

    def _pad_and_trim_sequence(self, tokens):
        max_seq_len = self.max_seq_len - 2
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
