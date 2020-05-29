import unittest
import pandas as pd
import bert
import os
from sentiments_data import SentimentsData
from unittest import mock
from unittest.mock import patch


def init_bert_tokenizer():
    model_name = "multi_cased_L-12_H-768_A-12"
    model_dir = bert.fetch_google_bert_model(model_name, ".model")
    model_ckpt = os.path.join(model_dir, "bert_model.ckpt")
    do_lower_case = not (model_name.find("cased") == 0 or model_name.find("multi_cased") == 0)
    bert.bert_tokenization.validate_case_matches_checkpoint(do_lower_case, model_ckpt)
    vocab_file = os.path.join(model_dir, "vocab.txt")
    return bert.bert_tokenization.FullTokenizer(vocab_file, do_lower_case)


class PreprocessTestCase(unittest.TestCase):
    tokenizer = init_bert_tokenizer()

    def setUp(self) -> None:
        self.df = pd.DataFrame([{"sentiments": ["중립"], "sentence": "안녕하세요"},
                           {"sentiments": ["중립"], "sentence": "안녕하세요"},
                           {"sentiments": ["중립"], "sentence": "안녕하세요"},
                           {"sentiments": ["중립"], "sentence": "안녕하세요"},
                           {"sentiments": ["중립"], "sentence": "안녕하세요"}])

        self.tokenize_result = ['안', '##녕', '##하', '##세', '##요']
        self.tokenize_result_2 = ['[CLS]', '안', '##녕', '##하', '##세', '##요', '[SEP]']
        self.index_result = [101, 9521, 118741, 35506, 24982, 48549, 102]
        self.max_seq_len = 128
        self.sentiment_data = SentimentsData(self.df, self.tokenizer, self.max_seq_len)

    def test_tokenizing(self):
        target_string = "안녕하세요"

        expect = ['안', '##녕', '##하', '##세', '##요']
        result = self.sentiment_data._tokenize(target_string)

        self.assertEqual(expect, result)

    def test_append_special_tokens(self):
        cls = "[CLS]"
        sep = "[SEP]"

        expect = [cls] + self.tokenize_result + [sep]
        result = self.sentiment_data._add_special_token(self.tokenize_result)

        self.assertEqual(cls, result[0])
        self.assertEqual(sep, result[-1])
        self.assertEqual(expect, result)

    def test_pad_when_sequence_length_is_under_max_seq_length(self):
        max_seq_len = 10

        sent_data = SentimentsData(self.df, self.tokenizer, max_seq_len)
        token = self.tokenize_result
        sentence = sent_data._pad_and_trim_sequence(token)

        expect = max_seq_len - 2
        result = len(sentence)

        self.assertEqual(expect, result)

    def test_pad_when_seqence_length_is_upper_max_seq_length(self):
        max_seq_len = 3
        sent_data = SentimentsData(self.df, self.tokenizer, max_seq_len)
        token = self.tokenize_result
        sentence = sent_data._pad_and_trim_sequence(token)

        expect = max_seq_len - 2 # sep, cls token
        result = len(sentence)
        self.assertEqual(expect, result)

    def test_convert_string_token_to_id_token_using_by_vocab(self):
        tokens = self.tokenize_result_2
        expect = self.index_result
        result = self.sentiment_data._convert_tokens_to_ids(tokens)

        self.assertEqual(expect, result)

    def test_whole_data_inserted(self):
        expect = self.index_result[:-1]
        result = self.sentiment_data.df.sentence[0][:len(expect)]
        print(self.sentiment_data.train_x)
        self.assertEqual(expect, result)

    def test_sentiments_to_numberic_code(self):
        df = pd.read_csv('sentiments.csv')[:30]
        senti_data = SentimentsData(df, self.tokenizer, self.max_seq_len)
        label = senti_data.df.label
        print(label)

    # @unittest.skip
    def test_split_train_valid_test_dataset(self):
        from sentiments_data import SentimentsData
        import pandas as pd
        df = pd.read_csv('sentiments.csv')
        data = SentimentsData(df, self.tokenizer, 128)
        print(data.train_x.shape)
        print(data.train_y.shape)

if __name__ == '__main__':
    unittest.main()
