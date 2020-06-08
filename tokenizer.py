import bert
import os


# Tokenize
def get_tokenizer(config):
    do_lower_case = not (config.model_name.find("cased") == 0 or config.model_name.find("multi_cased") == 0)
    bert.bert_tokenization.validate_case_matches_checkpoint(do_lower_case, config.model_ckpt)
    vocab_file = os.path.join(config.model_dir, "vocab.txt")
    tokenizer = bert.bert_tokenization.FullTokenizer(vocab_file, do_lower_case)
    return tokenizer