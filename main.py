import bert
import os

model_name = "multi_cased_L-12_H-768_A-12"
model_dir = bert.fetch_google_bert_model(model_name, ".model")
model_ckpt = os.path.join(model_dir, "bert_model.ckpt")
#
# bert_params = bert.params_from_pretrained_ckpt(model_dir)
# l_bert = bert.BertModelLayer.from_params(bert_params, name="bert")
#
# # use in Keras Model here, and call model.build()
#
# bert.load_bert_weights(l_bert, model_ckpt)      # should be called after model.build()

# Tokenize
do_lower_case = not (model_name.find("cased") == 0 or model_name.find("multi_cased") == 0)
bert.bert_tokenization.validate_case_matches_checkpoint(do_lower_case, model_ckpt)
vocab_file = os.path.join(model_dir, "vocab.txt")
tokenizer = bert.bert_tokenization.FullTokenizer(vocab_file, do_lower_case)
tokens = tokenizer.tokenize("hello")
token_ids = tokenizer.convert_tokens_to_ids(tokens)
print(tokens)
print(token_ids)
# train
# from tensorflow import keras
#
# max_seq_len = 128
# l_input_ids      = keras.layers.Input(shape=(max_seq_len,), dtype='int32')
# l_token_type_ids = keras.layers.Input(shape=(max_seq_len,), dtype='int32')
#
# # using the default token_type/segment id 0
# output = l_bert(l_input_ids)                              # output: [batch_size, max_seq_len, hidden_size]
# model = keras.Model(inputs=l_input_ids, outputs=output)
# model.build(input_shape=(None, max_seq_len))
#
# # provide a custom token_type/segment id as a layer input
# output = l_bert([l_input_ids, l_token_type_ids])          # [batch_size, max_seq_len, hidden_size]
# model = keras.Model(inputs=[l_input_ids, l_token_type_ids], outputs=output)
# model.build(input_shape=[(None, max_seq_len), (None, max_seq_len)])