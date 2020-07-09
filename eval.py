import os
import pickle
import numpy as np
import tensorflow as tf

from create_model import create_model
from tokenizer import get_tokenizer
from config import Config

with open("/home/ubuntu/bert_sentiment/data/test_set_full.pkl", 'rb') as f:
    (X_test, y_test) = pickle.load(f)

config = Config()

from create_model import create_model
model = create_model(config, adapter_size=None)
for tb in ["bert_base"]:
    print(tb)
    model.load_weights(os.path.join("/home/ubuntu/bert_sentiment/epoch_models", tb, 'sentiments.h5'))
    result = model.evaluate(x=X_test, y=y_test, batch_size=150)
    print(result)