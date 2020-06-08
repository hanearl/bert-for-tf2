import os
import datetime
import pickle
import json

from tensorflow import keras

from create_model import create_model
from create_model import create_learning_rate_scheduler
from callback import MyCustomCallback
from sentiments_data import SentimentsData
from config import Config

with open('./config.json', 'r') as f:
    json_config = json.load(f)

config = Config(json_config)

if not os.path.isdir(config.epoch_log_path):
    os.mkdir(config.epoch_log_path)

if not os.path.isdir(config.epoch_model_path):
    os.mkdir(config.epoch_model_path)

if not os.path.isdir(config.tb_path):
    os.mkdir(config.tb_path)

with open(os.path.join(config.data_path, "sentiments.pkl"), "rb") as f:
    data = pickle.load(f)

log_dir = os.path.join(config.tb_path, datetime.datetime.now().strftime("%Y%m%d-%H%M%s"))
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir)

adapter_size = None # use None to fine-tune all of BERT
model = create_model(data.max_seq_len, adapter_size=adapter_size)

model.fit(x=data.train_x, y=data.train_y,
          validation_split=0.1,
          batch_size=config.batch_size,
          shuffle=True,
          epochs=config.num_epochs,
          initial_epoch=config.initial_epoch,
          callbacks=[create_learning_rate_scheduler(max_learn_rate=1e-5,
                                                    end_learn_rate=1e-7,
                                                    warmup_epoch_count=config.warmup_epoch_count,
                                                    total_epoch_count=config.num_epochs),
                     keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True),

                     tensorboard_callback, MyCustomCallback()])
model.save_weights(os.path.join(config.epoch_model_path, 'sentiments_fin.h5'), overwrite=True)
