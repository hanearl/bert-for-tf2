import os
import sys
import datetime
import pickle

sys.path.append('bert-for-tf2')
from tensorflow import keras

from create_model import create_model
from create_model import create_learning_rate_scheduler
from callback import MyCustomCallback
from alarm_bot import ExamAlarmBot
from eval import Eval

class ExamHelper:
    def __init__(self, config):
        self.config = config
        self.create_dir()
        with open(os.path.join(self.config.data_path, "train_set.pkl"), "rb") as f:
            (self.train_x, self.train_y) = pickle.load(f)
        self.bot = ExamAlarmBot()

    def create_dir(self):
        if not os.path.isdir(self.config.epoch_log_path):
            os.mkdir(self.config.epoch_log_path)

        if not os.path.isdir(self.config.epoch_model_path):
            os.mkdir(self.config.epoch_model_path)

        if not os.path.isdir(self.config.tb_path):
            os.mkdir(self.config.tb_path)

    def run_exam(self):
        log_dir = os.path.join(self.config.tb_path, datetime.datetime.now().strftime("%Y%m%d-%H%M%s"))
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir)

        adapter_size = None # use None to fine-tune all of BERT
        model = create_model(self.config.max_seq_len, adapter_size=adapter_size)

        model.fit(x=self.train_x[:self.config.train_len], y=self.train_y[:self.config.train_len],
                  validation_split=0.2,
                  batch_size=self.config.batch_size,
                  shuffle=True,
                  epochs=self.config.num_epochs,
                  initial_epoch=self.config.initial_epoch,
                  callbacks=[create_learning_rate_scheduler(max_learn_rate=1e-5,
                                                            end_learn_rate=1e-7,
                                                            warmup_epoch_count=self.config.warmup_epoch_count,
                                                            total_epoch_count=self.config.num_epochs),
                             tensorboard_callback, MyCustomCallback()])
        model.save_weights(os.path.join(self.config.epoch_model_path, 'sentiments.h5'), overwrite=True)
        self.eval = Eval(self.config)
        self.eval.eval()
        self.bot.send_msg('{} train is done'.format(self.config.train_name))
