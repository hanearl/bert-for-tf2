import os
import sys
import datetime
import pickle

sys.path.append('bert-for-tf2')
from tensorflow import keras
from skmultilearn.model_selection import iterative_train_test_split

from create_model import create_model
from create_model import create_learning_rate_scheduler
from callback import MyCustomCallback
from alarm_bot import ExamAlarmBot
from eval import Eval

class ExamHelper:
    def __init__(self, config):
        self.config = config
        self.create_dir()
        with open(os.path.join(self.config.data_path, self.config.train_set), "rb") as f:
            (self.train_x, self.train_y) = pickle.load(f)

        with open(os.path.join(self.config.data_path, self.config.val_set), "rb") as f:
            (self.val_x, self.val_y) = pickle.load(f)
        self.bot = ExamAlarmBot()

    def create_dir(self):
        if not os.path.isdir(self.config.epoch_log_path):
            os.mkdir(self.config.epoch_log_path)

        if not os.path.isdir(self.config.epoch_model_path):
            os.mkdir(self.config.epoch_model_path)

        if not os.path.isdir(self.config.tb_path):
            os.mkdir(self.config.tb_path)

    def run_exam(self):
        with open(os.path.join(self.config.epoch_log_path, 'config.txt'), 'w') as f:
            f.write(str(self.config))

        log_dir = os.path.join(self.config.tb_path, datetime.datetime.now().strftime("%Y%m%d-%H%M%s"))
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir)

        adapter_size = None # use None to fine-tune all of BERT
        model = create_model(self.config, adapter_size=adapter_size)

        hist = model.fit(x=self.train_x, y=self.train_y,
                  validation_data=(self.val_x, self.val_y),
                  batch_size=self.config.batch_size,
                  shuffle=True,
                  epochs=self.config.num_epochs,
                  initial_epoch=self.config.initial_epoch,
                  callbacks=[create_learning_rate_scheduler(max_learn_rate=1e-5,
                                                            end_learn_rate=1e-7,
                                                            warmup_epoch_count=self.config.warmup_epoch_count,
                                                            total_epoch_count=self.config.num_epochs),
                             tensorboard_callback, MyCustomCallback(self.config)])

        model.save_weights(os.path.join(self.config.epoch_model_path, 'sentiments_fin.h5'), overwrite=True)
        eval = Eval(self.config)
        eval.eval()
        self.bot.send_msg('{} train is done'.format(self.config.train_name))

        return hist
