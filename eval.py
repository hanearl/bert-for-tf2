import os
import pickle
from create_model import create_model


class Eval:
    def __init__(self, config):
        self.config = config
        with open(os.path.join(config.data_path, config.test_set), 'rb') as f:
            (self.X_test, self.y_test) = pickle.load(f)
        self.model = create_model(config, adapter_size=None)

    def eval(self):
        self.model.load_weights(os.path.join(self.config.epoch_model_path, "sentiments_fin.h5"))
        result = self.model.evaluate(x=self.X_test, y=self.y_test, batch_size=self.config.batch_size)
        with open(os.path.join(self.config.epoch_log_path, 'eval.txt'), 'w') as f:
            f.write(str(result))