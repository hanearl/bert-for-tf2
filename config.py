import os
import json
import bert


class Config:
    def __init__(self, config_dir="./", config_name="config.json"):
        with open(os.path.join(config_dir, config_name), 'r') as f:
            config = json.load(f)
            for key, value in config.items():
                self[key] = value

        with open(os.path.join(config_dir, 'class.json'), 'r') as f:
            self.classes = json.load(f)

        self.project_path = os.path.join(self.drive_path, "bert_sentiment")
        self.bert_model_path = os.path.join(self.project_path, "bert_model")
        self.data_path = os.path.join(self.project_path, "data")
        self.epoch_log_path = os.path.join(self.project_path, "epoch_logs", self.train_name)
        self.epoch_model_path = os.path.join(self.project_path, "epoch_models", self.train_name)
        self.tb_path = os.path.join(self.project_path, "logs", self.train_name)

        self.model_dir = bert.fetch_google_bert_model(self.model_name, ".model")
        self.model_ckpt = os.path.join(self.bert_model_path, self.model_dir, "bert_model.ckpt")

        # bert ckpt path
        self.bert_ckpt_dir = os.path.join(self.bert_model_path, "multi_cased_L-12_H-768_A-12")
        self.bert_ckpt_file = os.path.join(self.bert_ckpt_dir, "bert_model.ckpt")
        self.bert_config_file = os.path.join(self.bert_ckpt_dir, "bert_config.json")

    def update_path(self):
        self.project_path = os.path.join(self.drive_path, "bert_sentiment")
        self.bert_model_path = os.path.join(self.project_path, "bert_model")
        self.data_path = os.path.join(self.project_path, "data")
        self.epoch_log_path = os.path.join(self.project_path, "epoch_logs", self.train_name)
        self.epoch_model_path = os.path.join(self.project_path, "epoch_models", self.train_name)
        self.tb_path = os.path.join(self.project_path, "logs", self.train_name)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def to_string(self):
        text = ""
        for k, v in self.__dict__.items():
            text += "{} : {} \n".format(k, v)

        return text
