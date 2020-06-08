import os
import bert


class Config:
    def __init__(self, config):
        self.max_seq_len = config['max_seq_len']
        self.focal_gamma = config['focal_gamma']
        self.save_model_period = config['save_model_period']
        self.loss_func = config['loss_func']
        self.batch_size = config['batch_size']
        self.warmup_epoch_count = config['warmup_epoch_count']
        self.initial_epoch = config['initial_epoch']
        self.num_epochs = config['num_epochs']
        self.drive_path = "/home/hanearl/Desktop"
        self.train_name = config["train_name"]

        self.project_path = os.path.join(self.drive_path, "bert_sentiment")
        self.bert_model_path = os.path.join(self.project_path, "bert_model")
        self.data_path = os.path.join(self.project_path, "data")
        self.epoch_log_path = os.path.join(self.project_path, "epoch_logs", self.train_name)
        self.epoch_model_path = os.path.join(self.project_path, "epoch_models", self.train_name)
        self.tb_path = os.path.join(self.project_path, "logs", self.train_name)

        self.model_name = config['model_name']
        self.model_dir = bert.fetch_google_bert_model(self.model_name, ".model")
        self.model_ckpt = os.path.join(self.bert_model_path, self.model_dir, "bert_model.ckpt")

        # bert ckpt path
        self.bert_ckpt_dir = os.path.join(self.bert_model_path, "multi_cased_L-12_H-768_A-12")
        self.bert_ckpt_file = os.path.join(self.bert_ckpt_dir, "bert_model.ckpt")
        self.bert_config_file = os.path.join(self.bert_ckpt_dir, "bert_config.json")


