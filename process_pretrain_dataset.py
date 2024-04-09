import os

from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from components.config.config import config_class
from components import DatasetForCausalLMPretrain, Dataset

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
WORKING_DIR_NAME = "train_dir"
WORKING_PATH = os.path.join(CURRENT_PATH, WORKING_DIR_NAME)
TOKENIZER_PATH = os.path.join(WORKING_PATH, "tokenizer")
DATA_PATH = os.path.join(WORKING_PATH, "data")

# -----------------------------定义配置-------------------------
# 通用配置
@config_class(WORKING_PATH)
class CommonConfig:
    def __init__(self) -> None:
        self.tokenizer_dir = os.path.join(WORKING_PATH, "tokenizer")
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_dir)
        self.tokenizer.pad_token = self.tokenizer.eos_token

# 数据配置
@config_class(WORKING_PATH)
class DataConfig(CommonConfig):
    def __init__(self) -> None:
        super().__init__()
        self.block_size = 1024
        self.data_path = [
            os.path.join(DATA_PATH, "train.txt"),
            os.path.join(DATA_PATH, "valid.txt")
        ]
        self.dataset_class = DatasetForCausalLMPretrain
        self.max_seq_len = 4
        # tokenizer并行化处理
        self.num_workers = 4


config = DataConfig()
train_ds, valid_ds = Dataset.from_config(config)
train_ds.save(os.path.join(DATA_PATH, "train_data.bin"))
valid_ds.save(os.path.join(DATA_PATH, "valid_data.bin"))
