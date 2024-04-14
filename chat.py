import os
import glob
import torch
import argparse
import numpy as np
from torch.utils.data import DataLoader
from rich.progress import (
    Progress, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn
)

from components import LlamaForCausalLM, config_class, Trainer, TrainerCallback, Dataset, DatasetForCausalLMPretrain
from utils.metrics import get_bleu4_score
from transformers import (
    Adafactor,
    AutoTokenizer
)

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
WORKING_DIR_NAME = "train_dir"
WORKING_PATH = os.path.join(CURRENT_PATH, WORKING_DIR_NAME)
DATA_PATH = os.path.join(WORKING_PATH, "data")


# 通用配置
@config_class(WORKING_PATH)
class CommonConfig:
    def __init__(self) -> None:
        self.tokenizer_dir = os.path.join(WORKING_PATH, "tokenizer")
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_dir)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.max_length = 100
        self.vocab_size = self.tokenizer.vocab_size

# 模型配置
@config_class(WORKING_PATH)
class LlamaConfig(CommonConfig):
    def __init__(self) -> None:
        super().__init__()
        # -----------model_parameters---------------------
        self.padding_idx = self.tokenizer.pad_token_id
        self.hidden_size = 512
        self.vocab_size = self.tokenizer.vocab_size
        self.num_hidden_layers = 16
        self.rms_norm_eps = 1e-05
        self.device = "cpu"
        # -----------embedding_parameters-----------------
        self.max_position_embeddings = 2048
        self.rope_theta = 10000
        self.attention_bias = False
        # -----------attention_parameters-----------------
        self.num_heads = 32
        self.attention_dropout = 0.0
        self.num_key_value_heads = 4
        # -----------mlp_parameters-----------------------
        self.intermidiate_size = 2048
        # -----------generation_parameters----------------
        self.eos_token_id = self.padding_idx
        self.pad_token_id = self.padding_idx
        self.eos_token_ids = [self.pad_token_id, self.eos_token_id]
        self.temperature = 0.98
        self.top_k = 50
        self.top_p = 0.80
        self.repetition_penalty = 1.1
        self.length_penalty = -2.0
        self.num_beams = 5
        self.do_sample = True

def find_latest_folder(directory):
    folders = [name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name))]
    if not folders:
        return None

    # 按照文件夹名进行排序，找到最新的文件夹
    sorted_folders = sorted(folders, reverse=True)
    latest_folder = sorted_folders[0]
    return os.path.abspath(os.path.join(directory, latest_folder))


def find_most_recent_checkpoint(directory):
    pattern = os.path.join(directory, '*.ckpt')
    checkpoint_files = glob.glob(pattern)

    if not checkpoint_files:
        return None

    # 提取文件名中的epoch和steps信息，并按照优先级排序
    def get_epoch_steps(filename):
        parts = os.path.basename(filename).split('_')
        return int(parts[2]), int(parts[3])

    sorted_checkpoints = sorted(checkpoint_files, key=lambda x: get_epoch_steps(x), reverse=True)
    return sorted_checkpoints[0]


model_config = LlamaConfig()

latest_directory = find_latest_folder("/root/autodl-tmp/nlp-utils/train_dir/model_save")
latest_checkpoint_path = find_most_recent_checkpoint(latest_directory)

print("automaticlly detect latest checkpoint {}".format(latest_checkpoint_path))
st = torch.load(latest_checkpoint_path)
model = LlamaForCausalLM(model_config)
model.to("cpu")
model.load_state_dict(st)

while True:
    query = input("User:")
    if query == "exit":
        break
    out = model_config.tokenizer(query, return_tensors="pt")
    input_ids = out["input_ids"]
    cur_len = input_ids.size()[-1]
    out = model.generate(input_ids, cur_len, 1)
    output = model_config.tokenizer.decode(out.detach().cpu().numpy()[0])
    print("Assistant:{}".format(output))