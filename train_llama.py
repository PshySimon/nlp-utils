import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from rich.progress import (
    Progress, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn
)
from transformers.generation.configuration_utils import GenerationConfig

from components import LlamaForCausalLM, config_class, Trainer, TrainerCallback, Dataset, DatasetForCausalLMPretrain
from utils.metrics import get_bleu4_score
from transformers import (
    Adafactor,
    AutoTokenizer
)
from torch.nn import functional as F
from components.datasets.datasets import DatasetForCausalLMPretrain
from components.datasets.data_reader import PureTextDataReader

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
WORKING_DIR_NAME = "train_dir"
WORKING_PATH = os.path.join(CURRENT_PATH, WORKING_DIR_NAME)
DATA_PATH = os.path.join(WORKING_PATH, "data")

# -----------------------------定义配置-------------------------
# 通用配置
@config_class(WORKING_PATH)
class CommonConfig:
    def __init__(self) -> None:
        self.tokenizer_dir = os.path.join(WORKING_PATH, "tokenizer")
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_dir)

# 训练配置
@config_class(WORKING_PATH)
class TrainConfig(CommonConfig):
    def __init__(self) -> None:
        super().__init__()
        self.saving_steps = 5000
        self.keep_latest_n_checkpoints = 3
        self.batch_size_per_gpu = 1
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.gradient_accumulation_steps = 2
        self.num_train_epoches = 1
        self.weight_decay = 0.0
        self.learning_rate = 5e-4
        self.adam_epsilon = 1e-8
        self.warmup_steps = 0
        self.max_grad_norm = 1.0
        self.logging_steps = 50
        self.div_factor = 50
        self.output_dir = WORKING_PATH
        self.seed = 42
        self.mixed_precision = "no"
        self.model_file = ""
        self.model_save_path = os.path.join(WORKING_PATH, "model_save")
        self.train_state_dir = os.path.join(WORKING_PATH, "training_state")

# 模型配置
@config_class(WORKING_PATH)
class LlamaConfig(CommonConfig):
    def __init__(self) -> None:
        super().__init__()
        # -----------model_parameters---------------------
        self.padding_idx = 0
        self.hidden_size = 512
        self.vocab_size = 32000
        self.num_hidden_layers = 4
        self.rms_norm_eps = 1e-05
        self.device = torch.device("cpu")
        # -----------embedding_parameters-----------------
        self.max_position_embeddings = 2048
        self.rope_theta = 10000
        self.attention_bias = False
        # -----------attention_parameters-----------------
        self.num_heads = 32
        self.attention_dropout = 0.0
        self.num_key_value_heads = 4
        # -----------mlp_parameters-----------------------
        self.intermidiate_size = 1024
        # -----------generation_parameters----------------
        self.eos_token_id = self.padding_idx
        self.pad_token_id = self.padding_idx
        self.max_generate_len = 2048
        self.generation_algorithm = "greedy_search"

# 数据配置
@config_class(WORKING_PATH)
class DataConfig(CommonConfig):
    def __init__(self) -> None:
        super().__init__()
        # 按照block_size截断句子，如果短于min_sentence_length就
        self.min_sentence_length = 5
        self.block_size = 2048
        self.max_seq_len = 2048

# -----------------------------初始化配置-----------------------
train_config = TrainConfig()
model_config = LlamaConfig()
data_config = DataConfig()
# -----------------------------数据处理-------------------------
train_dataset = Dataset.load(os.path.join(DATA_PATH, "train_data.bin"))
valid_dataset = Dataset.load(os.path.join(DATA_PATH, "valid_data.bin"))
# -----------------------------dataloader声明--------------------
train_dataloader = DataLoader(
    dataset=train_dataset,
    batch_size=train_config.batch_size_per_gpu
)
valid_dataloader = DataLoader(
    dataset=valid_dataset,
    batch_size=train_config.batch_size_per_gpu
)
# -----------------------------模型声明-------------------------------
model = LlamaForCausalLM(model_config)
# -----------------------------optimizer-----------------------------
optimizer = Adafactor(params=model.parameters(), lr=train_config.learning_rate, relative_step=False)
# -----------------------------scheduler-----------------------------
scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer, 
        max_lr=train_config.div_factor * train_config.learning_rate, 
        epochs=train_config.num_train_epoches, 
        steps_per_epoch=int(np.ceil(len(train_dataset) / (train_config.batch_size_per_gpu * train_config.gradient_accumulation_steps) )),  # 梯度累积相当于增大了batch_size
        div_factor=train_config.div_factor,
        cycle_momentum=False,
    )
        
# -----------------------------tokenizer-----------------------------
class CustomTrainerCallback(TrainerCallback):
    def before_init(self):
        return

    def after_init(self):
        if self.trainer.accelerator.is_main_process:
            self.params["progress"] = Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeRemainingColumn(),
                TimeElapsedColumn(),
                TextColumn("[bold blue]{task.fields[show_info]}"),
                refresh_per_second=1,  # 每1秒钟更新一次，不要频繁更新
            )
        steps_per_epoch = int(np.ceil(len(self.trainer.train_dataset) // self.trainer.get_total_batch_size()))
        eval_steps = int(np.ceil(len(self.trainer.valid_dataset) // self.trainer.get_total_batch_size()))
        self.params["epoch_progress"] = self.params["progress"].add_task(
            description='epoch: ', show_info='', total=self.trainer.train_config.num_train_epoches)
        self.params["steps_progress"] = self.params["progress"].add_task(
            description='steps: ', show_info='',                                      \
            total=np.ceil(steps_per_epoch / self.trainer.train_config.logging_steps))
        self.params["eval_progress"] = self.params["progress"].add_task(
            description='evaluate: ', show_info='', total=eval_steps, visible=False)
        self.params["progress"].start()
    
    def before_train(self):
        return
    
    def on_epoch_begin(self, epoch, epoch_loss_list, best_epoch, best_criterion_score):
        if self.trainer.accelerator.is_main_process:
            epoch_show_txt = 'epoch: {}/{}, avg_loss: {:.6f}, best_epoch: {}, best_bleu: {}'.format(
                epoch, self.trainer.train_config.num_train_epoches,
                np.average(epoch_loss_list) if epoch_loss_list else 0.,
                best_epoch,
                best_criterion_score
            )
            self.params["progress"].update(self.params["epoch_progress"], show_info=epoch_show_txt)
            self.params["progress"].reset(self.params["steps_progress"])
    
    def on_epoch_train_begin(self):
        return
    
    def train_batch(self, step, batch_data):
        loss, _ = model(**batch_data)
        return loss

    def on_save_model(self):
        return
    
    def on_logging_steps(self, step, steps_per_epoch, loss_cpu):
        if self.trainer.accelerator.is_main_process:
            step_show_txt = 'step: {}/{}, loss: {:.6f}'.format(
                step, steps_per_epoch, loss_cpu)
            self.params["progress"].advance(self.params["steps_progress"], advance=1)
            self.params["progress"].update(self.params["steps_progress"], show_info=step_show_txt)
    
    def on_epoch_train_end(self):
        return
    
    def on_epoch_evaluate_begin(self):
        return

    def evaluate(self, step, batch_data):
        input_ids = batch_data["input_ids"]
        labels = batch_data["labels"]
        preds = self.trainer.model.generate(input_ids)
        preds = self.trainer.accelerator.gather_for_metrics(preds)\
                                        .detach().cpu().numpy()
        labels = self.trainer.accelerator.gather_for_metrics(labels)\
                                        .detach().cpu().numpy()
        preds = self.trainer.tokenizer.batch_decode(preds)
        labels = self.trainer.tokenizer.batch_decode(labels)

        bleu_scores = [get_bleu4_score(reference, output) for reference, output in zip(labels, preds)]
        return bleu_scores
    
    def calculate_score(self, criterion_scores):
        if len(criterion_scores) == 0:
            return 0.
        return np.average(criterion_scores)
    
    def on_epoch_evaluate_end(self):
        return
    
    def on_epoch_end(self, epoch, epoch_loss_list, cur_criterion_score, best_criterion_score, best_epoch):
        # if self.trainer.accelerator.is_main_process:
        #     self.params["progress"].advance(self.params["epoch_progress"], advance=1)
        #     info_txt = 'epoch log: epoch:{}, avg_loss:{}, cur_bleu4:{}, best_bleu4:{}, best_epoch:{}'.\
        #                 format(
        #                     epoch,
        #                     np.average(epoch_loss_list) if epoch_loss_list else 0.,
        #                     cur_criterion_score,
        #                     best_criterion_score,
        #                     best_epoch)
        #     self.print_and_log(info_txt, self.trainer.accelerator)
        return

# -----------------------------声明训练器------------------------------
# 屏蔽accelerate细节
trainer = Trainer(
    train_config=train_config,
    model=model,
    tokenizer=train_config.tokenizer,
    dataset=(train_dataset, valid_dataset),
    dataloader=(train_dataloader, valid_dataloader),
    optimizer=optimizer,
    scheduler=scheduler,
    project_dir=WORKING_PATH,
    trainer_callback_class=CustomTrainerCallback
)
# -----------------------------开始训练-------------------------------
trainer.train(resume=False)