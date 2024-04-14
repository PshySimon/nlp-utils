import os
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

parser = argparse.ArgumentParser(description='Argument Parser for Training Script')

# 添加参数
parser.add_argument('--resume', type=lambda x: x.lower() in ('true', '1', 't', 'y', 'yes'), default=True, help='Resume training flag')
parser.add_argument('--model_name', type=str, default='model', help='Model name')
parser.add_argument('--device', type=str, default='cpu', help='Device (cpu or cuda)')
parser.add_argument('--saving_steps', type=int, default=1000, help='Saving steps during training')
parser.add_argument('--logging_steps', type=int, default=50, help='Logging steps during training')
parser.add_argument('--train_dataset_path', type=str, default=None, help='Path to training dataset')
parser.add_argument('--valid_dataset_path', type=str, default=None, help='Path to validation dataset')
parser.add_argument('--batch_size_per_gpu', type=int, default=8, help='Batch size per GPU')
parser.add_argument('--learning_rate', type=float, default=5e-4, help='Learning rate')
parser.add_argument('--gradient_accumulation_steps', type=int, default=4, help='Gradient accumulation steps')
parser.add_argument('--num_train_epoches', type=int, default=1, help='Number of training epochs')
parser.add_argument('--mixed_precision', type=str, default="no", help="Mixed precision for GPU")
parser.add_argument('--keep_latest_n_checkpoints', type=int, default=3, help="When saving models, keep latest checkpoints for reducing waste of disk")
parser.add_argument('--weight_decay', type=float, default=0.0, help="Weight decay of optimizer")
parser.add_argument('--seed', type=int, default=42, help="seed of random data")
parser.add_argument('--adam_epsilon', type=float, default=1e-8, help="adam epsilon")
parser.add_argument('--warmup_steps', type=int, default=0, help="warmup steps")
parser.add_argument('--max_grad_norm', type=float, default=1.0, help="max grad norm")
parser.add_argument('--div_factor', type=int, default=50, help="div factor")

# 解析参数
args = parser.parse_args()

# -----------------------------定义配置-------------------------
# 通用配置
@config_class(WORKING_PATH)
class CommonConfig:
    def __init__(self) -> None:
        self.tokenizer_dir = os.path.join(WORKING_PATH, "tokenizer")
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_dir)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.max_length = 1024
        self.vocab_size = self.tokenizer.vocab_size


# 训练配置
@config_class(WORKING_PATH)
class TrainConfig(CommonConfig):
    def __init__(self) -> None:
        super().__init__()
        self.saving_steps = args.saving_steps
        self.keep_latest_n_checkpoints = args.keep_latest_n_checkpoints
        self.batch_size_per_gpu = args.batch_size_per_gpu
        self.device = args.device
        self.gradient_accumulation_steps = args.gradient_accumulation_steps
        self.num_train_epoches = args.num_train_epoches
        self.weight_decay = args.weight_decay
        self.learning_rate = args.learning_rate
        self.adam_epsilon = args.adam_epsilon
        self.warmup_steps = args.warmup_steps
        self.max_grad_norm = args.max_grad_norm
        self.logging_steps = args.logging_steps
        self.div_factor = args.div_factor
        self.output_dir = WORKING_PATH
        self.seed = args.seed
        self.mixed_precision = args.mixed_precision
        self.model_file = ""
        self.model_save_path = os.path.join(WORKING_PATH, "model_save")
        self.train_state_dir = os.path.join(WORKING_PATH, "training_state")

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
        self.device = args.device
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

# 数据配置
@config_class(WORKING_PATH)
class DataConfig(CommonConfig):
    def __init__(self) -> None:
        super().__init__()
        # 按照block_size截断句子，如果短于min_sentence_length就
        self.min_sentence_length = 5
        self.block_size = 1024
        self.max_seq_len = 4

# -----------------------------初始化配置-----------------------
train_config = TrainConfig()
model_config = LlamaConfig()
data_config = DataConfig()
# -----------------------------数据处理-------------------------
train_dataset = Dataset.load(args.train_dataset_path)
valid_dataset = Dataset.load(args.valid_dataset_path)
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
        logits, loss = model(**batch_data)
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
        return
        # input_ids = batch_data["input_ids"]
        # labels = batch_data["labels"]
        # batch_size, cur_len = input_ids.size()
        # preds = self.trainer.model.generate(input_ids, cur_len, batch_size)
        # preds = self.trainer.accelerator.gather_for_metrics(preds)\
        #                                 .detach().cpu().numpy()
        # labels = self.trainer.accelerator.gather_for_metrics(labels)\
        #                                 .detach().cpu().numpy()
        # preds = self.trainer.tokenizer.batch_decode(preds)
        # labels = self.trainer.tokenizer.batch_decode(labels)

        # bleu_scores = [get_bleu4_score(reference, output) for reference, output in zip(labels, preds)]
        # return bleu_scores
    
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
    model_name=args.model_name,
    train_config=train_config,
    train_dataset_path = args.train_dataset_path,
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
trainer.train(resume=args.resume)