import os
import sys
import torch
import signal
import random
import datetime
import numpy as np
from collections import deque
from accelerate import Accelerator

from utils.logger import Logger


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Trainer:
    def __init__(self,
                 model_name,
                 train_config,
                 tokenizer,
                 model,
                 dataset,
                 dataloader,
                 optimizer,
                 project_dir,
                 scheduler,
                 trainer_callback_class) -> None:
        self.model_name = model_name
        self.train_config = train_config
        self.tokenizer = tokenizer
        self.model = model
        self.tokenizer = train_config.tokenizer
        self.train_dataset, self.valid_dataset = dataset
        self.train_dataloader, self.valid_dataloader = dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.trainer_callback = trainer_callback_class(self)
        self.accelerator = None
        self.training_params = {
            "cur_epoch": 0
        }
        current_datetime = datetime.datetime.now()
        logger_filename = "chat-trainer-{}".format(current_datetime)
        self.formatted_datetime = current_datetime.strftime("%Y%m%d%H%M%S")
        self.logger = Logger('chat_trainer', std_out=True, project_dir=project_dir,
                             save2file=True, file_name=logger_filename)

        self.latest_checkpoints = deque()
        signal.signal(signal.SIGINT, self.process_exit_handler)
        set_seed(train_config.seed)

    def process_exit_handler(self, signal_received, frame):
        if not os.path.exists(self.train_config.train_state_dir):
            os.makedirs(self.train_config.train_state_dir)
        if self.accelerator and self.model:
            self.accelerator.wait_for_everyone()
            self.accelerator.save_state(
                output_dir=self.train_config.train_state_dir,
                safe_serialization=False)
            self.accelerator.print('model ckeck point has been saved in {}'            \
                                   .format(self.train_config.train_state_dir))
        else:
            self.logger.info("exit...currently not running...")
        sys.exit(0)
    
    def save_model(self, suffix):
        if self.model and self.accelerator:
            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process:
                unwrap_model = self.accelerator.unwrap_model(self.model)
                model_dict = self.accelerator.get_state_dict(unwrap_model)
                model_save_path_with_date = os.path.join(
                    self.train_config.model_save_path,
                    self.formatted_datetime)
                model_save_path = os.path.join(model_save_path_with_date,
                                  "{}.ckpt".format(suffix))
                if not os.path.exists(model_save_path_with_date):
                    os.makedirs(model_save_path_with_date)
                self.delete_early_checkpoint()
                torch.save(model_dict, model_save_path)
                self.latest_checkpoints.append(model_save_path)

    def delete_early_checkpoint(self):
        # 如果是满的就腾一个位置出来
        while len(self.latest_checkpoints) >= self.train_config.keep_latest_n_checkpoints:
            file_name = self.latest_checkpoints.popleft()
            os.remove(file_name)

    def get_total_batch_size(self):
        available_gpu_nums = self.accelerator.state.num_processes
        total_batch_size = self.train_config.batch_size_per_gpu
        if available_gpu_nums >= 1:
            total_batch_size *= available_gpu_nums
        return total_batch_size
    
    def log(self, msg, save_to_file=True):
        if not self.accelerator or (self.accelerator and self.accelerator.is_main_process):
            return self.logger.info(msg, std_out=False, save_to_file=save_to_file)
        
    def train(self, resume=False):
        # ------------------------------before_init_callback------------------------------
        self.trainer_callback.before_init()
        # --------------------------------------------------------------------------------

        # ------------------------------init----------------------------------------------
        self.accelerator = Accelerator(
            mixed_precision=self.train_config.mixed_precision,
            gradient_accumulation_steps=self.train_config.gradient_accumulation_steps,
            project_dir=self.train_config.train_state_dir
        )
        self.accelerator.register_for_checkpointing(self.scheduler)
        device = self.accelerator.device
        self.log("using device {}, device num is {}".format(str(device), self.accelerator.num_processes), save_to_file=True)
        train_steps_per_epoch = int(np.ceil(len(self.train_dataset) // self.get_total_batch_size()))

        self.log('train dataset size: {}, steps per epoch:{}'                                       \
            .format(len(self.train_dataset), train_steps_per_epoch), save_to_file=True)
        self.model, self.optimizer, self.scheduler, self.train_dataloader, self.valid_dataloader =   \
            self.accelerator.prepare(
                self.model, 
                self.optimizer,
                self.scheduler, 
                self.train_dataloader, 
                self.valid_dataloader,
            )
        self.log("All required things has been done!")
        if resume:
            self.log("Resume model state dict from checkpoint ...")
        if resume:
            self.accelerator.load_state(input_dir=self.train_config.train_state_dir)
        # --------------------------------------------------------------------------------
            

        # ------------------------------after_init----------------------------------------
        self.trainer_callback.after_init()
        # --------------------------------------------------------------------------------


        best_criterion_score = 0.0
        best_epoch = 0
        epoch_loss_list = []

        # ------------------------------before_train--------------------------------------
        self.trainer_callback.before_train()
        # --------------------------------------------------------------------------------

        # ------------------------------train---------------------------------------------
        for epoch in range(self.training_params["cur_epoch"], self.train_config.num_train_epoches):
            # ------------------------------epoch_begin-----------------------------------
            self.trainer_callback.on_epoch_begin(epoch, epoch_loss_list, best_epoch, best_criterion_score)
            # ----------------------------------------------------------------------------
            epoch_loss_list = []
            self.model.train()
            # ------------------------------epoch_train_begin-----------------------------
            self.trainer_callback.on_epoch_train_begin()
            # ----------------------------------------------------------------------------
            for step, batch_data in enumerate(self.train_dataloader):

                # ------------------------------train_batch-------------------------------
                output_loss = self.trainer_callback.train_batch(step, batch_data)
                # ------------------------------------------------------------------------

                loss = output_loss.mean() / self.train_config.gradient_accumulation_steps
                self.accelerator.backward(loss)

                if (step + 1) % self.train_config.gradient_accumulation_steps == 0:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                if (step + 1) % self.train_config.saving_steps == 0 or step == train_steps_per_epoch:
                    # ------------------------------save_model-----------------------------------
                    self.trainer_callback.on_save_model()
                    # ---------------------------------------------------------------------------
                    self.save_model("{}_epoch_{}_{}_lastest".format(self.model_name, epoch, step + 1))
                
                if step % self.train_config.logging_steps == 0 or step == train_steps_per_epoch:
                    loss_cpu = loss.detach().item() * self.train_config.gradient_accumulation_steps
                    epoch_loss_list.append(loss_cpu)
                    info_txt = 'training loss: epoch:{}, step:{}, loss:{}, device:{}'.\
                        format(epoch, step, loss_cpu, str(self.accelerator.device))
                    self.log(info_txt, save_to_file=True)
                    # ------------------------------logging_steps--------------------------------
                    self.trainer_callback.on_logging_steps(step, train_steps_per_epoch, loss_cpu)
                    # ---------------------------------------------------------------------------

            
            # ------------------------------epoch_train_end-------------------------------
            self.trainer_callback.on_epoch_train_end()
            # ----------------------------------------------------------------------------

            # ------------------------------epoch_evaluate_begin--------------------------
            self.trainer_callback.on_epoch_evaluate_begin()
            # ----------------------------------------------------------------------------

            cur_criterion_score = self.evaluate()

            # ------------------------------epoch_evaluate_end----------------------------
            self.trainer_callback.on_epoch_evaluate_end()
            # ----------------------------------------------------------------------------

            if cur_criterion_score > best_criterion_score:
                best_criterion_score = cur_criterion_score
                best_epoch = epoch
                self.save_model("best")
                self.accelerator.save_state(output_dir=self.train_config.train_state_dir)
            # ------------------------------epoch_end-------------------------------------
            self.trainer_callback.on_epoch_end(epoch, epoch_loss_list, cur_criterion_score, best_criterion_score, best_epoch)
            # ----------------------------------------------------------------------------
            
    def evaluate(self):
        self.model.eval()
        eval_steps_per_epoch = int(np.ceil(len(self.valid_dataset) // self.get_total_batch_size()))

        criterion_scores = []
        if self.accelerator.is_main_process:
            pass

        with torch.no_grad():
            for step, batch_data in enumerate(self.valid_dataloader):
                batch_criterion_scores = self.trainer_callback.evaluate(step, batch_data)
                if batch_criterion_scores is not None:
                    criterion_scores.extend(batch_criterion_scores)
        return self.trainer_callback.calculate_score(criterion_scores)
    
    def test(self):
        pass


class TrainerCallback:
    def __init__(self, trainer):
        self.trainer = trainer
        self.params = {}

    def before_init(self):
        return
    
    def after_init(self):
        return
    
    def before_train(self):
        return
    
    def on_epoch_begin(self):
        return
    
    def on_epoch_train_begin(self):
        return
    
    def train_batch(self, step, batch_data):
        raise NotImplementedError("You must implement `train_batch` by yourslef!")

    def on_save_model(self):
        return
    
    def on_logging_steps(self):
        return
    
    def on_epoch_train_end(self):
        return
    
    def on_epoch_evaluate_begin(self):
        return

    def evaluate(self, step, batch_data):
        raise
    
    def calculate_score(self, criterion_scores):
        raise -1

    def on_epoch_evaluate_end(self):
        return
    
    def on_epoch_end(self):
        return


