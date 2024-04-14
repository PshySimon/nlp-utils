import pickle
from tqdm import tqdm
import numpy as np
from time import time
from concurrent.futures import ThreadPoolExecutor
from torch.utils.data import Dataset as TorchDataset

from .data_reader import get_reader, Reader


class Dataset(TorchDataset):
    def __init__(self, data, config):
        self.config = config
        self.tokenizer = config.tokenizer
        self.data = data

    @staticmethod
    def check_subclass(var, type_):
        if not isinstance(var, type):
            raise ValueError("Var should be a `class` of `{}`".format(type_.__name__))
        if var is type_:
            raise ValueError("Var should not be `{}` type".format(type_.__name__))
        if not issubclass(var, type_):
            raise ValueError("Var class should be subclass of `{}`".format(type_.__name__))

    @staticmethod
    def from_config(config):
        dataset_class = config.dataset_class
        Dataset.check_subclass(dataset_class, Dataset)

        reader_config = config.reader_config if hasattr(config, "reader_config") else {}
        data_path = config.data_path
        
        if isinstance(data_path, str):
            return dataset_class.from_config(
                    config, data_path, reader_config, config.tokenizer)
        elif isinstance(data_path, list):
            return [
                dataset_class.from_config(
                        config, path, reader_config, config.tokenizer)
                for path in data_path
            ]
        else:
            raise NotImplementedError("data_path is either `str` or `list`")
        
    
    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    @staticmethod
    def load(path):
        dataset_instance = None
        with open(path, 'rb') as f:
            dataset_instance = pickle.load(f)
        return dataset_instance


class DatasetForCausalLMPretrain(Dataset):
    def __init__(self, data, config):
        super().__init__(data, config)
        self.max_seq_len = config.max_seq_len
        self.pad_token_id = config.tokenizer.pad_token_id

    @staticmethod
    def from_config(config, file_path, reader_config, tokenizer):
        reader_class = get_reader(file_path)
        reader_instance = reader_class(file_path=file_path, **reader_config)()
        data = {"input_ids": [], "attn_mask": []}
        num_workers = config.num_workers if hasattr(config, "num_workers") else 4
        pbar = tqdm(total=len(reader_instance.data), desc="tokenizing sentences")

        def process_item(line):
            tokenized_result = tokenizer(line, add_special_tokens=False, 
                                         return_attention_mask=False)
            input_ids = tokenized_result["input_ids"]
            input_ids.append(tokenizer.eos_token_id)
            pbar.update(1)
            data["input_ids"].extend(input_ids)
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # 将func应用于data中的每个元素，并返回结果
            executor.map(process_item, reader_instance.data)
        length = len(data["input_ids"])
        truncated_length = (length // config.block_size) * config.block_size
        data["input_ids"] = np.array(data["input_ids"],dtype=np.uint16)[:truncated_length].reshape(-1, config.block_size)
        return DatasetForCausalLMPretrain(data, config)

    def __getitem__(self, index):
        input_ids = self.data["input_ids"][index].astype(np.int32)
        return {
            "input_ids": input_ids,
            "labels": input_ids
        }
    
    def __len__(self):
        return len(self.data["input_ids"])
    