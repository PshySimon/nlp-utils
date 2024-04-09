import os
import random
from tqdm import tqdm

from ..config.config import Configurable
from ..datasets.data_reader import Reader
from ..component import ComponentNode


"""
分割数据集,注意splitter依赖reader,所以前置node必须是Reader
splitter只能是终结点

head_node = PipelineNode()
reader = JsonDataReader(path,"[:]")
spliter = DataSplitter(
    split_config=[("train", 0.9), ("valid", 0.1)],
    extension=".txt"
)

pipeline = head_node | reader | spliter
pipeline.execute()
"""
class DataSplitter(ComponentNode):
    def __init__(self, **kwargs) -> None:
        args = Configurable(kwargs)
        self.split_config = args.split_config
        self.output_dir = args.output_dir
        self.extension = args.extension
        self.post_processor = args.post_processor

    def __call__(self, component: Reader):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        split_file_names = []
        total_num = len(component.gather())

        if not self.split_config:
            raise ValueError("Split rules cannot be empty!")

        num_splits_array = [0] * len(self.split_config)

        for i in range(len(self.split_config)):
            prob = self.split_config[i][1]
            split_num = int(total_num * prob)
            if i == 0:
                num_splits_array[i] = split_num
            else:
                num_splits_array[i] = split_num + num_splits_array[i-1]

        sample_indices = [idx for idx in range(total_num)]
        random.shuffle(sample_indices)

        extension = self.extension if self.extension is not None         \
                                   else component.get_extension() 

        for i in range(len(self.split_config)):
            start_idx = 0 if i == 0 else num_splits_array[i - 1]
            end_idx = num_splits_array[i]
            file_name = "{}{}".format(self.split_config[i][0], extension)
            split_file_names.append(file_name)
            file_path = os.path.join(self.output_dir, file_name)
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
            pbar = tqdm(total=end_idx - start_idx,
                        desc="splitting {} {} / {}".format(
                            self.split_config[i][0],
                            i+1, len(self.split_config)))
            for j in range(start_idx, end_idx):
                pbar.update(1)
                with open(file_path, "a", encoding="utf-8") as file_handler:
                    if self.post_processor is not None:
                        post_processed_data = self.post_processor(component.data[sample_indices[j]])
                        file_handler.write(post_processed_data)
                    else:
                        file_handler.write(str(component.data[sample_indices[j]]) + "\n")
            pbar.close()
        print("Split files success {}".format(split_file_names))
        return split_file_names
