import os
from components import DataSplitter
from components import JsonDataReader

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
WORKING_DIR_NAME = "train_dir"
WORKING_PATH = os.path.join(CURRENT_PATH, WORKING_DIR_NAME)
DATA_PATH = os.path.join(WORKING_PATH, "data")


def post_process(data_item):
    return data_item+"\n"


Pipeline = JsonDataReader.partial(map_func = lambda x: x.strip())   |          \
           DataSplitter.partial(output_dir = DATA_PATH,
                                post_processor=post_process,
                                extension = ".txt")

path = ["./data/part-202101281a.json"]
ret = Pipeline(
    file_path = path,
    json_eval_path = "[:].content",
    split_config = [("train", 0.05), ("valid", 0.05), ("test", 0.9)]
)