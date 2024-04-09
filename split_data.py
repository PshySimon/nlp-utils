import os
from components import DataSplitter
from components import JsonDataReader

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
WORKING_DIR_NAME = "train_dir"
WORKING_PATH = os.path.join(CURRENT_PATH, WORKING_DIR_NAME)
DATA_PATH = os.path.join(WORKING_PATH, "data")


def post_process(data_item):
    return data_item+"\n"


def get_files(directory, extension):
    json_files = []
    for filename in os.listdir(directory):
        if filename.endswith(extension) and os.path.isfile(os.path.join(directory, filename)):
            json_files.append(os.path.abspath(os.path.join(directory, filename)))
    return json_files


Pipeline = JsonDataReader.partial(map_func = lambda x: x.strip())   |          \
           DataSplitter.partial(output_dir = DATA_PATH,
                                post_processor=post_process,
                                extension = ".txt")

files = get_files("/root/autodl-fs/WuDaoCorpus2.0_base_200G", ".json")
ret = Pipeline(
    file_path = files,
    json_eval_path = "[:].content",
    split_config = [("train", 0.98), ("valid", 0.02)]
)