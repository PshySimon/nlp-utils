import os
import sys
import json
import pandas as pd
import ijson
from collections.abc import Iterable
from tqdm import tqdm
import re
import threading
import time
from functools import partial

from ..component import ComponentHead
from ..config.config import Configurable

MB = 1024 * 1024
ARRAY_SUFFIX_PATTERN = r".*\[.*\]$"
ARRAY_SEARCH_PATTERN = r"(\w*)\[([\d:])\]"

"""
fields='fields_a.fields_b[:].fields_c'
"""
def parse_fields(fields_path):
    if fields_path is None:
        raise RuntimeError("Cannot parse empty fields_path!")
    fields = fields_path.split(".")
    for i in range(len(fields)):
        if re.search(ARRAY_SUFFIX_PATTERN, fields[i]):
            break
    prefix = fields[:i]
    suffix = fields[i:]
    return ".".join(prefix), ".".join(suffix)

def extract_array_index(text):
    match = re.search(ARRAY_SEARCH_PATTERN, text)
    if match:
        return match.group(1), match.group(2)
    return None, None

def eval_json_path(obj, json_path: str):
    if not json_path:
        return obj
    fields = json_path.split(".")

    for i in range(len(fields)):
        if re.search(ARRAY_SUFFIX_PATTERN, fields[i]):
            field_name, array_index = extract_array_index(fields[i])
            if field_name is None or array_index is None:
                raise RuntimeError("Unknown json path {}".format(fields[i]))
            if field_name and field_name in obj:
                obj = obj[field_name]
            if not isinstance(obj, list):
                raise ValueError("Parsed an array field `{}`, got"          \
                                 "a nonarray obj".format(".".join(fields[:i+1])))
            if array_index.isdigit():
                obj = obj[int(array_index)]
            elif array_index == ":":
                return [eval_json_path(sub, ".".join(fields[i+1:])) for sub in obj]
            else:
                raise RuntimeError("Unexpected array index")
        else:
            if isinstance(obj, list):
                raise ValueError("List indices must be integers")
            if fields[i] and fields[i] in obj:
                obj = obj[fields[i]]
            else:
                raise RuntimeError("Unknown json path {}".format(fields[i]))
    return obj

def json_handler(fields_path):
    return partial(_json_handler, fields_path=fields_path)

def _json_handler(file_ptr, fields_path):
    data = []
    prefix, suffix = parse_fields(fields_path)
    for item in ijson.items(file_ptr, prefix):
        data.extend(eval_json_path(item, suffix))
    return data


class ReadFileWithProgressBar:
    def __init__(self,
                 handler_callback=None,
                 update_method_callback=None,
                 close_progress_bar_callback=None,
                 refresh_time=0.5) -> None:
        self.update_method = update_method_callback
        self.handler = handler_callback
        self.close = close_progress_bar_callback
        self.refresh_time = refresh_time
        self.thread = threading.Thread(target=self.__monitor)

        self.read_bytes = 0
        self.file_size = 0
        self.file_ptr = None
        self.is_finished = False

    def __refresh(self):
        if self.read_bytes == self.file_size:
            self.is_finished = True
        if self.file_ptr is None:
            return
        cur_read_bytes = self.file_ptr.tell()
        read_bytes_delta = cur_read_bytes - self.read_bytes
        if read_bytes_delta > 0:
            self.read_bytes = cur_read_bytes
            self.update_method(read_bytes_delta)
        return

    def __monitor(self):
        while True:
            if self.is_finished:
                self.__refresh()
                break
            self.__refresh()
            time.sleep(self.refresh_time)

    def get_default_update_method(self):
        pbar = tqdm(total=self.file_size, desc="reading data")
        self.close_progress_bar_callback = lambda: pbar.close()
        return lambda x: pbar.update(x)
    
    def get_default_handler(self, file_ptr):
        def _handler(file_ptr=file_ptr):
            return file_ptr.read()
        return _handler
    
    def init(self):
        if self.update_method is None:
            self.update_method = self.get_default_update_method()
        if self.handler is None:
            self.handler = self.get_default_handler(self.file_ptr)
        self.thread.start()

    def __call__(self, file_path):
        # init params
        self.is_finished = False
        self.file_ptr = open(file_path, "r", encoding="utf-8")
        self.file_size = os.path.getsize(file_path)
        try:
            self.init()
            data = self.handler(file_ptr=self.file_ptr)
        except Exception:
            self.is_finished = True
            self.thread.join()
            self.file_ptr.close()
            raise
        
        self.is_finished = True
        self.thread.join()
        self.close_progress_bar_callback()
        self.file_ptr.close()
        
        return data


def read_text_by_line(file_path, handler_func=None):
    if handler_func and not callable(handler_func):
        raise RuntimeError("param `map_func` pass to `read_text` is not callable!")

    text_contents = []
    file_size = os.path.getsize(file_path) // 1024

    pbar = tqdm(total=file_size, desc="Reading file {}".format(file_path.split("/")[-1]))
    read_bytes = 0
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            cur_read_bytes = f.tell()
            pbar.update((cur_read_bytes - read_bytes) // 1024)
            read_bytes = cur_read_bytes

            handled_texts = handler_func(line) if handler_func is not None else line
            if isinstance(handled_texts, str):
                text_contents.append(handled_texts)
            elif isinstance(handled_texts, Iterable):
                for handled_text in handled_texts:
                    text_contents.append(handled_text)
            else:
                raise RuntimeError("Unexpected text map from                   \
                                    `str` to `{}`".format(type(handled_texts)))
    return text_contents

class Reader(ComponentHead):

    def get_extension(self):
        if not self.file_path:
            raise RuntimeError("Reader must initialize a file_path!")
        return os.path.splitext(self.file_path)[1]

"""
针对json类的数据进行读取
例如：
data = {
    "name": "test data",
    "desc": "test data in json extension",
    "language": "en",
    "data": {
        "max_length": 100,
        "label_nums": 10,
        "content": [
            {"sentence": "This is a line.", "lable": 5},
            {"sentence": "This is a new line.", "lable": 3},
            {"sentence": "A line is declared here.", "lable": 8},
            {"sentence": "Athoner line.", "lable": 7},
            {"sentence": "Bad line.", "lable": 1}
        ]
    }

可以针对指定jsonpath来获取对应的字段,如获取content中的sentence字段:
reader = JsonDataReader("test.json", "name.data.content[:].sentence")

也可以指定映射函数对指定item进行操作
reader = JsonDataReader(
        "test.json",
        "name.data.content[:].sentence",
        lambda x: x.strip()
)
"""
class JsonDataReader(Reader):
    def __init__(self, **kwargs) -> None:
        args = Configurable(kwargs)
        self.file_path = args.file_path
        self.map_func = args.map_func
        self.json_eval_path = args.json_eval_path
        self.data = []

    def get_data(self, file_path, reader):
        if self.map_func is not None:
            return list(map(self.map_func, reader(file_path)))
        else:
            return reader(file_path)

    def __call__(self, component=None):
        if component is not None:
            raise RuntimeError("A data reader is `ComponentHead`,"         \
                               "cannot assign component before it.")
        if isinstance(self.file_path, str):
            reader = ReadFileWithProgressBar(json_handler(self.json_eval_path))
            self.data = self.get_data(self.file_path, reader)
        elif isinstance(self.file_path, list):
            for f in self.file_path:
                reader = ReadFileWithProgressBar(json_handler(self.json_eval_path))
                self.data.extend(self.get_data(f, reader))
        else:
            raise NotImplementedError("unimplemented type of file_path with  \
                                       `{}`".format(type(self.file_path)))
        return self

class JsonLineDataReader(Reader):
    def __init__(self, **kwargs):
        args = Configurable(kwargs)
        self.file_path = args.file_path
        self.json_eval_path = args.json_eval_path
        self.map_func = self.wrap_map_func(args.map_func)
        self.data = []

    def wrap_map_func(self, map_func):
        def wraped_func(data):
            data = json.loads(data)
            if self.json_eval_path:
                data = eval_json_path(data, self.json_eval_path)
            if not map_func:
                return data
            return map_func(data)
        return wraped_func

    def __call__(self, component=None):
        if component is not None:
            raise RuntimeError("A data reader is `ComponentHead`,"         \
                               "cannot assign component before it.")
        if isinstance(self.file_path, str):
            self.data = read_text_by_line(self.file_path, self.map_func)
        elif isinstance(self.file_path, list):
            for f in self.file_path:
                self.data.extend(read_text_by_line(f, self.map_func))
        else:
            raise NotImplementedError("unimplemented type of file_path with  \
                            `{}`".format(type(self.file_path)))
        return self

class PureTextDataReader(Reader):
    def __init__(self, **kwargs):
        args = Configurable(kwargs)
        self.file_path = args.file_path
        self.map_func = args.map_func
        self.data = []

    def __call__(self, component=None):
        if component is not None:
            raise RuntimeError("A data reader is `ComponentHead`,"         \
                               "cannot assign component before it.")
        if self.map_func is not None and not callable(self.map_func):
            raise ValueError("Passed map_func param,but it is not callable!")
        if isinstance(self.file_path, str):
            self.data = read_text_by_line(self.file_path, self.map_func)
        elif isinstance(self.file_path, list):
            for f in self.file_path:
                self.data.extend(read_text_by_line(f, self.map_func))
        else:
            raise NotImplementedError("unimplemented type of file_path with  \
                            `{}`".format(type(self.file_path)))
        return self

class DataFrameDataReader:
    pass

class ParquetDataReader:
    pass

class PdfDataReader:
    pass


def get_reader(filename: str):
    extension = filename.split(".")[-1]
    if extension == "txt":
        return PureTextDataReader
    elif extension == "json":
        return JsonDataReader
    elif extension == "jsonl":
        return JsonLineDataReader
    else:
        raise NotImplementedError("Unimplemented data reader for type `{}`".format(extension))
