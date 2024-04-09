# Unified modeling config
import re
import os
import json

JSON_INDENT_NUM = 4

# 不要为配置类新增属性，只覆盖
def setattr_s(obj, key, value):
    if hasattr(obj, key):
        setattr(obj, key, value)
    else:
        raise ValueError("Cannot deserialize config file for "
                         "attribute {} not in {} class".format(key, type(obj).__name__))
    
def getattr_s(obj, key):
    if hasattr(obj, key):
        return getattr(obj, key)
    else:
        raise ValueError("Attribute {} does not exist in"
                         " {} class".format(key, type(obj).__name__))
    
def check_before_deserialize(obj, json_dict):
    if obj is None or json_dict is None:
        return False
    
    obj_keys_set = set(obj.keys())
    json_dict_keys_set = set(json_dict.keys())

    if obj_keys_set is None or json_dict_keys_set is None:
        return False
    
    if obj_keys_set == json_dict_keys_set:
        return True
    return False


'''
config的注解类，只要在实现配置类的时候加上注解，即可自动加上
打印成json格式、序列化和反序列化的功能而不需要手动实现
例如：
@config_class("config.json")
class MyConfig:
    def __init__(self):
        self.config_a = 1.
        self.config_b = "xxx/yyy"
        self.config_c = True
        self.config_d = "cuda"

使用时，可以这样序列化：
config = Config()
config.save()

也可以反序列化：
config = Config.load()

还可以直接打印，输出成格式化后的json字符串：
print(config)
'''
def config_class(path):
    if not os.path.exists(path):
        os.makedirs(path)
    def decorator(cls):
        if os.path.isdir(path):
            save_path = os.path.join(path, "{}.json".format(cls.__name__))
        elif os.path.isfile(path):
            if path.endswith(".json"):
                save_path = path
            else:
                raise NotImplementedError("You specified an unimplemeted file type to save config, "
                                          "currently does not support other file type except json")
        else:
            raise ValueError("Unsupported save config path {}".format(path))

        def _to_dict(self):
            _dict = {}
            for key in self.__dict__:
                if key.startswith("__") or key.endswith("__"):
                    continue
                _dict[key] = getattr_s(self, key)
            return _dict

        def _save(self, save_path=save_path):
            with open(save_path, 'w') as f_write:
                json.dump(self.to_dict(), f_write, indent=JSON_INDENT_NUM)

        def _to_json_str(self):
            return json.dumps(self.to_dict(), indent=JSON_INDENT_NUM)
        
        def __str__(self):
            return self.to_json_str()

        def _load_from_instance(self, save_path=save_path):
            with open(save_path, 'r') as file:
                data = json.load(file)
            
            if not check_before_deserialize(self.to_dict(), data):
                raise RuntimeError("Mismatch config file {} for"
                                   " config class {}".format(save_path, type(self).__name__))
            
            for key in data:
                setattr_s(self, key, data[key])
        
        def _load():
            instance = cls()
            instance.load_from_instance()
            return instance

        setattr(cls, 'to_dict', _to_dict)
        setattr(cls, 'save', _save)
        setattr(cls, 'to_json_str', _to_json_str)
        setattr(cls, 'load_from_instance', _load_from_instance)
        setattr(cls, '__str__', __str__)
        setattr(cls, 'load', staticmethod(_load))
        return cls
    return decorator


def generate_config(py_file):
    # 读取Python代码文件
    with open(py_file, 'r', encoding="utf-8") as file:
        code = file.read()
    variable_names = re.findall(r'(?<=config\.)\w+', code)

    config_fields = set(variable_names)
    print(config_fields)
    print("param nums is {}".format(len(config_fields)))


class Configurable:
    def __init__(self, data):
        self.data = data

    def __getattr__(self, key):
        if key in self.data:
            return self.data[key]
        else:
            return None
