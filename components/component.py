from typing import Any

"""
Component和ComponentHead都需要实现gather方法
否则就默认返回data
"""
class ProcessMeta(type):
    def __or__(self, other):
        return Pipeline(self, other)

class Component(metaclass=ProcessMeta):
    def __init__(self) -> None:
        self.data = None

    def gather(self):
        return self.data
    
    @classmethod
    def partial(cls, **kwargs):
        class Partial(cls):
            def __init__(self, *args, **kwargs2):
                super().__init__(*args, **{**kwargs, **kwargs2})
        return Partial

class ComponentHead(Component):
    def __init__(self) -> None:
        super().__init__()

    def process(self, pre=None):
        if pre is not None:
            raise RuntimeError("Cannot assign pre node before `ComponentHead`!")

class ComponentNode(Component):
    def __init__(self) -> None:
        super().__init__()

    def process(self, pre):
        if pre is None:
            raise RuntimeError("You should assign pre node before `Component`!")

"""
Pipeline用法
pipeline_head = PipelineNode()
reader = JsonDataReader(
            "test.json",
            "name.data.content[:].sentence",
            lambda x: x.strip()
        )
pre_processor = Preprocessor()
lm_processor = PretrainDatasetMapper()

链式调用,数据通过component.data传递
pipeline = pipeline_head | reader | pre_processor | lm_processor

通过gather获取链式调用最终的数据
dataset = result.gather()
"""
class Pipeline:
    def __init__(self, *args) -> None:
        self.classes = args
            
    def __or__(self, other) -> Any:
        return Pipeline(*self.classes, other)

    def __call__(self, **kwargs):

        pre_node = None
        data = None
        for node in self.classes:
            if node is None:
                return None
            node_instance = node(**kwargs)
            data = node_instance(pre_node)
            pre_node = node_instance
            if data is None:
                return None
        return data