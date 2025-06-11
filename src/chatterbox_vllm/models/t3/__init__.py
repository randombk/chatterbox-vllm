from .t3 import T3VllmModel
from vllm import ModelRegistry

ModelRegistry.register_model("ChatterboxT3", T3VllmModel)
