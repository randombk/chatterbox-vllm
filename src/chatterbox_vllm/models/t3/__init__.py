from .t3 import T3VllmModel
from vllm import ModelRegistry
from vllm.transformers_utils.tokenizer_base import TokenizerRegistry

ModelRegistry.register_model("ChatterboxT3", T3VllmModel)
TokenizerRegistry.register("EnTokenizer", "chatterbox_vllm.models.t3.entokenizer", "EnTokenizer")
