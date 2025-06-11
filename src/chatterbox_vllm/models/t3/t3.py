from typing import Iterable, Optional

import torch
import torch.nn as nn
from vllm.model_executor.models.interfaces_base import VllmModelForTextGeneration
from vllm.model_executor.models.llama import LlamaModel
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.sequence import IntermediateTensors
from vllm.config import VllmConfig, ModelConfig


from chatterbox_vllm.models.t3.modules.learned_pos_emb import LearnedPositionEmbeddings
from chatterbox_vllm.models.t3.modules.t3_config import T3Config

from .modules.cond_enc import T3CondEnc

from vllm.transformers_utils.tokenizer_base import TokenizerRegistry
TokenizerRegistry.register("EnTokenizer", "chatterbox_vllm.models.t3.entokenizer", "EnTokenizer")

class T3VllmModel(nn.Module, VllmModelForTextGeneration):
    """Native vLLM implementation of the Chatterbox T3 """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str):
        super().__init__()
        self.vllm_config = vllm_config
        self.cfg: ModelConfig = vllm_config.model_config

        # Initialize LLaMA backbone
        self.tfmr = LlamaModel(vllm_config=vllm_config, prefix=prefix)

        # Initialize custom components
        t3enc_config = T3Config()
        self.dim = t3enc_config.n_channels
        self.cond_enc = T3CondEnc(t3enc_config)
        self.text_emb = nn.Embedding(t3enc_config.text_tokens_dict_size, self.dim)
        self.speech_emb = nn.Embedding(t3enc_config.speech_tokens_dict_size, self.dim)

        # custom position embedding
        if t3enc_config.input_pos_emb == "learned":
            max_text_seq_len = t3enc_config.max_text_tokens + 2
            self.text_pos_emb = LearnedPositionEmbeddings(max_text_seq_len, self.dim)

            max_mel_seq_len = t3enc_config.max_speech_tokens + 2 + 2
            self.speech_pos_emb = LearnedPositionEmbeddings(max_mel_seq_len, self.dim)

        # logit projection
        self.text_head = nn.Linear(self.dim, t3enc_config.text_tokens_dict_size, bias=False)
        # self.speech_head = nn.Linear(self.dim, t3enc_config.speech_tokens_dict_size, bias=False)

        self.speech_head = ParallelLMHead(t3enc_config.speech_tokens_dict_size, self.dim, prefix=prefix)
        self.logits_processor = LogitsProcessor(t3enc_config.speech_tokens_dict_size)
       
    
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loaded_params: set[str] = set()
        state_dicts = {}
        hf_llama_weights = {}
        for name, weight in weights:
            # Llama weights need to be passed through vllm's load_weights rather than load_state_dict
            if name.startswith("tfmr."):
                subname = name[5:]
                hf_llama_weights[subname] = weight
                continue
            
            loaded_params.add(name)
            attr, subname = name.split('.', 1)
            state_dict = state_dicts.get(attr, {})
            state_dict[subname] = weight
           
        for attr, state_dict in state_dicts.items():
            getattr(self, attr).load_state_dict(state_dict)

        llama_loaded_params = self.tfmr.load_weights(hf_llama_weights.items())
        loaded_params.update('tfmr.' + i for i in llama_loaded_params)
    
        return loaded_params


    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.speech_emb(input_ids)
    

    def compute_logits(self, hidden_states: torch.Tensor, sampling_metadata: SamplingMetadata) -> torch.Tensor:
        print("compute_logits", hidden_states.shape)
        return self.logits_processor(self.speech_head, hidden_states, sampling_metadata)


    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors],
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        inputs_embeds = self.get_input_embeddings(input_ids)
        # print("inputs_embeds", inputs_embeds.shape)

        return self.tfmr.forward(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds
        )
        