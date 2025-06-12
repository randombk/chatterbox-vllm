from typing import Iterable, Mapping, Optional, Sequence, Union

import torch
import torch.nn as nn
from transformers.feature_extraction_utils import BatchFeature

from vllm.config import VllmConfig, ModelConfig
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.models.interfaces import MultiModalEmbeddings, SupportsMultiModal
from vllm.model_executor.models.interfaces_base import VllmModelForTextGeneration
from vllm.model_executor.models.llama import LlamaModel
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalKwargs
from vllm.multimodal.parse import MultiModalDataParser, ModalityDataItems
from vllm.multimodal.processing import (
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    EmbeddingItems,
    MultiModalDataDict,
    MultiModalDataItems,
    MultiModalFieldConfig,
    PromptReplacement,
    PromptUpdate,
    MultiModalInputs,
    PlaceholderRange,
)
from vllm.multimodal.profiling import BaseDummyInputsBuilder
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.tokenizer_base import TokenizerRegistry

from chatterbox_vllm.models.t3.modules.learned_pos_emb import LearnedPositionEmbeddings
from chatterbox_vllm.models.t3.modules.t3_config import T3Config
from .modules.cond_enc import T3Cond, T3CondEnc

# Register tokenizer
TokenizerRegistry.register("EnTokenizer", "chatterbox_vllm.models.t3.entokenizer", "EnTokenizer")


class T3ProcessingInfo(BaseProcessingInfo):
    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        return {"conditionals": 1}


class T3MultiModalDummyInputsBuilder(BaseDummyInputsBuilder):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return "[START]Hello, world![STOP]"
    
    def get_dummy_mm_data(self, seq_len: int, mm_counts: Mapping[str, int]) -> MultiModalDataDict:
        return { "conditionals": None }
        # return { "conditionals": [torch.tensor([1, 2, 3])] }
        # return { 
        #     "conditionals": T3Cond(
        #         speaker_emb=torch.tensor([1, 2, 3]),
        #         clap_emb=torch.tensor([4, 5, 6]),
        #         cond_prompt_speech_tokens=torch.tensor([7, 8, 9]),
        #         cond_prompt_speech_emb=torch.tensor([10, 11, 12])
        #     )
        # }


class T3MultiModalDataParser(MultiModalDataParser):
    def parse_mm_data(self, mm_data: MultiModalDataDict) -> MultiModalDataItems:
        conditionals: Optional[torch.Tensor] = mm_data.get("conditionals", None)
        if conditionals is None:
            return MultiModalDataItems({})
        
        return MultiModalDataItems({
            "conditionals": ConditionalsEmbeddingItems(conditionals)
        })


class ConditionalsEmbeddingItems(ModalityDataItems[T3Cond, T3Cond]):
    def __init__(self, data: T3Cond) -> None:
        super().__init__(data, "conditionals")

    def get_count(self) -> int:
        return 1

    def get(self, index: int) -> torch.Tensor:
        return self.data

    def get_processor_data(self) -> Mapping[str, T3Cond]:
        return {}

    def get_passthrough_data(self) -> Mapping[str, T3Cond]:
        return {"conditionals": self.data}

    def get_feature_size(self, item_idx: int) -> int:
        return len(self.data.speaker_emb) + \
            (len(self.data.clap_emb) if self.data.clap_emb is not None else 0) + \
            (len(self.data.cond_prompt_speech_tokens) if self.data.cond_prompt_speech_tokens is not None else 0) + \
            (len(self.data.cond_prompt_speech_emb) if self.data.cond_prompt_speech_emb is not None else 0)
       

class T3MultiModalProcessor(BaseMultiModalProcessor[T3ProcessingInfo]):
    def _get_data_parser(self) -> MultiModalDataParser:
        return T3MultiModalDataParser()
    
    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return dict(
            conditionals=MultiModalFieldConfig.batched("conditionals")
        )

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargs,
    ) -> Sequence[PromptUpdate]:
        return [
            # Delibrate noop
            # PromptReplacement(
            #     modality="conditionals",
            #     target="[START]",
            #     replacement="[START]",
            # )
        ]

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        tokenizer = self.info.get_tokenizer()
        processed_outputs = tokenizer(prompt, return_tensors="pt")
        processed_outputs['conditionals'] = mm_data.get('conditionals', None)
        if processed_outputs['conditionals'] is not None:
            print("processed_outputs", processed_outputs['conditionals'].shape)
        return processed_outputs

    def apply(
        self,
        prompt: Union[str, list[int]],
        mm_data: MultiModalDataDict,
        hf_processor_mm_kwargs: Mapping[str, object],
        return_mm_hashes: bool = False,
    ) -> MultiModalInputs:
        """
        Process multi-modal inputs to be used in vLLM.

        The main steps are:

        1. Apply HF Processor on prompt text and multi-modal data together,
           outputting token IDs and processed tensors.
        2. Find and update sequences in the token IDs with placeholder tokens.
           The number of placeholder tokens equals the feature size of the
           multi-modal data outputted by the multi-modal encoder.
        3. Extract information about the placeholder tokens from the
           processed token IDs.
        """
        mm_items = self._to_mm_items(mm_data)
    
        (
            prompt_ids,
            mm_kwargs,
            mm_hashes,
            is_update_applied,
        ) = self._apply_hf_processor(
            prompt,
            mm_items,
            hf_processor_mm_kwargs,
            
            # Skip prompt caching calculation for now        
            return_mm_hashes=False,
        )

        # prompt_ids, prompt, mm_placeholders = self._maybe_apply_prompt_updates(
        #     mm_items=mm_items,
        #     hf_processor_mm_kwargs=hf_processor_mm_kwargs,
        #     prompt_ids=prompt_ids,
        #     mm_kwargs=mm_kwargs,
        #     is_update_applied=is_update_applied,
        # )

        # mm_placeholder_ranges = {
        #     modality: [item.to_range() for item in placeholders]
        #     for modality, placeholders in mm_placeholders.items()
        # }

        return MultiModalInputs(
            type="multimodal",
            prompt=prompt,
            prompt_token_ids=prompt_ids,
            mm_kwargs=mm_kwargs,
            mm_hashes={
                "conditionals": ["foo"],
            },
            mm_placeholders={
                "conditionals": [PlaceholderRange(offset=0, length=1, is_embed=None)]
            },
        )


@MULTIMODAL_REGISTRY.register_processor(T3MultiModalProcessor,
                                        info=T3ProcessingInfo,
                                        dummy_inputs=T3MultiModalDummyInputsBuilder)
class T3VllmModel(nn.Module, VllmModelForTextGeneration, SupportsMultiModal):
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


    def get_multimodal_embeddings(self, **kwargs: object) -> Optional[MultiModalEmbeddings]:
        conditionals: Optional[list[list[T3Cond]]] = kwargs.get("conditionals", None)
        
        if conditionals is None:
            return torch.zeros(256, 1, self.dim)
        
        # WIP
        return torch.zeros(len(conditionals), 1, self.dim)
        
        result = []
        for batch in conditionals:
            t3_cond = batch

            if not isinstance(t3_cond, T3Cond):
                print(t3_cond)
                t3_cond = t3_cond[0]

            if not isinstance(t3_cond, T3Cond):
                print(t3_cond)
                t3_cond = t3_cond[0]

            if not isinstance(t3_cond, T3Cond):
                print(t3_cond)
                t3_cond = t3_cond[0]

            if t3_cond.cond_prompt_speech_tokens is not None and t3_cond.cond_prompt_speech_emb is None:
                t3_cond.cond_prompt_speech_emb = self.speech_emb(t3_cond.cond_prompt_speech_tokens) + \
                    self.speech_pos_emb(t3_cond.cond_prompt_speech_tokens)
            result.append(self.cond_enc(t3_cond))
        return result


    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[MultiModalEmbeddings] = None,
    ) -> torch.Tensor:
        print("get_input_embeddings", input_ids.shape if input_ids is not None else None)
        print("get_input_embeddings", [i.shape for i in (multimodal_embeddings or [])])
        return self.speech_emb(input_ids)


    def compute_logits(self, hidden_states: torch.Tensor, sampling_metadata: SamplingMetadata) -> torch.Tensor:
        return self.logits_processor(self.speech_head, hidden_states, sampling_metadata)


    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors],
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> torch.Tensor:
        print("forward", input_ids.shape if input_ids is not None else None)
        print("forward", [i.shape for i in (intermediate_tensors or [])])
        print("forward", inputs_embeds.shape if inputs_embeds is not None else None)
        print("forward", kwargs.keys())

        if intermediate_tensors is not None:
            inputs_embeds = None
        elif inputs_embeds is None:
            vision_embeddings = self.get_multimodal_embeddings(**kwargs)
            inputs_embeds = self.get_input_embeddings(input_ids,
                                                      vision_embeddings)
            input_ids = None

        hidden_states = self.tfmr(input_ids,
                                            positions,
                                            intermediate_tensors,
                                            inputs_embeds=inputs_embeds)
        return hidden_states

    def get_language_model(self) -> torch.nn.Module:
        return self.tfmr


    def prepare_conditioning(self, t3_cond: T3Cond):
        """
        Token cond data needs to be embedded, so that needs to be here instead of in `T3CondEnc`.
        """
        if t3_cond.cond_prompt_speech_tokens is not None and t3_cond.cond_prompt_speech_emb is None:
            t3_cond.cond_prompt_speech_emb = self.speech_emb(t3_cond.cond_prompt_speech_tokens) + \
                self.speech_pos_emb(t3_cond.cond_prompt_speech_tokens)
        return self.cond_enc(t3_cond)  # (B, len_cond, dim)
