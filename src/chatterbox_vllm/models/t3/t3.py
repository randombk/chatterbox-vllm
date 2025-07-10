from typing import Iterable, Mapping, Optional, Sequence, Union

import torch
import torch.nn as nn
import random
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
    MultiModalDataDict,
    MultiModalDataItems,
    MultiModalFieldConfig,
    PromptUpdate,
    MultiModalInputs,
    PlaceholderRange,
    PromptReplacement,
    PromptUpdate,
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
        # This is hacked around in the get_multimodal_embeddings method for now.
        return { "conditionals": [torch.zeros(0)] * mm_counts["conditionals"] }


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

    # def get_feature_size(self, item_idx: int) -> int:
    #     return self.data.speaker_emb.shape[0] + \
    #         self.data.clap_emb.shape[0] + \
    #         self.data.cond_prompt_speech_tokens.shape[0] + \
    #         self.data.cond_prompt_speech_emb.shape[0]
       

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
        # Bypassed via `apply` method.
        return []
        # if mm_items.get_all_counts().get('conditionals', 0) == 0:
        #     return []

        # return [
        #     # The final embedding will look like <| cond | text | speech |>
        #     # This will prepare the cond portion.
        #     PromptReplacement(
        #         modality="conditionals",
        #         target="[START]",
        #         # replacement=["[START]"] * (mm_items.get_items('conditionals', ConditionalsEmbeddingItems).get_feature_size(0) + 1),
        #         replacement=["[START]"] * 15,
        #     )
        # ]

    def _call_hf_processor(
        self,
        prompt: str,
        # Not to be confused with `mm_data` in `self.apply`.
        # This refers to the data to be passed to HF processor.
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
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
        tokenization_kwargs: Optional[Mapping[str, object]] = None,
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
           (SKIPPED for T3 conditioning)
        3. Extract information about the placeholder tokens from the
           processed token IDs.
           (Stubbed for T3 conditioning)
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
            tokenization_kwargs,
            
            # Skip prompt caching calculation for now        
            return_mm_hashes=False,
        )

        # We are going to apply custom logic to squish the embeddings in the right format.
        # The final embedding will look like <| cond | text | speech |>
        n_cond_tokens = 34 # 1 for speaker_emb, 0 for clap_emb, 32 for cond_prompt_speech_emb, 1 for emotion_adv
        prompt = "[START]" * n_cond_tokens + prompt + "[STOP]"
        prompt_ids = (
            [prompt_ids[0]] * n_cond_tokens # Conditionals
            + prompt_ids # Text prompt
            + [0] # Start of speech token
        )

        return MultiModalInputs(
            type="multimodal",
            prompt=prompt,
            prompt_token_ids=prompt_ids,
            mm_kwargs=mm_kwargs,
            mm_hashes={
                # Assign a random hash for now, because we're not actually hashing the multimodal data.
                "conditionals": [str(random.random())],
            },
            mm_placeholders={
                "conditionals": [PlaceholderRange(offset=0, length=n_cond_tokens, is_embed=None)]
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
        self.tfmr = LlamaModel(vllm_config=vllm_config, prefix=prefix + ".tfmr")

        # Initialize custom components
        self.t3conf = T3Config()
        self.dim = self.t3conf.n_channels
        self.cond_enc = T3CondEnc(self.t3conf)
        self.text_emb = nn.Embedding(self.t3conf.text_tokens_dict_size, self.dim)
        self.speech_emb = nn.Embedding(self.t3conf.speech_tokens_dict_size, self.dim)

        # custom position embedding
        if self.t3conf.input_pos_emb == "learned":
            max_text_seq_len = self.t3conf.max_text_tokens + 2
            self.text_pos_emb = LearnedPositionEmbeddings(max_text_seq_len, self.dim)

            max_mel_seq_len = self.t3conf.max_speech_tokens + 2 + 2
            self.speech_pos_emb = LearnedPositionEmbeddings(max_mel_seq_len, self.dim)

        # logit projection
        self.text_head = nn.Linear(self.dim, self.t3conf.text_tokens_dict_size, bias=False)
        self.speech_head = ParallelLMHead(
            num_embeddings=self.t3conf.speech_tokens_dict_size,
            embedding_dim=self.dim,
            padding_size=1,
            prefix=prefix + ".speech_head",
        )
        self.logits_processor = LogitsProcessor(self.t3conf.speech_tokens_dict_size)

        # HACK: We need some way to track the number of text tokens in the prefill stage.
        self._text_tokens_len = 0


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
            state_dicts[attr] = state_dict

        for attr, state_dict in state_dicts.items():
            # print("Loading weights:", attr, state_dict.keys())
            getattr(self, attr).load_state_dict(state_dict)

        llama_loaded_params = self.tfmr.load_weights(hf_llama_weights.items())
        loaded_params.update('tfmr.' + i for i in llama_loaded_params)

        # Precompute speech positional embeddings
        position_ids = torch.arange(self.vllm_config.model_config.max_model_len, device=self.speech_head.weight.device)
        self.precomputed_speech_pos_emb = self.speech_pos_emb.get_fixed_embedding(position_ids)[0]

        return loaded_params


    def get_multimodal_embeddings(self, **kwargs: object) -> Optional[MultiModalEmbeddings]:
        conditionals: Optional[list[list[T3Cond]]] = kwargs.get("conditionals", [])
        
        result = []
        for batch in conditionals:
            if len(batch[0]) == 0:
                result.append(torch.zeros(34, self.dim))
                continue

            speaker_emb, clap_emb, cond_prompt_speech_tokens, cond_prompt_speech_emb, emotion_adv = batch[0]
            
            if cond_prompt_speech_tokens.shape != (0,) and cond_prompt_speech_emb.shape == (0,):
                cond_prompt_speech_emb = self.speech_emb(cond_prompt_speech_tokens)[0] + self.speech_pos_emb(cond_prompt_speech_tokens)
            
            t3_cond = T3Cond(
                speaker_emb=speaker_emb,
                clap_emb=clap_emb,
                cond_prompt_speech_tokens=cond_prompt_speech_tokens,
                cond_prompt_speech_emb=cond_prompt_speech_emb,
                emotion_adv=emotion_adv
            )
            result.append(self.cond_enc(t3_cond))
        return result


    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[MultiModalEmbeddings] = None,
    ) -> torch.Tensor:
        if multimodal_embeddings is None or len(multimodal_embeddings) == 0:
            # There's no multimodal embeddings, so we're decoding.
            return self.speech_emb(input_ids)
        else:
            # We're in the prefill stage, and need to wrangle the multimodal embeddings into the right format.
            # Embeddings are in the format of <| cond | text | speech |>
            
            # The first 34 tokens are the cond portion. The remainder, except for the last token are the text
            # portion. The last token is a placeholder for the start of speech token.
            conds = multimodal_embeddings[0]
            text_ids = input_ids[34:-1]
            text_emb = self.text_emb(text_ids)

            # HACK: Remember the number of text tokens so we can apply speech positional embeddings later
            # This will likely break batching, but will likely need a VLLM change to fix.
            self._text_tokens_len = len(text_ids)

            # text_emb[1].zero_()  # CFG uncond
            # print("text_emb", text_emb)

            start_of_speech_token = torch.tensor([self.t3conf.start_speech_token]).to(input_ids.device)
            start_of_speech_emb = self.speech_emb(start_of_speech_token.unsqueeze(0))[0]

            if self.t3conf.input_pos_emb == "learned":
                text_emb = text_emb + self.text_pos_emb(text_ids.unsqueeze(0))
                start_of_speech_emb = start_of_speech_emb + self.precomputed_speech_pos_emb[0]

            embeds = torch.cat([conds, text_emb, start_of_speech_emb], dim=0)

            # for i in range(len(embeds)):
            #     print("embeds", i, embeds[i])
            return embeds


    def compute_logits(self, hidden_states: torch.Tensor, sampling_metadata: SamplingMetadata) -> torch.Tensor:
        logits = self.logits_processor(self.speech_head, hidden_states, sampling_metadata)
        # print("t3/compute_logits/logit with the highest probability:", logits.argmax())
        return logits


    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors],
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> torch.Tensor:
        # print("t3/inputs_embeds", inputs_embeds.shape, inputs_embeds.dtype)
        # print("t3/positions", positions)

        # These are usually NULL:
        # print("t3/intermediate_tensors", intermediate_tensors)
        # print("t3/input_ids", input_ids)
        # print("t3/kwargs", kwargs)

        # HACK: We are going to apply speech positional embeddings here
        if len(inputs_embeds) == 1:
            position_offset = positions - (self._text_tokens_len + 34) # 0 is already accounted for via the start of speech token
            inputs_embeds = inputs_embeds + self.precomputed_speech_pos_emb[position_offset]
    
        hidden_states = self.tfmr(input_ids, positions, intermediate_tensors, inputs_embeds=inputs_embeds)

        # if len(inputs_embeds) > 1:
        #     print("t3/compute_logits/hidden_states", hidden_states.shape, hidden_states.dtype, hidden_states)

        return hidden_states

    def get_language_model(self) -> torch.nn.Module:
        return self.tfmr

