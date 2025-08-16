from typing import Iterable, Mapping, Optional, Sequence, Union
import os

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
from vllm.multimodal.inputs import MultiModalKwargs, MultiModalKwargsItem, MultiModalBatchedField
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
    PromptUpdate,
)
from vllm.multimodal.profiling import BaseDummyInputsBuilder
from vllm.sequence import IntermediateTensors

from chatterbox_vllm.models.t3.modules.learned_pos_emb import LearnedPositionEmbeddings
from chatterbox_vllm.models.t3.modules.t3_config import T3Config
from .modules.cond_enc import T3Cond, T3CondEnc


PREFILL_COND_START_TOKEN = 695  # [PLACEHOLDER55]; Marks the first token of the conditionals
PREFILL_COND_END_TOKEN = 696  # [PLACEHOLDER56]; Marks the last token of the conditionals
PREFILL_END_TOKEN = 697  # [PLACEHOLDER57]; Marks the end of the prefill block. This corresponds to the start of speech token.

CONDITIONING_SIZE = 34 # 1 for speaker_emb, 0 for clap_emb, 32 for cond_prompt_speech_emb, 1 for emotion_adv

# HACK: We need to be able to distinguish between the prefill tokens and the decode tokens.
# We'll do this by offsetting the speech tokens (only within vLLM) so they don't overlap with the
# normal speech tokens. This way, any token < SPEECH_TOKEN_OFFSET is a prefill token, and any token
# >= SPEECH_TOKEN_OFFSET is a decode token. This will only affect the logits and the encoding logic.
# No effect on the hidden states or the actual Llama model itself.
SPEECH_TOKEN_OFFSET = 1000


class T3ProcessingInfo(BaseProcessingInfo):
    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        return {"conditionals": 1}


class T3MultiModalDummyInputsBuilder(BaseDummyInputsBuilder):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return "[START]Hello, world![STOP]"

    def get_dummy_mm_data(self, seq_len: int, mm_counts: Mapping[str, int]) -> MultiModalDataDict:
        return { "conditionals": [torch.zeros(CONDITIONING_SIZE, 2048)] * mm_counts["conditionals"] }


class T3MultiModalDataParser(MultiModalDataParser):
    def parse_mm_data(self, mm_data: MultiModalDataDict) -> MultiModalDataItems:
        conditionals: Optional[torch.Tensor] = mm_data.get("conditionals", None)
        if conditionals is None:
            return MultiModalDataItems({})

        return MultiModalDataItems({
            "conditionals": ConditionalsEmbeddingItems(conditionals)
        })


class ConditionalsEmbeddingItems(ModalityDataItems[torch.Tensor, torch.Tensor]):
    def __init__(self, data: torch.Tensor) -> None:
        super().__init__(data, "conditionals")

    def get_count(self) -> int:
        return 1

    def get(self, index: int) -> torch.Tensor:
        assert index == 0, index
        return self.data

    def get_processor_data(self) -> Mapping[str, torch.Tensor]:
        return {}

    def get_passthrough_data(self) -> Mapping[str, torch.Tensor]:
        return {"conditionals": self.data}


def create_triangular_matrix(m, n):
    # Create row indices and column indices
    row_indices = torch.arange(m).unsqueeze(1)  # Shape: (m, 1)
    col_indices = torch.arange(n).unsqueeze(0)  # Shape: (1, n)

    # Create the triangular mask
    matrix = (col_indices <= row_indices).float()

    return matrix


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
        #
        # For prompt IDs, we're going to replace the input tokens that match the conditionals with a
        # sequence of tokens that won't normally appear in the text prompt. This will help us unbatch
        # batched inputs.
        final_prompt_ids = [
            # Conditionals (totaling CONDITIONING_SIZE tokens)
            PREFILL_COND_START_TOKEN,
            *([prompt_ids[0]] * (CONDITIONING_SIZE-2)),
            PREFILL_COND_END_TOKEN,

            # Text prompt,
            *prompt_ids,

            # Start of speech token / End of prefill block
            PREFILL_END_TOKEN,
        ]

        # HACK: Because vLLM can split the prefill across multiple batches, we need some way to
        # remember the offset of each text token.
        # We'll do this by extending the 32x1024 embedding to <len(final_prompt_ids)>x1024, filling in
        # the first 32x1024 with the original conditionals, and the rest with a triangular matrix of 1s
        # which will encode the offset of each text token.
        conditionals = mm_data.get("conditionals", None)
        assert conditionals is not None and len(conditionals) > 0, "Conditionals are required for prefill"
        assert len(conditionals) == 1, "Only one conditional embedding is supported for prefill"
        assert conditionals[0].shape[0] == CONDITIONING_SIZE, "Conditionals must be CONDITIONING_SIZE tokens long"

        new_conditionals = torch.cat([
            # First CONDITIONING_SIZE embeddings are the original conditionals
            conditionals[0],

            # The positions of the text ids are a triangular matrix of 1s
            create_triangular_matrix(len(prompt_ids), conditionals[0].shape[1]).to(conditionals[0].device),

            # The start of speech token is a vector of 0s,
            torch.zeros(1, conditionals[0].shape[1]).to(conditionals[0].device),
        ], dim=0)
        assert len(new_conditionals) == len(final_prompt_ids), "Number of new conditionals does not match number of prompt ids"

        new_mm_kwargs = MultiModalKwargs.from_items([
            MultiModalKwargsItem.from_elems(
                MultiModalBatchedField().build_elems(
                    modality="conditionals",
                    key="conditionals",
                    data=[new_conditionals],
                )
            )
        ])

        return MultiModalInputs(
            type="multimodal",
            prompt=prompt, # It's unclear if this is actually used for anything. Don't change it for now.
            prompt_token_ids=final_prompt_ids,
            mm_kwargs=new_mm_kwargs,
            mm_hashes={
                # Assign a random hash for now, because we're not actually hashing the multimodal data.
                "conditionals": [str(random.random())],
            },
            mm_placeholders={
                # "conditionals": [PlaceholderRange(offset=0, length=CONDITIONING_SIZE, is_embed=None)]
                # HACK: Tell vLLM that the conditionals modify the entire prompt. This will cause our hacked embeddings
                #       to be injected into the entire prompt, rather than just the conditioning portion.
                "conditionals": [PlaceholderRange(offset=0, length=len(final_prompt_ids), is_embed=None)]
            },
        )


@MULTIMODAL_REGISTRY.register_processor(T3MultiModalProcessor,
                                        info=T3ProcessingInfo,
                                        dummy_inputs=T3MultiModalDummyInputsBuilder)
class T3VllmModel(nn.Module, VllmModelForTextGeneration, SupportsMultiModal):
    """Native vLLM implementation of the Chatterbox T3 """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str):
        super().__init__()
        # HACK: We changed the hidden size to 2048 to trick VLLM into thinking that the model has a hidden size of 2048.
        #       This is needed to accomodate the extra data for the CFG uncond prompt.
        #       We need to change it back to 1024 for loading the actual llama model.
        vllm_config.model_config.hf_config.hidden_size = 1024
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
        max_text_seq_len = self.t3conf.max_text_tokens + 2
        self.text_pos_emb = LearnedPositionEmbeddings(max_text_seq_len, self.dim)

        max_mel_seq_len = self.t3conf.max_speech_tokens + 2 + 2
        self.speech_pos_emb = LearnedPositionEmbeddings(max_mel_seq_len, self.dim)

        # logit projection
        # self.text_head = nn.Linear(self.dim, self.t3conf.text_tokens_dict_size, bias=False)
        self.speech_head = ParallelLMHead(
            num_embeddings=self.t3conf.speech_tokens_dict_size,
            embedding_dim=self.dim,
            padding_size=1,
            prefix=prefix + ".speech_head",
        )
        self.logits_processor = LogitsProcessor(self.t3conf.speech_tokens_dict_size)

        self.cfg_scale = float(os.environ.get("CHATTERBOX_CFG_SCALE", "0.5"))
        print("Applying CFG scale:", self.cfg_scale)


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
            if hasattr(self, attr):
                # print("Loading weights:", attr, state_dict.keys())
                getattr(self, attr).load_state_dict(state_dict)

        llama_loaded_params = self.tfmr.load_weights(hf_llama_weights.items())
        loaded_params.update('tfmr.' + i for i in llama_loaded_params)

        # Precompute text positional embeddings
        text_position_ids = torch.arange(self.t3conf.max_text_tokens + 2, device=self.text_pos_emb.emb.weight.device)
        self.precomputed_text_pos_emb = self.text_pos_emb.get_fixed_embedding(text_position_ids)[0]

        # Precompute speech positional embeddings
        speech_position_ids = torch.arange(self.t3conf.max_speech_tokens + 2 + 2, device=self.speech_pos_emb.emb.weight.device)
        self.precomputed_speech_pos_emb = self.speech_pos_emb.get_fixed_embedding(speech_position_ids)[0]

        return loaded_params


    def get_multimodal_embeddings(self, **kwargs: object) -> Optional[MultiModalEmbeddings]:
        conditionals: Optional[list[list[T3Cond]]] = kwargs.get("conditionals", [])
        return [batch[0] for batch in conditionals]


    def split_prefill_decode(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: list[MultiModalEmbeddings],
    ) -> list[torch.Tensor, Optional[MultiModalEmbeddings]]:
        """
        vLLM combines the prefill and decode into a single input tensor. We need to split them back
        out, and match the decode parts with the corresponding multimodal embeddings.

        Because of the SPEECH_TOKEN_OFFSET, the prefill tokens will always be <SPEECH_TOKEN_OFFSET and
        and the decode tokens will always be >= SPEECH_TOKEN_OFFSET.

        Furthermore, the prefill always starts with PREFILL_COND_START_TOKEN, and
        ends with PREFILL_END_TOKEN. However, vLLM can split the prefill across multiple batches,
        so we won't always have the complete prefill block in a single batch - we might only have the
        beginning or the end of a block.

        We can see back-to-back prefill blocks, so we can't just look for continuous sequences of
        prefill tokens. This nuance is not relevant for decode tokens as their position does not matter.

        Returns a list of tuples, where the first element is the input IDs for the block,
        and the second element is the associated multimodal embedding if the block is a decode part.
        If the block is a prefill part, the second element is None.
        """

        if len(input_ids) == 0:
            return []
        
        remaining_multimodal_embeddings = torch.cat(multimodal_embeddings, dim=0)

        # with open("/ram/input_ids.txt", "w") as f:
        #     f.write(str(input_ids.tolist()))
        # with open("/ram/multimodal_embeddings.txt", "w") as f:
        #     f.write(str([i.tolist() for i in (multimodal_embeddings or [])]))

        in_prefill_block = input_ids[0] < SPEECH_TOKEN_OFFSET

        output = []

        # Keep a buffer of current tokens
        buffer = []

        # Iterate through the element and if we hit block header, set the block state to true
        # Else if we hit the block footer, set block state to false
        # Every time we swtich between block states, add the current buffer to the output if not empty
        # If we we switch out of block mode, add the multimodal embedding
        for input_id in input_ids:
            # Check if we've swapped between prefill and decode blocks, or if we've just hit the start of a new prefill block
            if (in_prefill_block != (input_id < SPEECH_TOKEN_OFFSET)) or (input_id == PREFILL_COND_START_TOKEN):
                if buffer:
                    if in_prefill_block:
                        # assert len(remaining_multimodal_embeddings) >= len(buffer), "Not enough remaining multimodal embeddings"
                        mme, remaining_multimodal_embeddings = remaining_multimodal_embeddings\
                            .split([len(buffer), len(remaining_multimodal_embeddings) - len(buffer)], dim=0)
                        output.append((torch.tensor(buffer).to(input_ids.device), mme))
                    else:
                        output.append((torch.tensor(buffer).to(input_ids.device), None))

                buffer = []
                in_prefill_block = (input_id < SPEECH_TOKEN_OFFSET)

            # Add new token to buffer
            buffer.append(input_id)

        # Add any elements left in the buffer
        if buffer:
            if in_prefill_block:
                # assert len(remaining_multimodal_embeddings) >= len(buffer), "Not enough remaining multimodal embeddings"
                mme, remaining_multimodal_embeddings = remaining_multimodal_embeddings\
                    .split([len(buffer), len(remaining_multimodal_embeddings) - len(buffer)], dim=0)
                output.append((torch.tensor(buffer).to(input_ids.device), mme))
            else:
                output.append((torch.tensor(buffer).to(input_ids.device), None))

        # if len(remaining_multimodal_embeddings) > 0:
        #     print("t3/split_prefill_decode/input_ids", input_ids)
        #     print("t3/split_prefill_decode/remaining_multimodal_embeddings", remaining_multimodal_embeddings.shape)
        #     print("t3/split_prefill_decode/multimodal_embeddings", [i.shape for i in (multimodal_embeddings or [])])
        # assert len(remaining_multimodal_embeddings) == 0, "Number of multimodal embeddings does not match number of prefill blocks"

        # assert sum(len(i[0]) for i in output) == len(input_ids), "Number of output elements does not match number of input elements"
        return output


    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[MultiModalEmbeddings] = None,
    ) -> torch.Tensor:
        if multimodal_embeddings is None or len(multimodal_embeddings) == 0:
            # There's no multimodal embeddings, so we're decoding.
            # Remember to undo the offset we applied to the speech tokens.
            embeds = self.speech_emb(input_ids - SPEECH_TOKEN_OFFSET)

            out = torch.cat([embeds, embeds], dim=1)

            # if len(out) != len(input_ids):
            #     print("t3/get_input_embeddings/out", out.shape, out.dtype)
            #     print("t3/get_input_embeddings/input_ids", input_ids.shape, input_ids.dtype)
            # assert len(out) == len(input_ids), "Number of output elements does not match number of input elements"
            return out
        else:
            # print("t3/get_input_embeddings/multimodal_embeddings", len(multimodal_embeddings))
            # print("t3/get_input_embeddings/input_ids", input_ids.shape, input_ids.dtype, input_ids)
            # print("t3/get_input_embeddings/multimodal_embeddings", [i.shape for i in (multimodal_embeddings or [])])

            out = []
            for ids, multimodal_embedding in self.split_prefill_decode(input_ids, multimodal_embeddings):
                # print("t3/get_input_embeddings/ids", ids.shape, ids.dtype, ids)
                # print("t3/get_input_embeddings/multimodal_embedding", multimodal_embedding.shape if multimodal_embedding is not None else None)

                if multimodal_embedding is None:
                    # There's no multimodal embeddings, so we're decoding.
                    # Remember to undo the offset we applied to the speech tokens.
                    embeds = self.speech_emb(ids - SPEECH_TOKEN_OFFSET)
                    final_embeds = torch.cat([embeds, embeds], dim=1)
                    # assert len(final_embeds) == len(ids), "Number of output elements does not match number of input elements"
                    
                    out.append(final_embeds)
                    continue

                # We're in the prefill stage, and need to wrangle the multimodal embeddings into the right format.
                # Embeddings are in the format of <| cond | text | speech |>
                #
                # However, due to vLLM batching, we may only get the first half or the last half of the prefill block.
                #
                # We're going to assume that the prefill block only span at most two batches - i.e. we'll always have
                # at least the start token or the end token. More is theorically possible, but not an edge case I'm going
                # to handle here.
                #
                # Note that we may have as little as a single token from the block.

                # To ease the implementation logic, we're going to implement each case separately.

                if ids[0] == PREFILL_COND_START_TOKEN and ids[-1] == PREFILL_END_TOKEN:
                    # We have the full prefill block.

                    # The first 34 tokens are the cond portion. The remainder, except for the last token are the text
                    # portion. The last token is a placeholder for the start of speech token.
                    text_ids = ids[CONDITIONING_SIZE:-1]
                    text_emb = self.text_emb(text_ids) + self.precomputed_text_pos_emb[0:len(text_ids)]

                    start_of_speech_token = torch.tensor([self.t3conf.start_speech_token]).to(ids.device)
                    start_of_speech_emb = self.speech_emb(start_of_speech_token.unsqueeze(0))[0] + self.precomputed_speech_pos_emb[0:1]

                    # Generate version with both text and no-text embeddings for CFG
                    conditioning_emb = multimodal_embedding[0:CONDITIONING_SIZE]
                    cond_embeds = torch.cat([conditioning_emb, text_emb, start_of_speech_emb], dim=0)
                    uncond_embeds = torch.cat([conditioning_emb, torch.zeros_like(text_emb), start_of_speech_emb], dim=0)

                    # Concatenate into one giant tensor, which will be split in the forward pass
                    final_embeds = torch.cat([cond_embeds, uncond_embeds], dim=1)
                    # assert len(final_embeds) == len(ids), "Number of output elements does not match number of input elements"
                    out.append(final_embeds)
                elif ids[0] == PREFILL_COND_START_TOKEN:
                    # We have the start of the prefill block.
                    # The only thing we an assume here is that we don't have the end token, so we can skip the start of speech token.
                    # print("t3/get_input_embeddings/start of prefill block")

                    # The first 34 tokens are the cond portion. The remainder are the text portion.
                    # This logic should correctly handle:
                    #  - We don't have any text tokens in this batch
                    #  - We only have part of the conditioning tokens in this batch
                    text_ids = ids[CONDITIONING_SIZE:]
                    text_emb = self.text_emb(text_ids) + self.precomputed_text_pos_emb[0:len(text_ids)]

                    # Generate version with both text and no-text embeddings for CFG
                    conditioning_emb = multimodal_embedding[0:min(CONDITIONING_SIZE, len(multimodal_embedding))]
                    cond_embeds = torch.cat([conditioning_emb, text_emb], dim=0)
                    uncond_embeds = torch.cat([conditioning_emb, torch.zeros_like(text_emb)], dim=0)

                    # Concatenate into one giant tensor, which will be split in the forward pass
                    final_embeds = torch.cat([cond_embeds, uncond_embeds], dim=1)
                    assert len(final_embeds) == len(ids), "Number of output elements does not match number of input elements"
                    out.append(final_embeds)
                elif ids[-1] == PREFILL_END_TOKEN:
                    # We have the end of the prefill block.
                    # The only thing we an assume here is that we have the start of speech token,
                    # and that our conditioning embeddings will at minimum be truncated. We can't
                    # assume anything about the text portion.

                    # Check if the end-of-conditioning token is present. If it is, we can assume that
                    # we have the full text block. If it's not, we can assume that there's no conditioning
                    # portion.
                    indices = torch.where(ids == PREFILL_COND_END_TOKEN)[0]
                    if len(indices) > 0:
                        # print("t3/get_input_embeddings/end of prefill block, has conditioning")
                        
                        # We have the full text input, and it's from indices[0]+1 to the end of the input.
                        # (indices[0] is the end of the conditioning)
                        text_ids = ids[indices[0]+1:-1]
                        text_emb = self.text_emb(text_ids) + self.precomputed_text_pos_emb[0:len(text_ids)]
                        
                        start_of_speech_token = torch.tensor([self.t3conf.start_speech_token]).to(ids.device)
                        start_of_speech_emb = self.speech_emb(start_of_speech_token.unsqueeze(0))[0] + self.precomputed_speech_pos_emb[0:1]

                        conditioning_emb = multimodal_embedding[:indices[0]+1]
                        
                        cond_embeds = torch.cat([conditioning_emb, text_emb, start_of_speech_emb], dim=0)
                        uncond_embeds = torch.cat([conditioning_emb, torch.zeros_like(text_emb), start_of_speech_emb], dim=0)

                        final_embeds = torch.cat([cond_embeds, uncond_embeds], dim=1)
                        # assert len(final_embeds) == len(ids), "Number of output elements does not match number of input elements"
                        out.append(final_embeds)
                    else:
                        # We don't have the conditioning portion, and we may only have part of the text portion.
                        # print("t3/get_input_embeddings/end of prefill block, no conditioning")

                        # Everything except the last token is the text portion.
                        text_ids = ids[:-1]
                        
                        # Extract the position IDs for the text portion by counting the number of 1s (minus 1) in the multimodal embedding
                        # that was injected via our hack above.
                        text_pos = torch.sum(multimodal_embedding[0:len(text_ids)], dim=1) - 1

                        text_emb = self.text_emb(text_ids) + self.precomputed_text_pos_emb[text_pos.tolist()]

                        start_of_speech_token = torch.tensor([self.t3conf.start_speech_token]).to(ids.device)
                        start_of_speech_emb = self.speech_emb(start_of_speech_token.unsqueeze(0))[0]  + self.precomputed_speech_pos_emb[0:1]
                        
                        cond_embeds = torch.cat([text_emb, start_of_speech_emb], dim=0)
                        uncond_embeds = torch.cat([torch.zeros_like(text_emb), start_of_speech_emb], dim=0)
                        final_embeds = torch.cat([cond_embeds, uncond_embeds], dim=1)
                        assert len(final_embeds) == len(ids), "Number of output elements does not match number of input elements"
                        out.append(final_embeds)

                else:
                    # Something else - we don't know what to do with this.
                    print("t3/get_input_embeddings/ERROR: prefill block contains neither start nor end. Please report this issue.")
                    print("t3/get_input_embeddings/ids", ids.shape, ids.dtype, ids)
                    print("t3/get_input_embeddings/multimodal_embedding", multimodal_embedding.shape if multimodal_embedding is not None else None)
                    raise ValueError(f"Unknown prefill block: {ids}")

            output = torch.cat(out, dim=0)

            # if len(output) != len(input_ids):
            #     print("t3/get_input_embeddings/output", output.shape, output.dtype)
            #     print("t3/get_input_embeddings/input_ids", input_ids.shape, input_ids.dtype)
            #     print("t3/get_input_embeddings/multimodal_embeddings", len(multimodal_embeddings))
            return output


    def compute_logits(self, hidden_states: torch.Tensor, sampling_metadata: SamplingMetadata) -> torch.Tensor:
        # print("t3/compute_logits/hidden_states", hidden_states.shape, hidden_states.dtype)
        # print("t3/compute_logits/sampling_metadata", sampling_metadata)

        # Split the hidden state vector into the three parts
        cond_hidden_states, uncond_hidden_states = hidden_states.split([self.dim, self.dim], dim=1)
        # print("t3/compute_logits/normal_hidden_states", normal_hidden_states.shape, normal_hidden_states.dtype)
        # print("t3/compute_logits/cfg_hidden_states", cfg_hidden_states.shape, cfg_hidden_states.dtype)

        cond_logits = self.logits_processor(self.speech_head, cond_hidden_states, sampling_metadata)
        uncond_logits = self.logits_processor(self.speech_head, uncond_hidden_states, sampling_metadata)

        logits = cond_logits + self.cfg_scale * (cond_logits - uncond_logits)

        # print("t3/compute_logits/logit with the highest probability (cond, uncond, post-cfg):", cond_logits.argmax(), uncond_logits.argmax(), logits.argmax())

        # HACK: Offset the logits so the resulting speech token is +SPEECH_TOKEN_OFFSET from the normal speech tokens.
        #       We'll do this by adding SPEECH_TOKEN_OFFSET fake dimensions to the left of the logits.
        #       This is a hack to help us unbatch batched inputs.
        logits = torch.cat([torch.zeros(logits.shape[0], SPEECH_TOKEN_OFFSET).to(logits.device), logits], dim=1)
        return logits


    def forward(
        self,
        input_ids: Optional[torch.Tensor],  # Almost always None
        positions: torch.Tensor,  # Position IDs since start of the context (i.e. since the first conditional token)
        intermediate_tensors: Optional[IntermediateTensors],  # Almost always None
        inputs_embeds: Optional[torch.Tensor] = None,  # The actual inputs to the model
        **kwargs: object,
    ) -> torch.Tensor:
        # print("t3/inputs_embeds", inputs_embeds.shape, inputs_embeds.dtype)
        # print("t3/positions", positions.shape, positions.dtype)

        # These are usually NULL:
        # print("t3/intermediate_tensors", intermediate_tensors)
        # print("t3/input_ids", input_ids)
        # print("t3/kwargs", kwargs)

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings(input_ids, [])

        # Split the inputs_embeds into the three parts
        cond_embeds, uncond_embeds = inputs_embeds.split([self.dim, self.dim], dim=1)
        # print("t3/embeds", embeds.shape, embeds.dtype)
        # print("t3/cfg_embeds", cfg_embeds.shape, cfg_embeds.dtype)

        # TODO: Apply speech positional embeddings here

        hidden_states = self.tfmr(
            input_ids=None,
            positions=torch.cat([positions, positions], dim=0),
            intermediate_tensors=None,
            inputs_embeds=torch.cat([cond_embeds, uncond_embeds], dim=0)
        )
        # print("t3/hidden_states", hidden_states.shape, hidden_states.dtype)

        # Reconcatenate the hidden states into the master tensor
        hidden_state_1, hidden_state_2 = hidden_states.split([len(cond_embeds), len(uncond_embeds)], dim=0)
        return torch.cat([hidden_state_1, hidden_state_2], dim=1)

    def get_language_model(self) -> torch.nn.Module:
        return self.tfmr