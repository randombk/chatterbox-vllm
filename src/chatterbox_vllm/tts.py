from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union, Tuple, Any, Generator
import time
import logging

from vllm import LLM, SamplingParams
from functools import lru_cache

import librosa
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from chatterbox_vllm.models.t3.modules.t3_config import T3Config

from .models.s3tokenizer import S3_SR, drop_invalid_tokens
from .models.s3gen import S3GEN_SR, S3Gen
from .models.voice_encoder import VoiceEncoder
from .models.t3 import SPEECH_TOKEN_OFFSET
from .models.t3.modules.cond_enc import T3Cond, T3CondEnc
from .models.t3.modules.learned_pos_emb import LearnedPositionEmbeddings
from .text_utils import punc_norm, SUPPORTED_LANGUAGES

REPO_ID = "ResembleAI/chatterbox"
logger = logging.getLogger(__name__)

@dataclass
class Conditionals:
    """
    Conditionals for T3 and S3Gen
    - T3 conditionals:
        - speaker_emb
        - clap_emb
        - cond_prompt_speech_tokens
        - cond_prompt_speech_emb
        - emotion_adv
    - S3Gen conditionals:
        - prompt_token
        - prompt_token_len
        - prompt_feat
        - prompt_feat_len
        - embedding
    """
    t3: T3Cond
    gen: dict

    def to(self, device):
        self.t3 = self.t3.to(device=device)
        for k, v in self.gen.items():
            if torch.is_tensor(v):
                self.gen[k] = v.to(device=device)
        return self

    @classmethod
    def load(cls, fpath):
        kwargs = torch.load(fpath, weights_only=True)
        return cls(T3Cond(**kwargs['t3']), kwargs['gen'])


class ChatterboxTTS:
    ENC_COND_LEN = 6 * S3_SR
    DEC_COND_LEN = 10 * S3GEN_SR

    def __init__(self, target_device: str, max_model_len: int,
                 t3: LLM, t3_config: T3Config, t3_cond_enc: T3CondEnc, 
                 t3_speech_emb: torch.nn.Embedding, t3_speech_pos_emb: LearnedPositionEmbeddings,
                 s3gen: S3Gen, ve: VoiceEncoder, default_conds: Conditionals,
                 variant: str = "english"):
        self.target_device = target_device
        self.max_model_len = max_model_len
        self.t3 = t3
        self.t3_config = t3_config
        self.t3_cond_enc = t3_cond_enc
        self.t3_speech_emb = t3_speech_emb
        self.t3_speech_pos_emb = t3_speech_pos_emb

        self.s3gen = s3gen
        self.ve = ve
        self.default_conds = default_conds
        self.variant = variant

    @property
    def sr(self) -> int:
        """Sample rate of synthesized audio"""
        return S3GEN_SR

    @classmethod
    def from_local(cls, ckpt_dir: str | Path, target_device: str = "cuda", 
                   max_model_len: int = 1000, compile: bool = False,
                   max_batch_size: int = 10,
                   variant: str = "english",

                   # FP16 enabled by default for better performance
                   s3gen_use_fp16: bool = True,
                   **kwargs) -> 'ChatterboxTTS':
        ckpt_dir = Path(ckpt_dir)

        t3_config = T3Config()

        # Load *just* the necessary weights to perform inference with T3CondEnc
        t3_weights = load_file(ckpt_dir / ("t3_cfg.safetensors" if variant == "english" else "t3_mtl23ls_v2.safetensors"))

        t3_enc = T3CondEnc(t3_config)
        t3_enc.load_state_dict({ k.replace('cond_enc.', ''):v for k,v in t3_weights.items() if k.startswith('cond_enc.') })
        t3_enc = t3_enc.to(device=target_device).eval()

        t3_speech_emb = torch.nn.Embedding(t3_config.speech_tokens_dict_size, t3_config.n_channels)
        t3_speech_emb.load_state_dict({ k.replace('speech_emb.', ''):v for k,v in t3_weights.items() if k.startswith('speech_emb.') })
        t3_speech_emb = t3_speech_emb.to(device=target_device).eval()

        t3_speech_pos_emb = LearnedPositionEmbeddings(t3_config.max_speech_tokens + 2 + 2, t3_config.n_channels)
        t3_speech_pos_emb.load_state_dict({ k.replace('speech_pos_emb.', ''):v for k,v in t3_weights.items() if k.startswith('speech_pos_emb.') })
        t3_speech_pos_emb = t3_speech_pos_emb.to(device=target_device).eval()

        total_gpu_memory = torch.cuda.get_device_properties(0).total_memory
        unused_gpu_memory = total_gpu_memory - torch.cuda.memory_allocated()
        
        # Heuristic: rough calculation for what percentage of GPU memory to give to vLLM.
        # Tune this until the 'Maximum concurrency for ___ tokens per request: ___x' is just over 1.
        # This rough heuristic gives 1.55GB for the model weights plus 128KB per token.
        vllm_memory_needed = (1.55*1024*1024*1024) + (max_batch_size * max_model_len * 1024 * 128)
        vllm_memory_percent = vllm_memory_needed / unused_gpu_memory

        print(f"Giving vLLM {vllm_memory_percent * 100:.2f}% of GPU memory ({vllm_memory_needed / 1024**2:.2f} MB)")

        base_vllm_kwargs = {
            "model": "./t3-model" if variant == "english" else "./t3-model-multilingual",
            "task": "generate",
            "tokenizer": "EnTokenizer" if variant == "english" else "MtlTokenizer",
            "tokenizer_mode": "custom",
            "gpu_memory_utilization": vllm_memory_percent,
            "enforce_eager": not compile,
            "max_model_len": max_model_len,
        }

        t3 = LLM(**{**base_vllm_kwargs, **kwargs})

        ve = VoiceEncoder()
        ve.load_state_dict(load_file(ckpt_dir / "ve.safetensors"))
        ve = ve.to(device=target_device).eval()

        s3gen = S3Gen(use_fp16=s3gen_use_fp16)
        s3gen.load_state_dict(load_file(ckpt_dir / "s3gen.safetensors"), strict=False)
        s3gen = s3gen.to(device=target_device).eval()
        if s3gen_use_fp16:
            # Cast to FP16 but keep tokenizer and speaker_encoder in FP32
            s3gen = s3gen.half()
            s3gen.tokenizer = s3gen.tokenizer.float()
            s3gen.speaker_encoder = s3gen.speaker_encoder.float()

        default_conds = Conditionals.load(ckpt_dir / "conds.pt")
        default_conds.to(device=target_device)

        return cls(
            target_device=target_device, max_model_len=max_model_len,
            t3=t3, t3_config=t3_config, t3_cond_enc=t3_enc, t3_speech_emb=t3_speech_emb, t3_speech_pos_emb=t3_speech_pos_emb,
            s3gen=s3gen, ve=ve, default_conds=default_conds,
            variant=variant,
        )

    @classmethod
    def from_pretrained(cls,
                        repo_id: str = REPO_ID,
                        revision: str = "1b475dffa71fb191cb6d5901215eb6f55635a9b6",
                        *args, **kwargs) -> 'ChatterboxTTS':
        for fpath in ["ve.safetensors", "t3_cfg.safetensors", "s3gen.safetensors", "tokenizer.json", "conds.pt"]:
            local_path = hf_hub_download(repo_id=repo_id, filename=fpath, revision=revision)

        # Ensure the symlink in './t3-model/model.safetensors' points to t3_cfg_path
        t3_cfg_path = Path(local_path).parent / "t3_cfg.safetensors"
        model_safetensors_path = Path.cwd() / "t3-model" / "model.safetensors"
        model_safetensors_path.unlink(missing_ok=True)
        model_safetensors_path.symlink_to(t3_cfg_path)

        return cls.from_local(Path(local_path).parent, variant="english", *args, **kwargs)

    @classmethod
    def from_pretrained_multilingual(cls,
                                    repo_id: str = REPO_ID,
                                    revision: str = "05e904af2b5c7f8e482687a9d7336c5c824467d9",
                                    *args, **kwargs) -> 'ChatterboxTTS':
        for fpath in ["ve.safetensors", "t3_mtl23ls_v2.safetensors", "s3gen.safetensors", "grapheme_mtl_merged_expanded_v1.json", "conds.pt", "Cangjie5_TC.json"]:
            local_path = hf_hub_download(repo_id=repo_id, filename=fpath, revision=revision)

        # Ensure the symlink in './t3-model-multilingual/model.safetensors' points to t3_cfg_path
        t3_cfg_path = Path(local_path).parent / "t3_mtl23ls_v2.safetensors"
        model_safetensors_path = Path.cwd() / "t3-model-multilingual" / "model.safetensors"
        
        # Ensure the directory exists
        model_safetensors_path.parent.mkdir(parents=True, exist_ok=True)
        
        model_safetensors_path.unlink(missing_ok=True)
        model_safetensors_path.symlink_to(t3_cfg_path)

        return cls.from_local(Path(local_path).parent, variant="multilingual", *args, **kwargs)
    
    def get_supported_languages(self) -> dict[str, str]:
        """Return dictionary of supported language codes and names."""
        if self.variant == "multilingual":
            return SUPPORTED_LANGUAGES.copy()
        else:
            return { "en": "English" }

    @lru_cache(maxsize=10)
    def get_audio_conditionals(self, wav_fpath: Optional[str] = None) -> Tuple[dict[str, Any], torch.Tensor]:
        if wav_fpath is None:
            s3gen_ref_dict = self.default_conds.gen
            t3_cond_prompt_tokens = self.default_conds.t3.cond_prompt_speech_tokens
            ve_embed = self.default_conds.t3.speaker_emb
        else:
            ## Load reference wav
            s3gen_ref_wav, _sr = librosa.load(wav_fpath, sr=S3GEN_SR)
            ref_16k_wav = librosa.resample(s3gen_ref_wav, orig_sr=S3GEN_SR, target_sr=S3_SR)

            s3gen_ref_wav = s3gen_ref_wav[:self.DEC_COND_LEN]
            s3gen_ref_dict = self.s3gen.embed_ref(s3gen_ref_wav, S3GEN_SR)

            # Speech cond prompt tokens
            s3_tokzr = self.s3gen.tokenizer
            t3_cond_prompt_tokens, _ = s3_tokzr.forward([ref_16k_wav[:self.ENC_COND_LEN]], max_len=self.t3_config.speech_cond_prompt_len)
            t3_cond_prompt_tokens = torch.atleast_2d(t3_cond_prompt_tokens)

            # Voice-encoder speaker embedding
            ve_embed = torch.from_numpy(self.ve.embeds_from_wavs([ref_16k_wav], sample_rate=S3_SR))
            ve_embed = ve_embed.mean(axis=0, keepdim=True)

        cond_prompt_speech_emb = self.t3_speech_emb(t3_cond_prompt_tokens)[0] + self.t3_speech_pos_emb(t3_cond_prompt_tokens)

        cond_emb = self.t3_cond_enc(T3Cond(
            speaker_emb=ve_embed,
            cond_prompt_speech_tokens=t3_cond_prompt_tokens,
            cond_prompt_speech_emb=cond_prompt_speech_emb,
            emotion_adv=0.5 * torch.ones(1, 1)
        ).to(device=self.target_device)).to(device="cpu")  # Conditionals need to be given to VLLM in CPU

        return s3gen_ref_dict, cond_emb

    def update_exaggeration(self, cond_emb: torch.Tensor, exaggeration: float) -> torch.Tensor:
        if exaggeration == 0.5:
            return cond_emb

        new_cond_emb = cond_emb.clone()
        new_cond_emb[-1] = self.t3_cond_enc.emotion_adv_fc(
            (exaggeration * torch.ones(1, 1)).to(self.target_device)
        ).to('cpu')
        return new_cond_emb

    def generate(
        self,
        prompts: Union[str, list[str]],
        audio_prompt_path: Optional[str] = None,
        language_id: Optional[str] = 'en',
        exaggeration: float = 0.5,
        temperature: float = 0.8,
        max_tokens=1000, # Capped at max_model_len

        # From original Chatterbox HF generation args
        top_p=0.8,
        repetition_penalty=2.0,

        # Supports anything in https://docs.vllm.ai/en/v0.9.2/api/vllm/index.html?h=samplingparams#vllm.SamplingParams
        *args, **kwargs,
    ) -> list[any]:
        s3gen_ref, cond_emb = self.get_audio_conditionals(audio_prompt_path)

        return self.generate_with_conds(
            prompts=prompts,
            s3gen_ref=s3gen_ref,
            cond_emb=cond_emb,
            temperature=temperature,
            language_id=language_id,
            exaggeration=exaggeration,
            max_tokens=max_tokens,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            *args, **kwargs
        )

    def generate_with_conds(
        self,
        prompts: Union[str, list[str]],
        s3gen_ref: dict[str, Any],
        cond_emb: torch.Tensor,
        language_id: Optional[str] = 'en',
        temperature: float = 0.8,
        exaggeration: float = 0.5,
        max_tokens=1000, # Capped at max_model_len

        # Number of diffusion steps to use for S3Gen
        # The original Chatterbox uses 10. 5 is often enough for good quality audio, though some quality loss can be detected.
        # This can be as low as 2 or 3 for faster generation, though the audio quality will degrade substantially.
        diffusion_steps: int = 10,
        
        # Batch size for S3 generation (waveform synthesis)
        s3_batch_size: int = 8,

        # From original Chatterbox HF generation args
        top_p=1.0,
        min_p=0.05,
        repetition_penalty=2.0,

        # Supports anything in https://docs.vllm.ai/en/v0.9.2/api/vllm/index.html?h=samplingparams#vllm.SamplingParams
        *args, **kwargs,
    ) -> list[any]:
        if isinstance(prompts, str):
            prompts = [prompts]

        # Validate language_id
        if language_id and language_id.lower() not in self.get_supported_languages():
            supported_langs = ", ".join(self.get_supported_languages().keys())
            raise ValueError(
                f"Unsupported language_id '{language_id}'. "
                f"Supported languages: {supported_langs}"
            )

        cond_emb = self.update_exaggeration(cond_emb, exaggeration)

        # For multilingual, prepend the language token before normalization
        if self.variant == "multilingual":
            # Use angle brackets to avoid conflicts with other start/stop tokens.
            # This will be parsed and replaced in the tokenizer.
            prompts = [f"<{language_id.lower()}>{p}" for p in prompts]

        # Norm and tokenize text
        prompts = ["[START]" + punc_norm(p) + "[STOP]" for p in prompts]
        
        # Pre-batch Hebrew diacritization if applicable
        if self.variant == "multilingual" and language_id.lower() == "he":
            try:
                # Get the tokenizer from vLLM - it's available as a property
                tokenizer = self.t3.llm_engine.tokenizer.tokenizer
                if hasattr(tokenizer, 'prebatch_hebrew_texts'):
                    tokenizer.prebatch_hebrew_texts(prompts, language_id.lower())
            except Exception as e:
                logger.debug(f"Could not prebatch Hebrew diacritization: {e}")
        
        # Batch tokenization for better performance
        tokenizer = self.t3.llm_engine.tokenizer.tokenizer
        print(f"[TOKENIZER-BATCH] Tokenizing {len(prompts)} prompts in batch")
        tokenization_start = time.time()
        
        # Use batch_encode_plus for parallel tokenization
        batch_encoding = tokenizer.batch_encode_plus(
            prompts,
            add_special_tokens=False,
            return_attention_mask=True,
            return_tensors=None  # Get lists, not tensors
        )
        
        tokenization_time = time.time() - tokenization_start
        print(f"[TOKENIZER-BATCH] Completed in {tokenization_time:.3f}s ({len(prompts)/tokenization_time:.1f} prompts/sec)")
        
        # Convert to token_ids format for vLLM
        prompt_token_ids = batch_encoding['input_ids']

        with torch.inference_mode():
            start_time = time.time()
            batch_results = self.t3.generate(
                [
                    {
                        "prompt_token_ids": token_ids,
                        "multi_modal_data": {
                            "conditionals": [cond_emb],
                        },
                    }
                    for token_ids in prompt_token_ids
                ],
                sampling_params=SamplingParams(
                    temperature=temperature,

                    stop_token_ids=[self.t3_config.stop_speech_token + SPEECH_TOKEN_OFFSET],
                    max_tokens=min(max_tokens, self.max_model_len),
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,

                    *args, **kwargs,
                )
            )
            t3_gen_time = time.time() - start_time
            print(f"[T3] Speech Token Generation time: {t3_gen_time:.2f}s")

            # run torch gc
            torch.cuda.empty_cache()

            start_time = time.time()
            results = []
            
            # Collect all speech tokens first
            all_speech_tokens = []
            for i, batch_result in enumerate(batch_results):
                for output in batch_result.outputs:
                    speech_tokens = torch.tensor([token - SPEECH_TOKEN_OFFSET for token in output.token_ids], device="cuda")
                    speech_tokens = drop_invalid_tokens(speech_tokens)
                    speech_tokens = speech_tokens[speech_tokens < 6561]
                    all_speech_tokens.append(speech_tokens)
            
            # Process waveforms in batches for better GPU utilization
            # s3_batch_size is now an argument
            print(f"[S3] Processing {len(all_speech_tokens)} prompts in batches of {s3_batch_size}")
            
            for batch_idx in range(0, len(all_speech_tokens), s3_batch_size):
                batch_tokens = all_speech_tokens[batch_idx:batch_idx + s3_batch_size]
                
                if batch_idx % (s3_batch_size * 5) == 0:
                    print(f"[S3] Processing batch {batch_idx//s3_batch_size + 1}/{(len(all_speech_tokens) + s3_batch_size - 1)//s3_batch_size}")
                
                # Pad batch for parallel S3 generation
                speech_token_lens = torch.tensor([t.size(0) for t in batch_tokens], dtype=torch.long, device="cuda")
                batch_tokens_padded = pad_sequence(batch_tokens, batch_first=True, padding_value=0)
                
                wavs, _ = self.s3gen.inference(
                    speech_tokens=batch_tokens_padded,
                    speech_token_lens=speech_token_lens,
                    ref_dict=s3gen_ref,
                    n_timesteps=diffusion_steps,
                )
                
                for i, wav in enumerate(wavs):
                    # Unpad audio based on token length
                    # 25 tokens per second, S3GEN_SR (24000) samples per second
                    # => 960 samples per token
                    valid_samples = int(speech_token_lens[i] * 960)
                    
                    # Ensure we don't go out of bounds
                    if valid_samples < wav.shape[-1]:
                        results.append(wav[..., :valid_samples].cpu())
                    else:
                        results.append(wav.cpu())
                
                # Periodic cleanup
                if batch_idx % (s3_batch_size * 2) == 0:
                    torch.cuda.empty_cache()
            
            s3gen_gen_time = time.time() - start_time
            print(f"[S3Gen] Wavform Generation time: {s3gen_gen_time:.2f}s")

            return results
    
    def generate_streaming(
        self,
        prompts: Union[str, list[str]],
        language_id: str = "en",
        audio_prompt_path: Optional[str] = None,
        exaggeration: float = 0.5,
        temperature: float = 0.8,
        max_tokens: int = 1000,
        top_p: float = 0.8,
        repetition_penalty: float = 2.0,
        
        # Streaming-specific parameters
        chunk_size: int = 150,  # Generate audio every N speech tokens
        
        *args, **kwargs,
    ) -> Generator[Tuple[int, torch.Tensor], None, None]:
        """
        Streaming TTS generation that yields audio chunks as they're produced.
        
        Unlike batched processing where all clients wait for the entire batch to complete,
        this processes prompts individually and yields results immediately as each completes.
        For very long prompts, it can also yield intermediate audio chunks.
        
        Args:
            chunk_size: Number of speech tokens to accumulate before generating audio chunk.
                       Smaller = lower latency, larger = better audio quality.
                       Recommended: 100-200 tokens.
        
        Yields:
            Tuple[int, torch.Tensor]: (prompt_index, audio_waveform)
                - prompt_index: Which prompt this audio belongs to (0-indexed)
                - audio_waveform: Complete or partial audio waveform tensor
        
        Example for client-server setup:
            >>> for prompt_idx, audio_chunk in tts.generate_streaming(client_requests, language_id="he"):
            ...     send_to_client(client_id=prompt_idx, audio=audio_chunk)
            ...     # Client starts hearing audio immediately, doesn't wait for other clients
        """
        if isinstance(prompts, str):
            prompts = [prompts]

        # Validate language_id
        if language_id and language_id.lower() not in self.get_supported_languages():
            supported_langs = ", ".join(self.get_supported_languages().keys())
            raise ValueError(
                f"Unsupported language_id '{language_id}'. "
                f"Supported languages: {supported_langs}"
            )

        # Get audio conditionals once
        s3gen_ref, cond_emb = self.get_audio_conditionals(audio_prompt_path)
        cond_emb = self.update_exaggeration(cond_emb, exaggeration)

        # For multilingual, prepend the language token
        if self.variant == "multilingual":
            prompts = [f"<{language_id.lower()}>{p}" for p in prompts]

        # Norm and tokenize
        prompts = ["[START]" + punc_norm(p) + "[STOP]" for p in prompts]
        
        # Pre-batch Hebrew diacritization if applicable (helps with preprocessing)
        if self.variant == "multilingual" and language_id.lower() == "he":
            try:
                tokenizer = self.t3.llm_engine.tokenizer.tokenizer
                if hasattr(tokenizer, 'prebatch_hebrew_texts'):
                    tokenizer.prebatch_hebrew_texts(prompts, language_id.lower())
            except Exception as e:
                logger.debug(f"Could not prebatch Hebrew diacritization: {e}")
        
        # Tokenize all prompts upfront
        tokenizer = self.t3.llm_engine.tokenizer.tokenizer
        logger.info(f"[STREAMING] Tokenizing {len(prompts)} prompts")
        batch_encoding = tokenizer.batch_encode_plus(
            prompts,
            add_special_tokens=False,
            return_attention_mask=True,
            return_tensors=None
        )
        prompt_token_ids = batch_encoding['input_ids']

        # Prepare sampling params
        sampling_params = SamplingParams(
            temperature=temperature,
            stop_token_ids=[self.t3_config.stop_speech_token + SPEECH_TOKEN_OFFSET],
            max_tokens=min(max_tokens, self.max_model_len),
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            *args, **kwargs,
        )

        t3_start = time.time()

        with torch.inference_mode():
            logger.info(f"[STREAMING] Starting INTERLEAVED streaming generation for {len(prompts)} prompts")
            
            # Helper to manage request state
            class RequestState:
                def __init__(self, idx, token_ids):
                    self.idx = idx
                    self.token_ids = token_ids
                    self.tokens = []  # Accumulated generated speech tokens
                    self.processed_tokens = 0  # Number of tokens already sent to S3
                    self.finished = False

            requests_map = {} # request_id -> RequestState
            
            # 1. Add requests to engine
            for i, token_ids in enumerate(prompt_token_ids):
                req_id = f"req_{i}_{time.time()}"
                requests_map[req_id] = RequestState(i, token_ids)
                
                # Construct PromptType dictionary
                prompt_data = {
                    "prompt_token_ids": token_ids,
                    "multi_modal_data": {"conditionals": [cond_emb]}
                }
                
                self.t3.llm_engine.add_request(
                    request_id=req_id,
                    prompt=prompt_data,
                    params=sampling_params
                )

            # 2. Loop until all requests are done
            last_chunk_time = time.time()
            
            while self.t3.llm_engine.has_unfinished_requests():
                step_outputs = self.t3.llm_engine.step()
                
                # Check for updates and accumulated chunks
                for output in step_outputs:
                    req_id = output.request_id
                    state = requests_map[req_id]
                    
                    # Update tokens from output
                    # vLLM returns all tokens generated so far for this request
                    current_tokens = output.outputs[0].token_ids
                    
                    # In standard vLLM, token_ids includes everything generated so far
                    # We just need to check if we have enough new tokens since last process
                    
                    # Convert to speech tokens
                    speech_tokens_raw = [t - SPEECH_TOKEN_OFFSET for t in current_tokens]
                    
                    # Store current valid tokens (filtering done later or on the fly?)
                    # For performance, we just treat them as raw indices and filter before S3
                    state.tokens = speech_tokens_raw
                    state.finished = output.finished

                    # Check if ready for chunk processing
                    # Condition: (Unprocessed tokens >= chunk_size) OR (Finished and has remaining tokens)
                    pending_count = len(state.tokens) - state.processed_tokens
                    
                    if pending_count >= chunk_size or (state.finished and pending_count > 10):
                        # Extract chunk
                        chunk_start = state.processed_tokens
                        chunk_end = len(state.tokens)
                        
                        # If not finished, maybe clamp to exactly chunk_size multiple?
                        # Simplified: Just take everything available
                        
                        new_raw_tokens = state.tokens[chunk_start:chunk_end]
                        
                        # Filter invalid tokens
                        valid_tokens = [t for t in new_raw_tokens if 0 <= t < 6561]
                        
                        if len(valid_tokens) > 10: # Minimum useful audio
                            # Generate Audio for this chunk
                            # NOTE: treating chunk as independent generation for now
                            
                            chunk_tensor = torch.tensor([valid_tokens], device="cuda")
                            
                            # Log first chunk time
                            if state.processed_tokens == 0:
                                ttfb = time.time() - t3_start
                                logger.info(f"[STREAMING] Request {state.idx} TTFB: {ttfb:.3f}s")

                            wav, _ = self.s3gen.inference(
                                speech_tokens=chunk_tensor,
                                ref_dict=s3gen_ref,
                                n_timesteps=10, # Maybe lower for intermediate chunks?
                            )
                            
                            yield (state.idx, wav.cpu())
                        
                        state.processed_tokens = chunk_end
            
            # Clean up once at the end
            torch.cuda.empty_cache()
        
    def shutdown(self):
        del self.t3
        torch.cuda.empty_cache()
