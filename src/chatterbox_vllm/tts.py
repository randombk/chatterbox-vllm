from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from vllm import LLM, SamplingParams
from functools import lru_cache

import librosa
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from chatterbox_vllm.models.t3.modules.t3_config import T3Config

from .models.s3tokenizer import S3_SR, drop_invalid_tokens
from .models.s3gen import S3GEN_SR, S3Gen
from .models.voice_encoder import VoiceEncoder
from .models.t3.modules.cond_enc import T3Cond

REPO_ID = "ResembleAI/chatterbox"

def punc_norm(text: str) -> str:
    """
        Quick cleanup func for punctuation from LLMs or
        containing chars not seen often in the dataset
    """
    if len(text) == 0:
        return "You need to add some text for me to talk."

    # Capitalise first letter
    if text[0].islower():
        text = text[0].upper() + text[1:]

    # Remove multiple space chars
    text = " ".join(text.split())

    # Replace uncommon/llm punc
    punc_to_replace = [
        ("...", ", "),
        ("…", ", "),
        (":", ","),
        (" - ", ", "),
        (";", ", "),
        ("—", "-"),
        ("–", "-"),
        (" ,", ","),
        ("“", "\""),
        ("”", "\""),
        ("‘", "'"),
        ("’", "'"),
    ]
    for old_char_sequence, new_char in punc_to_replace:
        text = text.replace(old_char_sequence, new_char)

    # Add full stop if no ending punc
    text = text.rstrip(" ")
    sentence_enders = {".", "!", "?", "-", ","}
    if not any(text.endswith(p) for p in sentence_enders):
        text += "."

    return text


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

    def __init__(self, t3: LLM, s3gen: S3Gen, ve: VoiceEncoder, default_conds: Conditionals):
        self.sr = S3GEN_SR  # sample rate of synthesized audio
        self.t3 = t3
        self.s3gen = s3gen
        self.ve = ve
        self.default_conds = default_conds
        self.hp = T3Config()

    @classmethod
    def from_local(cls, ckpt_dir: str, **kwargs) -> 'ChatterboxTTS':
        ckpt_dir = Path(ckpt_dir)

        ve = VoiceEncoder()
        ve.load_state_dict(load_file(ckpt_dir / "ve.safetensors"))
        ve.eval()

        s3gen = S3Gen()
        s3gen.load_state_dict(load_file(ckpt_dir / "s3gen.safetensors"), strict=False)
        s3gen.to(device="cuda").eval()

        default_conds = Conditionals.load(ckpt_dir / "conds.pt")
        default_conds.to(device="cuda")

        t3 = LLM(
            model=f"./t3-model",
            task="generate",
            tokenizer="EnTokenizer",
            tokenizer_mode="custom",
            **kwargs,
        )

        return cls(t3, s3gen, ve, default_conds=default_conds)

    @classmethod
    def from_pretrained(cls, *args, **kwargs) -> 'ChatterboxTTS':
        for fpath in ["ve.safetensors", "t3_cfg.safetensors", "s3gen.safetensors", "tokenizer.json", "conds.pt"]:
            local_path = hf_hub_download(repo_id=REPO_ID, filename=fpath, revision="1b475dffa71fb191cb6d5901215eb6f55635a9b6")

        # Ensure the symlink in './t3-model/model.safetensors' points to t3_cfg_path
        t3_cfg_path = Path(local_path).parent / "t3_cfg.safetensors"
        model_safetensors_path = Path.cwd() / "t3-model" / "model.safetensors"
        model_safetensors_path.unlink(missing_ok=True)
        model_safetensors_path.symlink_to(t3_cfg_path)

        return cls.from_local(Path(local_path).parent, *args, **kwargs)

    @lru_cache(maxsize=10)
    def get_audio_conditionals(self, wav_fpath: Optional[str] = None) -> Conditionals:
        if wav_fpath is None:
            return self.default_conds
        
        ## Load reference wav
        s3gen_ref_wav, _sr = librosa.load(wav_fpath, sr=S3GEN_SR)
        ref_16k_wav = librosa.resample(s3gen_ref_wav, orig_sr=S3GEN_SR, target_sr=S3_SR)

        s3gen_ref_wav = s3gen_ref_wav[:self.DEC_COND_LEN]
        s3gen_ref_dict = self.s3gen.embed_ref(s3gen_ref_wav, S3GEN_SR)

        # Speech cond prompt tokens
        s3_tokzr = self.s3gen.tokenizer
        t3_cond_prompt_tokens, _ = s3_tokzr.forward([ref_16k_wav[:self.ENC_COND_LEN]], max_len=self.hp.speech_cond_prompt_len)
        t3_cond_prompt_tokens = torch.atleast_2d(t3_cond_prompt_tokens)

        # Voice-encoder speaker embedding
        ve_embed = torch.from_numpy(self.ve.embeds_from_wavs([ref_16k_wav], sample_rate=S3_SR))
        ve_embed = ve_embed.mean(axis=0, keepdim=True)

        t3_cond = T3Cond(
            speaker_emb=ve_embed,
            cond_prompt_speech_tokens=t3_cond_prompt_tokens,
            emotion_adv=0.5 * torch.ones(1, 1),
        )

        return Conditionals(t3_cond, s3gen_ref_dict)

    def generate(
        self,
        prompts: list[str],
        audio_prompt_path: Optional[str] = None,
        exaggeration: float = 0.5,
        # cfg_weight: float = 0.5,
        temperature: float = 0.8,
    ) -> list[any]:
        conds = self.get_audio_conditionals(audio_prompt_path)
        t3conds = conds.t3
        s3gen_ref = conds.gen

        # Update exaggeration if needed
        if exaggeration != 0.5:
            t3conds = T3Cond(
                speaker_emb=t3conds.speaker_emb,
                cond_prompt_speech_tokens=t3conds.cond_prompt_speech_tokens,
                emotion_adv=exaggeration * torch.ones(1, 1),
            )

        # Norm and tokenize text
        with torch.inference_mode():
            batch_results = self.t3.generate(
                [
                    {
                        "prompt": "[START]" + punc_norm(text) + "[STOP]",
                        "multi_modal_data": {
                            "conditionals": [t3conds.to(device="cpu")],
                        },
                    }
                    for text in prompts
                ],
                sampling_params=SamplingParams(
                    temperature=temperature,
                    
                    stop_token_ids=[self.hp.stop_speech_token],
                    max_tokens=1000,

                    # From original Chatterbox HF generation args
                    top_p=0.8,
                    repetition_penalty=2.0,

                    # HACK: I don't see a way to pass custom sampling params to vLLM.
                    # We'll squirrel away CFG Scale in the 'frequency_penalty' field, and extract/reset/set it
                    # when processing the logits.
                    # frequency_penalty=cfg_weight,
                )
            )

            results = []
            for i in range(len(batch_results)):
                for j in range(len(batch_results[i].outputs)):
                    speech_tokens = batch_results[i].outputs[j].token_ids
                    speech_tokens = torch.tensor(speech_tokens)
                    speech_tokens = drop_invalid_tokens(speech_tokens)
                    speech_tokens = speech_tokens[speech_tokens < 6561]

                    wav, _ = self.s3gen.inference(
                        speech_tokens=speech_tokens.to(device="cuda"),
                        ref_dict=s3gen_ref,
                    )
                    results.append(wav.cpu())

            return results
        
