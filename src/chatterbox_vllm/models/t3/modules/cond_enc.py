from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn, Tensor

from .perceiver import Perceiver
from .t3_config import T3Config


@dataclass
class T3Cond(nn.Module):
    """
    Dataclass container for most / all conditioning info.
    TODO: serialization methods aren't used, keeping them around for convenience
    """

    speaker_emb: Tensor = torch.ones(0)
    clap_emb: Tensor = torch.ones(0)
    cond_prompt_speech_tokens: Tensor = torch.ones(0)
    cond_prompt_speech_emb: Tensor = torch.ones(0)
    emotion_adv: Tensor = 0.5 * torch.ones(1, 1)

    def __init__(self, **kwargs):
        super().__init__()
        for k, v in kwargs.items():
            if v is None:
                v = torch.ones(0)
            elif k == 'cond_prompt_speech_emb' and len(v.shape) == 3:
                # Remove batch dimension
                v = v[0]
            elif k == 'emotion_adv' and len(v.shape) == 3:
                # Remove batch dimension
                v = v[0]
            setattr(self, k, v)

    def save(self, fpath):
        torch.save(self.__dict__, fpath)

    @staticmethod
    def load(fpath, map_location="cpu"):
        kwargs = torch.load(fpath, map_location=map_location, weights_only=True)
        return T3Cond(**kwargs)
    
    def to(self, device):
        self.speaker_emb = self.speaker_emb.to(device)
        self.clap_emb = self.clap_emb.to(device)
        self.cond_prompt_speech_tokens = self.cond_prompt_speech_tokens.to(device)
        self.cond_prompt_speech_emb = self.cond_prompt_speech_emb.to(device)
        self.emotion_adv = self.emotion_adv.to(device)
        return self
    
    def __repr__(self):
        return f"T3Cond(speaker_emb={self.speaker_emb.shape}/{self.speaker_emb.dtype}, clap_emb={self.clap_emb.shape}/{self.clap_emb.dtype}, cond_prompt_speech_tokens={self.cond_prompt_speech_tokens.shape}/{self.cond_prompt_speech_tokens.dtype}, cond_prompt_speech_emb={self.cond_prompt_speech_emb.shape}/{self.cond_prompt_speech_emb.dtype}, emotion_adv={self.emotion_adv.shape}/{self.emotion_adv.dtype})"


class T3CondEnc(nn.Module):
    """
    Handle all non-text conditioning, like speaker embeddings / prompts, CLAP, emotion, etc.
    """

    def __init__(self, hp: T3Config):
        super().__init__()
        self.hp = hp
        if hp.encoder_type == "voice_encoder":
            self.spkr_enc = nn.Linear(hp.speaker_embed_size, hp.n_channels)
        else:
            raise NotImplementedError(str(hp.encoder_type))
        
        # emotion adv
        self.emotion_adv_fc = None
        if hp.emotion_adv:
            self.emotion_adv_fc = nn.Linear(1, hp.n_channels, bias=False)

        # perceiver resampler
        self.perceiver = None
        if hp.use_perceiver_resampler:
            self.perceiver = Perceiver()

    def forward(self, cond: T3Cond):
        # Validate
        assert (cond.cond_prompt_speech_tokens.shape == (0,)) == (cond.cond_prompt_speech_emb.shape == (0,)), \
            "no embeddings for cond_prompt_speech_tokens"

        # print("T3CondEnc/cond.speaker_emb", cond.speaker_emb.shape, cond.speaker_emb.dtype)
        # print("T3CondEnc/self.spkr_enc", self.spkr_enc.weight.shape, self.spkr_enc.weight.dtype)
        cond_spkr = self.spkr_enc(cond.speaker_emb.view(self.hp.speaker_embed_size))
        cond_spkr = cond_spkr.unsqueeze(0)  # (dim,) -> (1, dim) - add sequence dimension
        
        empty = torch.zeros(0, cond_spkr.shape[-1], device=cond_spkr.device, dtype=cond_spkr.dtype)  # (0, dim)
        # print("T3CondEnc/cond_spkr", cond_spkr.shape, cond_spkr.dtype)

        # TODO CLAP
        assert cond.clap_emb.shape == (0,), "clap_embed not implemented"
        cond_clap = empty  # (0, dim)

        # Cond prompt
        cond_prompt_speech_emb = cond.cond_prompt_speech_emb
        # print("T3CondEnc/cond_prompt_speech_emb 1", cond_prompt_speech_emb.shape, cond_prompt_speech_emb.dtype)
        if cond_prompt_speech_emb.shape == (0,):
            cond_prompt_speech_emb = empty  # (0, dim)
        elif self.hp.use_perceiver_resampler:
            cond_prompt_speech_emb = self.perceiver(cond_prompt_speech_emb)

        # Emotion Adv: must provide a value if this model uses emotion conditioning
        cond_emotion_adv = empty  # (0, dim)
        if self.hp.emotion_adv:
            assert cond.emotion_adv.shape != (0,)
            cond_emotion_adv = self.emotion_adv_fc(cond.emotion_adv)

        # print("T3CondEnc/cond_spkr", cond_spkr.shape, cond_spkr.dtype)
        # print("T3CondEnc/cond_clap", cond_clap.shape, cond_clap.dtype)
        # print("T3CondEnc/cond_prompt_speech_emb", cond_prompt_speech_emb.shape, cond_prompt_speech_emb.dtype)
        # print("T3CondEnc/cond_emotion_adv", cond_emotion_adv.shape, cond_emotion_adv.dtype)

        # Concat and return
        cond_embeds = torch.cat((
            cond_spkr,
            cond_clap,
            cond_prompt_speech_emb,
            cond_emotion_adv,
        ), dim=0)
        return cond_embeds
