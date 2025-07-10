#!/usr/bin/env python3

from typing import List
import os
import torchaudio as ta
from chatterbox_vllm.tts import ChatterboxTTS


if __name__ == "__main__":
    model = ChatterboxTTS.from_pretrained(
        gpu_memory_utilization = 0.4,
        max_model_len = 1000,

        # Disable CUDA graphs to reduce startup time for one-off generation.
        enforce_eager = True,
    )

    text = "You are listening to a demo of the Chatterbox TTS model running on VLLM."
    AUDIO_PROMPT_PATH = "docs/audio-sample-01.wav"
    
    wav = model.generate(text, audio_prompt_path=AUDIO_PROMPT_PATH)
    ta.save("test.wav", wav, model.sr)
