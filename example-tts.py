#!/usr/bin/env python3

from typing import List
import os
import torchaudio as ta
# from chatterbox_vllm.tts import ChatterboxTTS
from chatterbox_vllm.models.t3.t3_vllm import T3VllmModel
from vllm import LLM, ModelRegistry

ModelRegistry.register_model("ChatterboxT3", T3VllmModel)

if __name__ == "__main__":
    # model = ChatterboxTTS.from_pretrained()

    # text = "Ezreal and Jinx teamed up with Ahri, Yasuo, and Teemo to take down the enemy's Nexus in an epic late-game pentakill."
    # AUDIO_PROMPT_PATH = "AUDIO_PROMPT.mp3"
    
    # wav = model.generate(text, audio_prompt_path=AUDIO_PROMPT_PATH)
    # ta.save("test.wav", wav, model.sr)

    t3 = LLM(
        model=f"./model-dev",
        task="generate",
        tokenizer="EnTokenizer",
        tokenizer_mode="custom",
        max_model_len=10000,
    )
    print("OK")
    