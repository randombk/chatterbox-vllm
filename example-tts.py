#!/usr/bin/env python3

from typing import List
import torchaudio as ta
from chatterbox_vllm.tts import ChatterboxTTS


if __name__ == "__main__":
    model = ChatterboxTTS.from_pretrained(
        gpu_memory_utilization = 0.4,
        max_model_len = 1000,

        # Disable CUDA graphs to reduce startup time for one-off generation.
        enforce_eager = True,
    )

    prompts = [
        "You are listening to a demo of the Chatterbox TTS model running on VLLM.",
        "This is a separate prompt to test the batching implementation.",
        "And here is a third prompt. It's a bit longer than the first one, but not by much.",
    ]
    AUDIO_PROMPT_PATH = "docs/audio-sample-02.mp3"
    
    audios = model.generate(prompts, audio_prompt_path=AUDIO_PROMPT_PATH, exaggeration=0.8)
    for i, audio in enumerate(audios):
        ta.save(f"test-{i}.mp3", audio, model.sr)
