#!/usr/bin/env python3

import gc
import torch
import torchaudio as ta
from chatterbox_vllm.tts import ChatterboxTTS

AUDIO_PROMPT_PATH = "docs/audio-sample-01.mp3"
MAX_MODEL_LEN = 1000 # Maximum length of generated audio in tokens

if __name__ == "__main__":
    # Print current GPU memory usage
    print(f"[START] Starting GPU memory usage: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

    total_gpu_memory = torch.cuda.get_device_properties(0).total_memory
    unused_gpu_memory = total_gpu_memory - torch.cuda.memory_allocated()
    
    # Heuristic: rough calculation for what percentage of GPU memory to give to vLLM.
    # Tune this until the 'Maximum concurrency for ___ tokens per request: ___x' is just over 1.
    # This rough heuristic gives 1.53GB for the model weights plus 128KB per token.
    vllm_memory_needed = (1.53*1024*1024*1024) + (MAX_MODEL_LEN * 1024 * 128)
    vllm_memory_percent = vllm_memory_needed / unused_gpu_memory

    print(f"Giving vLLM {vllm_memory_percent * 100:.2f}% of GPU memory ({vllm_memory_needed / 1024**2:.2f} MB)")
    model = ChatterboxTTS.from_pretrained(
        gpu_memory_utilization = vllm_memory_percent,
        max_model_len = MAX_MODEL_LEN,

        # Disable CUDA graphs to reduce startup time for one-off generation.
        enforce_eager = True,
    )

    print(f"[POST-INIT] GPU memory usage after model load: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    
    # Generate audio conditioning
    # The resulting s3gen_ref and cond_emb can be reused for multiple generations, or saved/loaded from disk.
    s3gen_ref, cond_emb = model.get_audio_conditionals(AUDIO_PROMPT_PATH)

    # Generate audio
    cond_emb = model.update_exaggeration(cond_emb, exaggeration=0.5)
    audios = model.generate_with_conds(
        ["You are listening to a demo of the Chatterbox TTS model running on VLLM."],
        s3gen_ref=s3gen_ref,
        cond_emb=cond_emb,
        min_p=0.1,
    )
    print(f"[POST-GEN] GPU memory usage after generating audio: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

    for audio_idx, audio in enumerate(audios):
        ta.save(f"test-{audio_idx}.mp3", audio, model.sr)
