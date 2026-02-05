#!/usr/bin/env python3
"""
Test FP16 Audio Generation.
"""

import time
import torch
import torchaudio
from chatterbox_vllm.tts import ChatterboxTTS

def main():
    print("Initializing ChatterboxTTS (This should default to s3gen_use_fp16=True)...")
    
    # We rely on the updated default in tts.py
    # Using gpu_memory_utilization=0.4 to be safe
    tts = ChatterboxTTS.from_pretrained_multilingual(
        max_batch_size=5,  
        max_model_len=1500,
        gpu_memory_utilization=0.4, 
    )
    
    print("\nModel Initialized.")
    # Check if s3gen is properly cast
    try:
        print(f"S3Gen Flow FP16 flag: {tts.s3gen.flow.fp16}")
        print(f"S3Gen Model Dtype: {tts.s3gen.dtype}")
    except Exception as e:
        print(f"Could not check debug flags: {e}")

    text = "בדיקת שמע אחת שתיים שלוש. אני מדבר בעברית כדי לבדוק את האיכות."
    print(f"\nGenerating audio for: '{text}'")

    start_time = time.time()
    outputs = tts.generate([text], language_id="he", diffusion_steps=5)
    end_time = time.time()
    
    print(f"Generation took: {end_time - start_time:.3f}s")
    
    output_path = "test_fp16_output.wav"
    # output is a list of tensors. Each tensor is (T,) or (1,T)?
    # tts.generate appends output_wavs[0] which is (T,).
    # torchaudio.save expects (C, T) or (T,) maybe? It usually wants (Channels, Time).
    audio_data = outputs[0].unsqueeze(0).cpu().to(torch.float32) 
    # Ensure it's float32 for saving, wav usually requires float
    
    torchaudio.save(output_path, audio_data, tts.sr)
    print(f"\nSaved audio to: {output_path}")

if __name__ == "__main__":
    main()
