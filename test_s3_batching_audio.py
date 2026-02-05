#!/usr/bin/env python3
"""
Generate test audio files to verify S3 batching produces correct output.
"""

import torch
import torchaudio
from pathlib import Path
from chatterbox_vllm.tts import ChatterboxTTS

def main():
    print("Initializing ChatterboxTTS...")
    tts = ChatterboxTTS.from_pretrained_multilingual(
        gpu_memory_utilization=0.3,
        max_model_len=1000,
    )
    
    # Output to current directory
    output_dir = Path(".")
    
    # Test prompts (Hebrew)
    prompts = [
        "שלום, זהו מבחן של יצירת S3 באצווה.",
        "הטכנולוגיה משתפרת מיום ליום.",
        "הנה שני התורים המוקדמים ביותר שמצאתי עבורכם לאורתופד. התור הראשון, לדוקטור לוי אלמוג, ביום שני ה26 בינואר בשעה אחת וחמש דקות בפתח תקווה. התור השני, לרופא, ביום ראשון ה-11 בינואר בשעה שמונה וחמישים ושש דקות בבוקר, ברמת גן הרצל 34 קומה 4. איזה תור תרצו, הראשון או השני?",
        "בודקים אם איכות השמע נשמרת עם עיבוד באצווה.",
    ]
    
    print(f"\nGenerating audio for {len(prompts)} Hebrew prompts with batching...")
    print("Using female1.wav as cloned voice")
    print("=" * 70)
    
    # Generate with batching using cloned voice
    outputs = tts.generate(prompts, language_id="he", audio_prompt_path="female1.wav")
    
    print(f"\nGenerated {len(outputs)} audio files")
    print(f"Output type: {type(outputs[0])}")
    
    # Save each output
    for i, (output, prompt) in enumerate(zip(outputs, prompts), 1):
        output_path = output_dir / f"batched_{i}.wav"
        
        # Handle different output types
        if isinstance(output, torch.Tensor):
            waveform = output
            sample_rate = 24000  # Default S3Gen sample rate
        elif hasattr(output, 'waveform'):
            waveform = output.waveform
            sample_rate = output.sample_rate
        else:
            print(f"Unknown output type for prompt {i}: {type(output)}")
            continue
        
        # Ensure waveform is 2D (channels, samples)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        elif waveform.dim() == 3:
            waveform = waveform.squeeze(0)
        
        # Save to file
        torchaudio.save(str(output_path), waveform.cpu(), sample_rate)
        
        duration = waveform.shape[-1] / sample_rate
        print(f"✓ Saved: {output_path.name}")
        print(f"  Prompt: {prompt[:50]}...")
        print(f"  Duration: {duration:.2f}s, Shape: {waveform.shape}, SR: {sample_rate}Hz")
    
    print("\n" + "=" * 70)
    print(f"✓ All audio files saved to: {output_dir.absolute()}")
    print("  You can now listen to these files to verify quality")
    print("=" * 70)

if __name__ == "__main__":
    main()
