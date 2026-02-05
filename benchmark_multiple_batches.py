#!/usr/bin/env python3
"""
Benchmark multiple batches with varied Hebrew prompts to test sustained performance.
"""

import time
import random
import torch
from pathlib import Path
from chatterbox_vllm.tts import ChatterboxTTS

# Data for generating varied prompts
DOCTORS = [
    "דוקטור לוי אלמוג", "דוקטור שרה כהן", "דוקטור מיכאל דוד",
    "דוקטורה רחל ברק", "דוקטור יוסי מזרחי", "דוקטורה תמר גולן",
    "דוקטור אבי שפירא", "דוקטורה נעמי רוזן", "דוקטור דני לב",
    "דוקטורה ליאת אברהם"
]

SPECIALTIES = [
    "אורתופד", "רופא עור", "אורולוג", "אולטראסאונד", 
    "רופא אף אוזן גרון", "קרדיולוג", "גסטרואנטרולוג",
    "רופא משפחה", "נוירולוג", "רופא ילדים"
]

CITIES = [
    "פתח תקווה", "תל אביב", "רמת גן", "בני ברק", "חולון",
    "רחובות", "נתניה", "הרצליה", "רמת השרון", "כפר סבא"
]

STREETS = [
    "הרצל", "ויצמן", "רוטשילד", "דיזנגוף", "בן גוריון",
    "ז'בוטינסקי", "בן יהודה", "המלך ג'ורג'", "שינקין", "אלנבי"
]

DAYS = [
    "ראשון", "שני", "שלישי", "רביעי", "חמישי", "שישי"
]

MONTHS = [
    "ינואר", "פברואר", "מרץ", "אפריל", "מאי", "יוני",
    "יולי", "אוגוסט", "ספטמבר", "אוקטובר", "נובמבר", "דצמבר"
]

def generate_varied_prompt():
    """Generate a random appointment prompt with varied details."""
    specialty = random.choice(SPECIALTIES)
    
    # First appointment
    doctor1 = random.choice(DOCTORS)
    day1 = random.choice(DAYS)
    date1 = random.randint(1, 28)
    month1 = random.choice(MONTHS)
    hour1 = random.choice(["שמונה", "תשע", "עשר", "אחת עשרה", "שתיים עשרה", "אחת", "שתיים", "שלוש", "ארבע"])
    minute1 = random.choice(["", "וחמש", "ועשר", "ורבע", "ועשרים", "וחצי"])
    city1 = random.choice(CITIES)
    
    # Second appointment  
    day2 = random.choice(DAYS)
    date2 = random.randint(1, 28)
    month2 = random.choice(MONTHS)
    hour2 = random.choice(["שמונה", "תשע", "עשר", "אחת עשרה", "שתיים עשרה", "אחת", "שתיים", "שלוש", "ארבע"])
    minute2 = random.choice(["", "וחמש", "ועשר", "ורבע", "ועשרים", "וחצי"])
    street2 = random.choice(STREETS)
    number2 = random.randint(1, 150)
    floor2 = random.randint(1, 8)
    city2 = random.choice(CITIES)
    
    time_of_day = random.choice(["בבוקר", "בצהריים", "אחר הצהריים"])
    
    minute1_text = f" {minute1} דקות" if minute1 else ""
    minute2_text = f" {minute2} דקות" if minute2 else ""
    
    prompt = f"הנה שני התורים המוקדמים ביותר שמצאתי עבורכם ל{specialty}. התור הראשון, ל{doctor1}, ביום {day1} ה{date1} ב{month1} בשעה {hour1}{minute1_text} ב{city1}. התור השני, לרופא, ביום {day2} ה-{date2} ב{month2} בשעה {hour2}{minute2_text} {time_of_day}, ב{city2} {street2} {number2} קומה {floor2}. איזה תור תרצו, הראשון או השני?"
    
    return prompt

def main():
    print("\n" + "="*70)
    print("MULTIPLE BATCHES PERFORMANCE TEST")
    print("="*70)
    
    num_batches = 3
    batch_size = 8
    total_prompts = num_batches * batch_size
    
    print(f"\nTest configuration:")
    print(f"  Number of batches: {num_batches}")
    print(f"  Batch size: {batch_size} prompts")
    print(f"  Total prompts: {total_prompts}")
    print(f"  Diffusion steps: 5 (optimized for speed)")
    
    # Initialize model
    print(f"\nInitializing model with compilation enabled...")
    tts = ChatterboxTTS.from_pretrained_multilingual(
        gpu_memory_utilization=0.4,
        max_model_len=1500,
        compile=True,
    )
    
    # Warmup
    print(f"\nWarming up...")
    warmup_prompt = generate_varied_prompt()
    _ = tts.generate([warmup_prompt], language_id="he", diffusion_steps=5)
    torch.cuda.synchronize()
    
    print("\n" + "="*70)
    print("RUNNING BATCHED INFERENCE")
    print("="*70)
    
    batch_times = []
    all_audio_durations = []
    
    overall_start = time.time()
    
    for batch_num in range(1, num_batches + 1):
        # Generate varied prompts for this batch
        prompts = [generate_varied_prompt() for _ in range(batch_size)]
        
        print(f"\nBatch {batch_num}/{num_batches}:")
        print(f"  Sample prompt: {prompts[0][:80]}...")
        
        torch.cuda.synchronize()
        batch_start = time.time()
        
        outputs = tts.generate(prompts, language_id="he", diffusion_steps=5)
        
        torch.cuda.synchronize()
        batch_time = time.time() - batch_start
        batch_times.append(batch_time)
        
        # Calculate audio durations
        for output in outputs:
            audio_duration = output.shape[-1] / tts.sr
            all_audio_durations.append(audio_duration)
        
        per_prompt = batch_time / batch_size
        throughput = batch_size / batch_time
        
        print(f"  Time: {batch_time:.3f}s ({per_prompt:.3f}s per prompt)")
        print(f"  Throughput: {throughput:.2f} prompts/sec")
        
        # Clear cache between batches
        torch.cuda.empty_cache()
    
    overall_time = time.time() - overall_start
    
    # Calculate statistics
    avg_batch_time = sum(batch_times) / len(batch_times)
    min_batch_time = min(batch_times)
    max_batch_time = max(batch_times)
    std_batch_time = (sum((t - avg_batch_time) ** 2 for t in batch_times) / len(batch_times)) ** 0.5
    
    avg_audio_duration = sum(all_audio_durations) / len(all_audio_durations)
    total_audio_duration = sum(all_audio_durations)
    
    overall_throughput = total_prompts / overall_time
    avg_per_prompt = overall_time / total_prompts
    
    print("\n" + "="*70)
    print("PERFORMANCE SUMMARY")
    print("="*70)
    
    print(f"\nBatch Statistics ({num_batches} batches of {batch_size} prompts):")
    print(f"  Average batch time: {avg_batch_time:.3f}s")
    print(f"  Min batch time: {min_batch_time:.3f}s")
    print(f"  Max batch time: {max_batch_time:.3f}s")
    print(f"  Std deviation: {std_batch_time:.3f}s")
    print(f"  Consistency: {((1 - std_batch_time/avg_batch_time) * 100):.1f}%")
    
    print(f"\nOverall Performance:")
    print(f"  Total time: {overall_time:.2f}s")
    print(f"  Total prompts: {total_prompts}")
    print(f"  Average per prompt: {avg_per_prompt:.3f}s")
    print(f"  Overall throughput: {overall_throughput:.2f} prompts/sec")
    
    print(f"\nAudio Statistics:")
    print(f"  Total audio generated: {total_audio_duration:.1f}s")
    print(f"  Average audio duration: {avg_audio_duration:.2f}s")
    print(f"  Real-time factor: {avg_per_prompt / avg_audio_duration:.2f}x")
    
    print(f"\nProduction Estimates:")
    print(f"  Prompts per minute: {overall_throughput * 60:.1f}")
    print(f"  Prompts per hour: {overall_throughput * 3600:.0f}")
    print(f"  Audio minutes per hour: {(overall_throughput * 3600 * avg_audio_duration) / 60:.1f}")
    
    # Performance rating
    print(f"\n" + "="*70)
    if overall_throughput >= 0.4:
        print(f"✅ EXCELLENT: {overall_throughput:.2f} prompts/sec sustained throughput!")
        print(f"   Ready for production with batch size {batch_size}")
    elif overall_throughput >= 0.3:
        print(f"✅ GOOD: {overall_throughput:.2f} prompts/sec sustained throughput")
        print(f"   Suitable for most production workloads")
    else:
        print(f"⚠️  MODERATE: {overall_throughput:.2f} prompts/sec sustained throughput")
        print(f"   Consider optimizations for high-volume production")
    
    print(f"="*70 + "\n")
    
    # Cleanup
    tts.shutdown()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
