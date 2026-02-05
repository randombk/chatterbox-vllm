#!/usr/bin/env python3
"""
Test streaming functionality with concurrent T3/S3 generation.
"""

import time
import torch
from chatterbox_vllm.tts import ChatterboxTTS

def main():
    print("Initializing ChatterboxTTS for Streaming Test...")
    tts = ChatterboxTTS.from_pretrained_multilingual(
        gpu_memory_utilization=0.3,
        max_model_len=1000,
    )
    
    # Long prompt to ensure enough tokens for streaming chunks
    # Hebrew prompt roughly 200 tokens
    prompt = "שלום רב, כאן המרכז הרפואי. אנחנו מתקשרים בקשר לתור שקבעתם לדוקטור כהן ביום שלישי הקרוב. רצינו לעדכן שיש שינוי קל בשעה, והתור הוקדם לשעה ארבע ארבעים וחמש. אם השעה לא נוחה לכם, אנא לחצו אחת כדי לקבוע מועד חדש, או הישארו על הקו לנציג שירות. תודה רבה והמשך יום נעים."
    
    # Create batch of 2 to test concurrent streaming
    prompts = [prompt, prompt]
    
    print("\nStarting Streaming Generation...")
    print(f"Prompts: {len(prompts)}")
    print("=" * 60)
    
    start_time = time.time()
    first_chunk_received = {}
    total_audio_chunks = 0
    
    # Use small chunk size to trigger frequent updates
    CHUNK_SIZE = 50 
    
    for req_idx, audio_chunk in tts.generate_streaming(
        prompts, 
        language_id="he", 
        chunk_size=CHUNK_SIZE
    ):
        elapsed = time.time() - start_time
        
        # Track Time To First Byte (Audio)
        if req_idx not in first_chunk_received:
            first_chunk_received[req_idx] = elapsed
            print(f"⚡ [Req {req_idx}] First Audio Chunk received at {elapsed:.2f}s!")
            
        duration = audio_chunk.shape[-1] / 24000
        print(f"  » [Req {req_idx}] Received chunk: {duration:.2f}s audio (at {elapsed:.2f}s)")
        total_audio_chunks += 1
    
    total_time = time.time() - start_time
    print("=" * 60)
    print(f"Total Streaming Time: {total_time:.2f}s")
    print(f"Total Audio Chunks: {total_audio_chunks}")
    
    for i in range(len(prompts)):
        ttfb = first_chunk_received.get(i, "N/A")
        print(f"Request {i} TTFB: {ttfb:.2f}s")

if __name__ == "__main__":
    main()
