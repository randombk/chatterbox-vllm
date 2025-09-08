#!/usr/bin/env python3

import time
import re
from typing import List

import torch
import torchaudio as ta

from chatterbox_vllm.tts import ChatterboxTTS

AUDIO_PROMPT_PATH = "docs/zh_m1.mp3"
TEXT_PATH = "docs/benchmark-text-zh-1.txt"
MAX_CHUNK_SIZE = 50 # characters

# Process in batches of X chunks at a time. Tune this based on your GPU memory.
#   15 seems to work for 8GB VRAM
#   40 seems to work for 16GB VRAM
#   80 seems to work for 24GB VRAM
# You may need to adjust the batch size based on your GPU memory.
BATCH_SIZE = 15

END_OF_SENTENCE_SEQ = '。'

# Given a line of text, split it into chunks of at most MAX_CHUNK_SIZE characters
# at sentence boundaries, forming roughly equal-sized chunks
def split_text_by_sentence(text: str) -> List[str]:
    sentences = text.split(END_OF_SENTENCE_SEQ)
    n_chunks_needed = len(text) // MAX_CHUNK_SIZE + 1
    approx_chunk_size = len(text) // n_chunks_needed

    chunks = []
    current_chunk = []
    current_length = 0
    for sentence in sentences:
        # Remove excess whitespace like consecutive spaces, newlines, etc.
        sentence = " ".join(sentence.split())
        sentence = sentence.strip()

        if current_length + len(sentence) > approx_chunk_size:
            chunks.append(END_OF_SENTENCE_SEQ.join(current_chunk))
            current_chunk = [sentence]
            current_length = len(sentence)
        else:
            current_chunk.append(sentence)
            current_length += len(sentence)

    if current_chunk:
        chunks.append(END_OF_SENTENCE_SEQ.join(current_chunk))

    # If chunks end with a-z0-9, add a period to the end
    chunks = [chunk + END_OF_SENTENCE_SEQ if re.match(r"[a-zA-Z0-9]", chunk[-1]) else chunk for chunk in chunks if len(chunk) > 0]

    return chunks


if __name__ == "__main__":
    with open(TEXT_PATH, "r") as f:
        text = f.read()
    
    # Remove lines starting with #
    text = "\n".join([line for line in text.split("\n") if not line.startswith("#")])

    # Chunk text by newlines
    text = [i.strip() for i in text.split("\n") if len(i.strip()) > 0]

    # Split text into chunks
    text = [split_text_by_sentence(line) for line in text]

    # Flatten list
    text = [item for sublist in text for item in sublist]

    print(f"[BENCHMARK] Text chunked into {len(text)} chunks")
    
    start_time = time.time()
    model = ChatterboxTTS.from_pretrained_multilingual(
        max_batch_size = BATCH_SIZE,
        max_model_len = MAX_CHUNK_SIZE * 20, # Rough heuristic for how many speech tokens per text token
    )
    model_load_time = time.time()
    print(f"[BENCHMARK] Model loaded in {model_load_time - start_time} seconds")

    audios = model.generate(
        text,
        audio_prompt_path=AUDIO_PROMPT_PATH,
        language_id="zh",
        exaggeration=0.5,

        # Supports anything in https://docs.vllm.ai/en/v0.9.2/api/vllm/index.html?h=samplingparams#vllm.SamplingParams
        min_p=0.1,
        top_p=0.8,
    )
    generation_time = time.time()
    print(f"[BENCHMARK] Generation completed in {generation_time - model_load_time} seconds")

    # Stitch audio chunks together
    full_audio = torch.cat(audios, dim=-1)
    ta.save(f"benchmark-zh.mp3", full_audio, model.sr)
    print(f"[BENCHMARK] Audio saved to benchmark-zh.mp3")
    print(f"[BENCHMARK] Total time: {time.time() - start_time} seconds")

    model.shutdown()
