# Chatterbox vLLM - Setup and Usage Guide

This guide describes how to set up the Chatterbox vLLM Text-to-Speech system on a new machine. It includes the recent performance optimizations for batching, mixed-precision (FP16), and streaming.

## 1. Prerequisites

Ensure your machine meets the following requirements:
*   **OS**: Linux (Ubuntu 20.04/22.04 recommended)
*   **GPU**: NVIDIA GPU with CUDA support (VRAM >= 24GB recommended for high throughput)
*   **Drivers**: NVIDIA Drivers (compatible with CUDA 11.8 or 12.1)
*   **Software**: 
    *   git
    *   Python 3.10 or higher
    *   Conda (optional but recommended for environment management)

## 2. Repository Setup

Clone the repository and switch to the optimization branch (if not merged to main yet):

```bash
git clone https://github.com/Eyalcohenx/chatterbox-vllm.git
cd chatterbox-vllm
# Ensure you are on the feature branch if required (currently feature/s3-batching-and-streaming)
git checkout feature/s3-batching-and-streaming
```

## 3. Environment Installation

It is highly recommended to use a virtual environment.

### Using Conda
```bash
conda create -n chatterbox python=3.10 -y
conda activate chatterbox
```

### Install Dependencies
Install the package in editable mode. This will automatically install dependencies listed in `pyproject.toml` (vllm, torch, etc.).

```bash
pip install -e .
```

*Note: If you encounter issues (e.g., with `vllm` or `flash-attn`), ensure you have the correct CUDA toolkit installed or use the pre-built wheels provided by vLLM documentation.*

## 4. Model Setup

The system is designed to automatically download the required models from Hugging Face on the first run. 

### Authentication (Important)
If the model repositories are private (e.g., specific chatterbox checkpoints), you must authenticate with Hugging Face:

```bash
pip install huggingface_hub
huggingface-cli login
# Enter your User Access Token when prompted (Read permissions are sufficient)
```

The system will automatically manage:
*   Caching models in `~/.cache/huggingface`
*   Creating a local symlink directory `t3-model-multilingual` for vLLM compatibility (this is handled automatically by the code).

## 5. Verification & Testing

We have prepared specific scripts to verify the installation and new optimizations.

### 5.1 Basic Audio Generation Test
Run this to generate a single sample and ensure basic functionality works (FP16 enabled).

```bash
python test_fp16_audio.py
```
*   **Success Criteria**: A file named `test_fp16_output.wav` is created and sounds correct.

### 5.2 Batch Processing Test
Verify that the S3Gen batching optimizations act correctly (testing batch size 4).

```bash
python test_s3_batching_audio.py
```
*   **Success Criteria**: 4 files (`batched_1.wav` to `batched_4.wav`) are generated. The script will report execution time.

### 5.3 Benchmarking (Performance)
To test the full throughput of the system with valid workloads:

```bash
python benchmark_multiple_batches.py
```
*   **Expected Output**: The script will run warm-up batches and then measure Prompts Per Second (PPS). You should see `s3gen_use_fp16=True` in the logs.

### 5.4 Streaming Concurrency
To verify that real-time streaming works correctly with concurrent requests:

```bash
python test_streaming_interleaved.py
```

## 6. Usage in Application

To use the optimized `ChatterboxTTS` in your own code:

```python
from chatterbox_vllm import ChatterboxTTS

# Initialize (downloads model automatically)
tts = ChatterboxTTS.from_pretrained_multilingual()

# Generate Audio (Non-stream)
audio = tts.predict(
    text="Hello world, this is a test.",
    voice="en_male_1",
    language="en",
    s3gen_use_fp16=True  # Optimizations are ON by default now, but can be explicit
)

# Stream Audio
for chunk in tts.predict_stream(
    text="This is a long sentence being streamed.",
    voice="en_female_1",
    language="en"
):
    # Process audio chunk (numpy array)
    pass
```

## 7. Troubleshooting

*   **`OSError: [Errno 2] No such file or directory: '.../t3-model-multilingual/model.safetensors'`**: 
    The latest code fixes this automatically. If you see this, ensure `src/chatterbox_vllm/tts.py` has the directory creation fix (Lines ~180-200).
    
*   **CUDA OOM**: 
    If you run out of memory, try reducing `max_model_len` inside `tts.py` or check if other processes are using the GPU. The default `gpu_memory_utilization` for vLLM is set to 0.6 to leave room for the audio model (S3Gen).

*   **Slow performance**:
    Ensure `s3gen_use_fp16=True`. Check logs from `benchmark_multiple_batches.py` to confirm FP16 mode is active.
