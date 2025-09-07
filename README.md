# Chatterbox TTS on vLLM

This is a port of https://github.com/resemble-ai/chatterbox to vLLM. Why?

* Improved performance and more efficient use of GPU memory.
  * Early benchmarks show ~4x speedup in generation tokens/s without batching, and over 10x with batching. This is a significant improvement over the original Chatterbox implementation, which was bottlenecked by unnecessary CPU-GPU sync/transfers within the HF transformers implementation.
  * More rigorous benchmarking is WIP, but will likely come after batching is fully fleshed out.
* Easier integration with state-of-the-art inference infrastructure.

DISCLAIMER: THIS IS A PERSONAL PROJECT and is not affiliated with my employer or any other corporate entity in any way. The project is based solely on publicly-available information. All opinions are my own and do not necessarily represent the views of my employer.

## Generation Samples

![Sample 1](docs/audio-sample-01.mp3)
<audio controls>
  <source src="docs/audio-sample-01.mp3" type="audio/mp3">
</audio>

![Sample 2](docs/audio-sample-02.mp3)
<audio controls>
  <source src="docs/audio-sample-02.mp3" type="audio/mp3">
</audio>

![Sample 3](docs/audio-sample-03.mp3)
<audio controls>
  <source src="docs/audio-sample-03.mp3" type="audio/mp3">
</audio>


# Project Status: Usable and with Benchmark-Topping Throughput

* ✅ Basic speech cloning with audio and text conditioning.
* ✅ Outputs match the quality of the original Chatterbox implementation.
* ✅ Context Free Guidance (CFG) is implemented.
  * Due to a vLLM limitation, CFG can not be tuned on a per-request basis and can only be configured via the `CHATTERBOX_CFG_SCALE` environment variable.
* ✅ Exaggeration control is implemented.
* ✅ vLLM batching is implemented and produces a significant speedup.
* ℹ️ Project uses vLLM internal APIs and extremely hacky workarounds to get things done.
  * Refactoring to the idiomatic vLLM way of doing things is WIP, but will require some changes to vLLM.
  * Until then, this is a Rube Goldberg machine that will likely only work with vLLM 0.9.2.
  * Follow https://github.com/vllm-project/vllm/issues/21989 for updates.
* ℹ️ Substantial refactoring is needed to further clean up unnecessary workarounds and code paths.
* ℹ️ Server API is not implemented and will likely be out-of-scope for this project.
* ❌ Learned speech positional embeddings are not applied, pending support in vLLM. However, this doesn't seem to be causing a very noticeable degradation in quality.
* ❌ APIs are not yet stable and may change.
* ❌ Benchmarks and performance optimizations are not yet implemented.

# Installation

This project only supports Linux and WSL2 with Nvidia hardware. AMD _may_ work with minor tweaks, but is not tested.

Prerequisites: `git` and [`uv`](https://pypi.org/project/uv/) must be installed

```
git clone https://github.com/randombk/chatterbox-vllm.git
cd chatterbox-vllm
uv venv
source .venv/bin/activate
uv sync
```

The package should automatically download the correct model weights from the Hugging Face Hub.

If you encounter CUDA issues, try resetting the venv and using `uv pip install -e .` instead of `uv sync`.

# Example

[This example](https://github.com/randombk/chatterbox-vllm/blob/master/example-tts.py) can be run with `python example-tts.py` to generate audio samples for three different prompts using three different voices.

```python
import torchaudio as ta
from chatterbox_vllm.tts import ChatterboxTTS


if __name__ == "__main__":
    model = ChatterboxTTS.from_pretrained(
        gpu_memory_utilization = 0.4,
        max_model_len = 1000,

        # Disable CUDA graphs to reduce startup time for one-off generation.
        enforce_eager = True,
    )

    for i, audio_prompt_path in enumerate([None, "docs/audio-sample-01.mp3", "docs/audio-sample-03.mp3"]):
        prompts = [
            "You are listening to a demo of the Chatterbox TTS model running on VLLM.",
            "This is a separate prompt to test the batching implementation.",
            "And here is a third prompt. It's a bit longer than the first one, but not by much.",
        ]
    
        audios = model.generate(prompts, audio_prompt_path=audio_prompt_path, exaggeration=0.8)
        for audio_idx, audio in enumerate(audios):
            ta.save(f"test-{i}-{audio_idx}.mp3", audio, model.sr)
```

# Benchmarks

To run a benchmark, tweak and run `benchmark.py`.
The following results were obtained with batching on a 6.6k-word input (`docs/benchmark-text-1.txt`), generating ~40min of audio.

Notes:
 * I'm not _entirely_ sure what the tokens/s figures from vLLM are showing - the figures probably aren't directly comparable to others, but the results speak for themselves.
 * With vLLM, **the T3 model is no longer the bottleneck**
   * Vast majority of time is now spent on the S3Gen model, which is not ported/portable to vLLM. This currently uses the original reference implementation from the Chatterbox repo, so there's potential for integrating some of the other community optimizations here.
   * This also means the vLLM section of the model never fully ramps to its peak throughput in these benchmarks.
 * Benchmarks are done without CUDA graphs, as that is currently causing correctness issues.
 * There are some issues with my very rudimentary chunking logic, which is causing some occasional artifacts in output quality.

## Run 1: RTX 3090

**Results using v0.1.3**

System Specs:
 * RTX 3090: 24GB VRAM
 * AMD Ryzen 9 7900X @ 5.70GHz
 * 128GB DDR5 4800 MT/s

Settings & Results:
* Input text: `docs/benchmark-text-1.txt` (6.6k words)
* Input audio: `docs/audio-sample-03.mp3`
* Exaggeration: 0.5, CFG: 0.5, Temperature: 0.8
* CUDA graphs disabled, vLLM max memory utilization=0.6
* Generated output length: 39m54s
* Wall time: 1m33s (including model load and application init)
* Generation time (without model startup time): 87s
  * Time spent in T3 Llama token generation: 13.3s
  * Time spent in S3Gen waveform generation: 60.8s

Logs:
```
[BENCHMARK] Text chunked into 154 chunks
Giving vLLM 56.48% of GPU memory (13587.20 MB)
[config.py:1472] Using max model len 1200
[default_loader.py:272] Loading weights took 0.14 seconds
[gpu_model_runner.py:1801] Model loading took 1.0179 GiB and 0.198331 seconds
[gpu_model_runner.py:2238] Encoder cache will be initialized with a budget of 8192 tokens, and profiled with 178 conditionals items of the maximum feature size.
[gpu_worker.py:232] Available KV cache memory: 11.78 GiB
[kv_cache_utils.py:716] GPU KV cache size: 102,880 tokens
[kv_cache_utils.py:720] Maximum concurrency for 1,200 tokens per request: 85.73x
[BENCHMARK] Model loaded in 7.499186038970947 seconds
Adding requests: 100%|██████| 154/154 [00:00<00:00, 1105.84it/s]
Processed prompts: 100%|█████| 154/154 [00:13<00:00, 11.75it/s, est. speed input: 2193.47 toks/s, output: 4577.88 toks/s]
[T3] Speech Token Generation time: 13.25s
[S3Gen] Wavform Generation time: 60.85s
[BENCHMARK] Generation completed in 74.89441227912903 seconds
[BENCHMARK] Audio saved to benchmark.mp3
[BENCHMARK] Total time: 87.40947437286377 seconds

real	1m33.458s
user	2m21.452s
sys	0m2.362s
```


## Run 2: RTX 3060ti

**Results outdated; using v0.1.0**

System Specs:
 * RTX 3060ti: 8GB VRAM
 * Intel i7-7700K @ 4.20GHz
 * 32GB DDR4 2133 MT/s

Settings & Results:
* Input text: `docs/benchmark-text-1.txt` (6.6k words)
* Input audio: `docs/audio-sample-03.mp3`
* Exaggeration: 0.5, CFG: 0.5, Temperature: 0.8
* CUDA graphs disabled, vLLM max memory utilization=0.6
* Generated output length: 40m15s
* Wall time: 4m26s
* Generation time (without model startup time): 238s
  * Time spent in T3 Llama token generation: 36.4s
  * Time spent in S3Gen waveform generation: 201s

Logs:
```
[BENCHMARK] Text chunked into 154 chunks.
INFO [config.py:1472] Using max model len 1200
INFO [default_loader.py:272] Loading weights took 0.39 seconds
INFO [gpu_model_runner.py:1801] Model loading took 1.0107 GiB and 0.497231 seconds
INFO [gpu_model_runner.py:2238] Encoder cache will be initialized with a budget of 8192 tokens, and profiled with 241 conditionals items of the maximum feature size.
INFO [gpu_worker.py:232] Available KV cache memory: 3.07 GiB
INFO [kv_cache_utils.py:716] GPU KV cache size: 26,816 tokens
INFO [kv_cache_utils.py:720] Maximum concurrency for 1,200 tokens per request: 22.35x
Adding requests: 100%|████| 40/40 [00:00<00:00, 947.42it/s]
Processed prompts: 100%|████| 40/40 [00:09<00:00,  4.15it/s, est. speed input: 799.18 toks/s, output: 1654.94 toks/s]
[T3] Speech Token Generation time: 9.68s
[S3Gen] Wavform Generation time: 53.66s
Adding requests: 100%|████| 40/40 [00:00<00:00, 858.75it/s]
Processed prompts: 100%|████| 40/40 [00:08<00:00,  4.69it/s, est. speed input: 938.19 toks/s, output: 1874.97 toks/s]
[T3] Speech Token Generation time: 8.58s
[S3Gen] Wavform Generation time: 53.86s
Adding requests: 100%|████| 40/40 [00:00<00:00, 815.60it/s]
Processed prompts: 100%|████| 40/40 [00:09<00:00,  4.19it/s, est. speed input: 726.62 toks/s, output: 1531.24 toks/s]
[T3] Speech Token Generation time: 9.60s
[S3Gen] Wavform Generation time: 49.89s
Adding requests: 100%|████| 34/34 [00:00<00:00, 938.61it/s]
Processed prompts: 100%|████| 34/34 [00:08<00:00,  3.98it/s, est. speed input: 714.68 toks/s, output: 1439.42 toks/s]
[T3] Speech Token Generation time: 8.59s
[S3Gen] Wavform Generation time: 43.58s
[BENCHMARK] Generation completed in 238.42230987548828 seconds
[BENCHMARK] Audio saved to benchmark.mp3
[BENCHMARK] Total time: 259.1808190345764 seconds

real    4m26.803s
user    4m42.393s
sys     0m4.285s
```


# Chatterbox Architecture

I could not find an official explanation of the Chatterbox architecture, so below is my best explanation based on the codebase. Chatterbox broadly follows the [CosyVoice](https://funaudiollm.github.io/cosyvoice2/) architecture, applying intermediate fusion multimodal conditioning to a 0.5B parameter Llama model.

<div align="center">
  <img src="https://github.com/randombk/chatterbox-vllm/raw/refs/heads/master/docs/chatterbox-architecture.svg" alt="Chatterbox Architecture" width="100%" />
  <p><em>Chatterbox Architecture Diagram</em></p>
</div>

# Implementation Notes

## CFG Implementation Details

vLLM does not support CFG natively, so substantial hacks were needed to make it work. At a high level, we trick vLLM into thinking the model has double the hidden dimension size as it actually does, then splitting and restacking the states to invoke Llama with double the original batch size. This does pose a risk that vLLM will underestimate the memory requirements of the model - more research is needed into whether vLLM's initial profiling pass will capture this nuance.


<div align="center">
  <img src="https://github.com/randombk/chatterbox-vllm/raw/refs/heads/master/docs/vllm-cfg-impl.svg" alt="vLLM CFG Implementation" width="100%" />
  <p><em>vLLM CFG Implementation</em></p>
</div>

# Changelog

## `0.1.5`
* Fix Python packaging missing the tokenizer.json file

## `0.1.4`
* Change default step count back to 10 due to feedback about quality degradation.
* Fixed a bug in the `gradio_tts_app.py` implementation (#13).
* Fixed a bug with how symlinks don't work if the module is installed normally (vs as a dev environment) (#12).
  * The whole structure of how this project should be integrated into downstream repos is something that needs rethinking.

## `0.1.3`
* Added ability to tweak S3Gen diffusion steps, and default it to 5 (originally 10). This improves performance with nearly indistinguishable quality loss.

## `0.1.2`
* Update to `vllm 0.10.0`
* Fixed error where batched requests sometimes get truncated, or otherwise jumbled.
  * This also removes the need to double-apply batching when submitting requests. You can submit as many prompts as you'd like into the `generate` function, and `vllm` should perform the batching internally without issue. See changes to `benchmark.py` for details.
  * There is still a (very rare, theoretical) possibility that this issue can still happen. If it does, submit a ticket with repro steps, and tweak your max batch size or max token count as a workaround.


## `0.1.1`
* Misc minor cleanups
* API changes:
  * Use `max_batch_size` instead of `gpu_memory_utilization`
  * Use `compile=False` (default) instead of `enforce_eager=True`
  * Look at the latest examples to follow API changes. As a reminder, I do not expect the API to become stable until `1.0.0`.

## `0.1.0`
* Initial publication to pypi
* Moved audio conditioning processing out of vLLM to avoid re-computing it for every request.

## `0.0.1`
* Initial release
