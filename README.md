# Chatterbox TTS on vLLM

This is a port of https://github.com/resemble-ai/chatterbox to vLLM. Why?

* Improved performance and more efficient use of GPU memory.
  * Early benchmarks show ~4x speedup in generation toks/s without batching, and over 10x with batching. This is a significant improvement over the original Chatterbox implementation, which was bottlenecked by unnecessary CPU-GPU sync/transfers within the HF transformers implementation.
  * More rigorous benchmarking is WIP, but will likely come after batching is fully fleshed out.
* Easier integration with state-of-the-art inference infrastructure.

DISCLAIMER: THIS IS A PERSONAL PROJECT and is not affiliated with my employer or any other corporate entity in any way. The project is based solely on publicly-available information. All opinions are my own and do not necessarily represent the views of my employer.

## Generation Samples

![Sample 1](docs/audio-sample-01.wav)
<audio controls>
  <source src="docs/audio-sample-01.wav" type="audio/wav">
</audio>

![Sample 2](docs/audio-sample-02.wav)
<audio controls>
  <source src="docs/audio-sample-02.wav" type="audio/wav">
</audio>

![Sample 3](docs/audio-sample-03.wav)
<audio controls>
  <source src="docs/audio-sample-03.wav" type="audio/wav">
</audio>


# Project Status: Minimal Viable Implementation

* ✅ Basic speech cloning with audio and text conditioning.
* ✅ Outputs match the quality of the original Chatterbox implementation.
* ✅ CFG=0.5 is implemented.
* ✅ Exaggeration control is implemented.
* ℹ️ Project uses vLLM internal APIs and hacks to get things done. Refactoring to idomatic vLLM way of doing things is WIP, but may require some changes to vLLM.
* ℹ️ Substantial refactoring is needed to further clean up unnecessary workarounds and code paths.
* ℹ️ Server API is not implemented and will likely be out-of-scope for this project.
* ❌ CFG scale is hard-coded at 0.5 and is not yet implemented.
* ❌ APIs are not yet stable and may change.
* ❌ vLLM batching is not (yet) supported.
* ❌ Benchmarks and performance optimizations are not yet implemented.
* ❌ Installation process can be tricky and has room for improvement.


# Installation

1. The chatterbox model weights should exist in your HF cache. Fix the symlink inside `./t3-model` if needed. This is a temporary setup to unblock development, and will be replaced with a better setup once a MVP is up and running.
1. Set up the environment via:

```
uv venv
source .venv/bin/activate
uv pip install -e .
```
3. You should be able to run `python example-tts.py` to generate audio samples.

# Chatterbox Architecture

I could not find an official explanation of the Chatterbox architecture, so below is my best explanation based on the codebase. Chatterbox broadly follows the [CosyVoice](https://funaudiollm.github.io/cosyvoice2/) architecture, applying intermediate fusion multimodal conditioning to a 0.5B parameter Llama model.

<div align="center">
  <img src="https://github.com/randombk/chatterbox-vllm/raw/refs/heads/master/docs/chatterbox-architecture.svg" alt="Chatterbox Architecture" width="100%" />
  <p><em>Chatterbox Architecture Diagram</em></p>
</div>

# Implementation Notes

## CFG Implementation Details

VLLM does not support CFG natively, so substantial hacks were needs to make it work. At a high level, we trick VLLM into thinking the model has double the hidden dimension size as it actually does, then splitting and restacking the states to invoke Llama with double the original batch size. This does pose a risk that VLLM will underestimate the memory requirements of the model - more research is needed into whether VLLM's initial profiling pass will capture this nuance.


<div align="center">
  <img src="https://github.com/randombk/chatterbox-vllm/raw/refs/heads/master/docs/vllm-cfg-impl.svg" alt="VLLM CFG Implementation" width="100%" />
  <p><em>VLLM CFG Implementation</em></p>
</div>