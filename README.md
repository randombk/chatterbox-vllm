# Chatterbox TTS on vLLM

This is a port of https://github.com/resemble-ai/chatterbox to vLLM. Why?

* Improved performance and more efficient use of GPU memory.
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
* ℹ️ Project uses vLLM internal APIs and hacks to get things done. Refactoring to idomatic vLLM way of doing things is WIP, but may require some changes to vLLM.
* ℹ️ Substantial refactoring is needed to further clean up unnecessary workarounds and code paths.
* ❌ APIs are not yet stable and may change.
* ❌ CFG and exaggeration are not yet implemented.
* ❌ vLLM batching is not (yet) supported.
* ❌ Benchmarks and performance optimizations are not yet implemented.
* ❌ Installation process can be tricky and has room for improvement.
* ❌ Server API is not implemented. This will likely be out-of-scope for this project.


# Installation

1. The chatterbox model weights should exist in your HF cache. Fix the symlink inside `./t3-model` if needed. This is a temporary setup to unblock development, and will be replaced with a better setup once a MVP is up and running.
1. Set up the environment via:

```
uv venv
source .venv/bin/activate
uv pip install -e .
```
3. You should be able to run `python example-tts.py` to generate audio samples.
4. That's as far as things have gotten thus far.

