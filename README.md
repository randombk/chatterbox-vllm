# Chatterbox TTS - VLLM Server

This is a port of https://github.com/resemble-ai/chatterbox to VLLM

# WORK IN PROGRESS

**This does not work in its current state.** Significant model components still need to be ported over to vLLM before this is close to working. The model technically runs and generates output, but only noise is generated.

**Current Status**:
 * Most of the plumbing should be in place, but there' still _something_ wrong causing the model to generate complete gibberish.
 * The embedding tensor matches the reference implementation. Whatever is wrong comes after this point.
 * Something seems wrong with the autoregression logic. The KV caching logic is suspect. That might be the source of the descrepancies, but it should not explain the embedding issues.
 * The project is currently _very_ hacked together, using ugly workarounds and digging into VLLM internals to get things done. I'm sure there's a proper way or API to do most of the things I'm doing, but I couldn't find it.
 * There's a lot of optimizations around moving the audio processing parts to/from GPU. For simplicity sake I moved them all to GPU until I could figure out what I want to do with them.
 * VLLM batching is 100% not going to work without major cleanup to adopt VLLM's batching model.

I expect my work on this will pause until mid-July, PRs are welcome.

## How to set up dev environment

1. The chatterbox model weights should exist in your HF cache. Fix the symlink inside `./t3-model` if needed. This is a temporary setup to unblock development, and will be replaced with a better setup once a MVP is up and running.
1. Set up the environment via:

```
uv venv
source .venv/bin/activate
uv pip install -e .
```
3. You should be able to run `python example-tts.py` and not have it crash.
4. That's as far as things have gotten thus far.

# Disclaimer

THIS IS A PERSONAL PROJECT and is not affiliated with my employer in any way. The project is based solely on publicly-available information. All opinions are my own and do not necessarily represent the views of my employer.
