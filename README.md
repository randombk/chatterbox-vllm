# Chatterbox TTS - VLLM Server

This is a port of https://github.com/resemble-ai/chatterbox to VLLM

# WORK IN PROGRESS

This does not work in its current state. Significant model components still need to be ported over to vLLM before this is close to working.

Currently, only a very limited subset of the patch T3 Llama model has been ported, and is able to load the weights and get past early engine init. However conditioning has not been implemented.

I expect my work on this will pause until mid-July, but PRs are welcome.

## How to set up dev environment

1. The chatterbox model weights should exist in your HF cache. Fix the symlink inside `model-dev` if needed.
1. Set up the environment via:

```
uv venv
source .venv/bin/activate
uv pip install -e .
```
3. You should be able to run `python example-tts.py` and not have it crash.
4. That's as far as things have gotten thus far.