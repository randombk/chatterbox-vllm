import random
import os

import numpy as np
import torch
import gradio as gr
import torchaudio as ta

from chatterbox_vllm.tts import ChatterboxTTS

DEVICE = "cuda"

config_seed = None
global_model = None

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    global config_seed
    config_seed = seed


def load_model():
    print("Loading model...")
    global global_model
    global_model = ChatterboxTTS.from_pretrained(
        gpu_memory_utilization = 0.6,
        max_model_len = 1000,

        # Disable CUDA graphs - it's causing tensors to get corrupted right now.
        enforce_eager = True,
    )
    return global_model

def generate(text, audio_prompt_path, exaggeration, temperature, seed_num,
             #cfgw,
             diffusion_steps,
             min_p, top_p, repetition_penalty):
    if seed_num != 0:
        set_seed(int(seed_num))

    print(f"Using text: {text}")
    print(f"Using audio_prompt_path: {audio_prompt_path}")
    print(f"Using seed: {config_seed}")
    print(f"Using temperature: {temperature}")
    print(f"Using exaggeration: {exaggeration}")
    print(f"Using min_p: {min_p}")
    print(f"Using top_p: {top_p}")
    print(f"Using repetition_penalty: {repetition_penalty}")

    wav = global_model.generate(
        [text],
        audio_prompt_path=audio_prompt_path,
        exaggeration=exaggeration,
        temperature=temperature,
        # cfg_weight=cfgw,
        diffusion_steps=diffusion_steps,
        min_p=min_p,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        seed=config_seed,
    )
    return (global_model.sr, wav[0].squeeze(0).numpy())


with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            text = gr.Textbox(
                value="Now let's make my mum's favourite. So three mars bars into the pan. Then we add the tuna and just stir for a bit, just let the chocolate and fish infuse. A sprinkle of olive oil and some tomato ketchup. Now smell that. Oh boy this is going to be incredible.",
                label="Text to synthesize (max chars 300)",
                max_lines=5
            )
            ref_wav = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Reference Audio File", value=None)
            exaggeration = gr.Slider(0.25, 2, step=.05, label="Exaggeration (Neutral = 0.5, extreme values can be unstable)", value=.5)
            # cfg_weight = gr.Slider(0.0, 1, step=.05, label="CFG/Pace", value=0.5)

            with gr.Accordion("More options", open=False):
                seed_num = gr.Number(value=0, label="Random seed (0 for random)")
                diffusion_steps = gr.Slider(1, 15, step=1, label="Diffusion Steps (more = slower and higher quality)", value=10)
                temp = gr.Slider(0.05, 5, step=.05, label="temperature", value=.8)
                min_p = gr.Slider(0.00, 1.00, step=0.01, label="min_p || Newer Sampler. Recommend 0.02 > 0.1. Handles Higher Temperatures better. 0.00 Disables", value=0.05)
                top_p = gr.Slider(0.00, 1.00, step=0.01, label="top_p || Original Sampler. 1.0 Disables(recommended). Original 0.8", value=1.00)
                repetition_penalty = gr.Slider(1.00, 2.00, step=0.1, label="repetition_penalty", value=1.2)

            run_btn = gr.Button("Generate", variant="primary")

        with gr.Column():
            audio_output = gr.Audio(label="Output Audio")

    run_btn.click(
        fn=generate,
        inputs=[
            text,
            ref_wav,
            exaggeration,
            temp,
            seed_num,
            #cfg_weight,
            diffusion_steps,
            min_p,
            top_p,
            repetition_penalty,
        ],
        outputs=audio_output,
    )

if __name__ == "__main__":
    # Don't let Gradio manage the model loading, it's causing issues.
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    load_model()

    print("Starting Gradio app...")
    demo.queue(
        max_size=50,
        default_concurrency_limit=1,
    ).launch(share=True)
