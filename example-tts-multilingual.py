#!/usr/bin/env python3

from typing import List
import torchaudio as ta
from chatterbox_vllm.tts import ChatterboxTTS


if __name__ == "__main__":
    model = ChatterboxTTS.from_pretrained_multilingual(
        max_batch_size = 3,
        max_model_len = 1000,
    )

    for language_id, audio_prompt_path, prompts in [
        ("fr", "docs/fr_f1.flac", [
            "Bonjour, comment ça va? Ceci est le modèle de synthèse vocale multilingue Chatterbox, il prend en charge 23 langues.",
        ]),
        ("de", "docs/de_f1.flac", [
            "Letzten Monat haben wir einen neuen Meilenstein erreicht: zwei Milliarden Aufrufe auf unserem YouTube-Kanal.",
        ]),
        ("zh", "docs/zh_m1.mp3", [
            "你好，很高兴见到你。",
            "上个月，我们达到了一个新的里程碑. 我们的YouTube频道观看次数达到了二十亿次，这绝对令人难以置信。",
        ]),
    ]:   
        audios = model.generate(prompts, audio_prompt_path=audio_prompt_path, exaggeration=0.5, language_id=language_id)
        for audio_idx, audio in enumerate(audios):
            ta.save(f"test-{language_id}-{audio_idx}.mp3", audio, model.sr)

    model.shutdown()