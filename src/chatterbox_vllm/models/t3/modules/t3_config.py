class T3Config:
    start_text_token = 255
    stop_text_token = 0
    max_text_tokens = 2048

    # HACK: We're hard-coding this into t3.py for now
    # text_tokens_dict_size_english = 704
    # text_tokens_dict_size_multilingual = 2352

    start_speech_token = 6561
    stop_speech_token = 6562
    speech_tokens_dict_size = 8194
    max_speech_tokens = 4096

    llama_config_name = "Llama_520M"
    input_pos_emb = "learned"
    speech_cond_prompt_len = 150

    # For T3CondEnc
    encoder_type = "voice_encoder"
    speaker_embed_size = 256
    use_perceiver_resampler = True
    emotion_adv = True
    n_channels = 1024 # hidden_size from config.json
