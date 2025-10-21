"""
TrueBDC-ASR Wrapper (Swagger-enabled)
- FastAPI on 0.0.0.0:8888
- Whisper (openai/whisper-large-v3-turbo) EXACTLY as in the Gradio client:
  * torch.float16
  * torch.backends.cuda.matmul.allow_tf32 = True
  * torch._dynamo.config.suppress_errors = True
  * AutoProcessor + AutoModelForSpeechSeq2Seq
  * pipeline("automatic-speech-recognition", ..., device="cuda:2")
  * warmup via sdpa_kernel(SDPBackend.MATH) with 2s of silence
- Optional: forward transcript to upstream /tts and return audio (base64) in the same JSON
- Swagger UI at /docs; ReDoc at /redoc; OpenAPI at /openapi.json
"""

import base64
import io
import logging
import os
import tempfile
import time
import warnings
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import requests
import torch
import torch._dynamo as dynamo
import wave
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from scipy.io.wavfile import write as write_wav  # match original warmup creation

# --------------------------
# Configuration (exact model + your requested GPU2)
# --------------------------

WRAPPER_HOST = "0.0.0.0"
WRAPPER_PORT = 8888

# Upstream base: bind is 0.0.0.0:8000, but connecting should use loopback
_DEFAULT_BASE = "http://127.0.0.1:8000"
API_BASE_URL = os.getenv("TTS_API_BASE_URL", _DEFAULT_BASE).rstrip("/")
if API_BASE_URL.startswith("http://0.0.0.0"):
    API_BASE_URL = API_BASE_URL.replace("0.0.0.0", "127.0.0.1", 1)

# EXACT model id from your Gradio app
WHISPER_TURBO_ID = "openai/whisper-large-v3-turbo"

# GPU 2 per your instruction
ASR_DEVICE_INDEX = 2

# timeouts
REQUEST_TIMEOUT = float(os.getenv("TTS_REQUEST_TIMEOUT_S", "120"))

# --------------------------
# Logging / warnings exactly like your client
# --------------------------

warnings.filterwarnings("ignore", category=UserWarning, module="torch")
logging.getLogger("transformers").setLevel(logging.ERROR)
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
torch.backends.cuda.matmul.allow_tf32 = True
dynamo.config.suppress_errors = True

# --------------------------
# Small helpers
# --------------------------

def _wav_duration_seconds(path: Path) -> float:
    try:
        with wave.open(str(path), "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            return frames / float(rate) if rate else 0.0
    except wave.Error:
        return 0.0

def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")

# --------------------------
# Upstream TTS client (used only when /whisper?return_audio=true)
# --------------------------

class TTSClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "TrueBDC-WhisperWrapper/1.0"})

    def get_predefined_voices(self) -> Dict[str, str]:
        url = f"{self.base_url}/get_predefined_voices"
        r = self.session.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        if not isinstance(data, list) or not data:
            raise RuntimeError("Unexpected voices payload from upstream.")
        voice_map: Dict[str, str] = {}
        for d in data:
            vals = list(d.values())
            if len(vals) >= 2:
                name, filename = vals[0], vals[1]
                voice_map[name] = filename
            elif len(vals) == 1:
                filename = vals[0]
                name = Path(filename).stem.replace("_", " ").title()
                voice_map[name] = filename
        if not voice_map:
            raise RuntimeError("Parsed empty voices map from upstream.")
        return voice_map

    def tts(self, text: str, predefined_voice_id: str, output_format: str = "wav") -> bytes:
        url = f"{self.base_url}/tts"
        # Payload matches your original client (no guessing)
        payload = {
            "text": text,
            "voice_mode": "predefined",
            "predefined_voice_id": predefined_voice_id,
            "output_format": output_format,
            # leave other server defaults intact, as in your client
            # "temperature": ...,
            # "speed_factor": ...,
        }
        r = self.session.post(url, json=payload, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        return r.content

# --------------------------
# Whisper ASR (exact pipeline + warmup as in your client)
# --------------------------

class ASREngine:
    def __init__(self, device_index: int):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the Whisper ASR model.")
        if device_index < 0 or device_index >= torch.cuda.device_count():
            raise RuntimeError(
                f"Requested cuda:{device_index}, but only {torch.cuda.device_count()} visible."
            )

        self.device = f"cuda:{device_index}"
        torch.cuda.set_device(device_index)

        self.torch_dtype = torch.float16

        # Create dummy warmup file EXACTLY like your client (2s silence, float32)
        self.warmup_audio_path = Path("warmup.wav")
        if not self.warmup_audio_path.exists():
            dummy_audio = np.zeros(16000 * 2, dtype=np.float32)
            write_wav(self.warmup_audio_path, 16000, dummy_audio)

        # Load processor + model exactly as shown
        self.processor = AutoProcessor.from_pretrained(WHISPER_TURBO_ID)
        self.asr_model = AutoModelForSpeechSeq2Seq.from_pretrained(
            WHISPER_TURBO_ID,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        ).to(self.device)

        # Create the recognition pipeline exactly as in the Gradio app
        self.asr_pipe = pipeline(
            "automatic-speech-recognition",
            model=self.asr_model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            torch_dtype=self.torch_dtype,
            device=self.device,
        )

        # Warmup with sdpa_kernel(SDPBackend.MATH), as in your app
        with sdpa_kernel(SDPBackend.MATH):
            _ = self.asr_pipe(str(self.warmup_audio_path), generate_kwargs={"language": "english"})

    def transcribe_path(self, audio_path: str) -> str:
        with sdpa_kernel(SDPBackend.MATH):
            out = self.asr_pipe(audio_path, generate_kwargs={"language": "english"})
        return (out.get("text") or "").strip()

class PinnedWorker:
    """CUDA context pinned to a single thread, matching your pattern."""
    def __init__(self, device_index: int):
        self.exec = ThreadPoolExecutor(max_workers=1, thread_name_prefix="PinnedCUDAWorker")
        self.engine: ASREngine = self.exec.submit(ASREngine, device_index).result()

    def transcribe(self, audio_path: str) -> str:
        fut = self.exec.submit(self.engine.transcribe_path, audio_path)
        return fut.result()

# --------------------------
# FastAPI app (+ Swagger)
# --------------------------

app = FastAPI(
    title="TrueBDC-ASR Wrapper",
    version="1.0.0",
    description=(
        "ASR microservice that mirrors your original Whisper pipeline on GPU 2 and can optionally "
        "hand off the transcript to your TrueBDC TTS server."
    ),
    docs_url="/docs",          # Swagger UI
    redoc_url="/redoc",        # ReDoc
    openapi_url="/openapi.json"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_headers=["*"],
    allow_methods=["*"],
    allow_credentials=False,
)

_tts: Optional[TTSClient] = None
_asr: Optional[PinnedWorker] = None
_cached_voices: Dict[str, str] = {}
_cached_voices_at: float = 0.0
VOICES_TTL_S = 60

@app.on_event("startup")
def _startup():
    global _tts, _asr
    _tts = TTSClient(API_BASE_URL)
    _asr = PinnedWorker(ASR_DEVICE_INDEX)

# --------------------------
# Schemas
# --------------------------

class WhisperResponse(BaseModel):
    text: str = Field(..., description="Transcribed text")
    input_duration_s: float = Field(..., description="Input WAV duration in seconds")
    asr_time_s: float = Field(..., description="ASR wall time in seconds")
    rtf_asr: float = Field(..., description="Real-time factor (duration / ASR time)")
    audio_b64: Optional[str] = Field(None, description="Base64-encoded synthesized audio if return_audio=true")
    output_format: Optional[str] = Field(None, description="Audio format used for synthesis (wav|opus|mp3)")
    selected_voice_id: Optional[str] = Field(None, description="Upstream predefined voice id used for synthesis")

# --------------------------
# Routes
# --------------------------

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def root():
    return """
    <html>
      <head><title>TrueBDC-ASR Wrapper</title></head>
      <body>
        <h2>TrueBDC-ASR Wrapper</h2>
        <p>Swagger UI is available at <a href="/docs">/docs</a>.</p>
        <ul>
          <li><a href="/health">/health</a></li>
          <li><a href="/self_test">/self_test</a></li>
          <li>POST <code>/whisper</code> (see Swagger)</li>
        </ul>
      </body>
    </html>
    """

@app.get("/health", response_class=PlainTextResponse, tags=["Meta"])
def health():
    return "ok"

@app.get(
    "/self_test",
    tags=["Meta"],
    summary="Self-test ASR (and optionally TTS)",
    description="Checks CUDA, runs ASR warm path, and optionally pings upstream TTS.",
)
def self_test(
    with_tts: bool = Query(False, description="Also test upstream voices and /tts"),
    output_format: str = Query("wav", pattern="^(wav|opus|mp3)$"),
):
    assert _asr is not None
    info = {
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "asr_device": _asr.engine.device,
        "asr_gpu_name": torch.cuda.get_device_name(ASR_DEVICE_INDEX),
        "model_id": WHISPER_TURBO_ID,
        "sdpa_backend": "MATH",
    }
    t0 = time.time()
    text = _asr.engine.transcribe_path(str(_asr.engine.warmup_audio_path))
    asr_dt = time.time() - t0
    info["asr_smoke"] = {"text": text, "time_s": round(asr_dt, 4)}

    tts_res = {"tested": False, "ok": False}
    if with_tts:
        try:
            global _cached_voices, _cached_voices_at
            now = time.time()
            if not _cached_voices or (now - _cached_voices_at) > VOICES_TTL_S:
                _cached_voices = _tts.get_predefined_voices()
                _cached_voices_at = now
            _, voice_id = next(iter(_cached_voices.items()))
            audio = _tts.tts("This is a TrueBDC TTS self test.", predefined_voice_id=voice_id, output_format=output_format)
            tts_res.update({"tested": True, "ok": True, "voice_id": voice_id, "bytes": len(audio)})
        except Exception as e:
            tts_res.update({"tested": True, "ok": False, "error": str(e)})

    return {"system": info, "tts": tts_res}

@app.post(
    "/whisper",
    response_model=WhisperResponse,
    tags=["ASR"],
    summary="Transcribe WAV, optionally synthesize via upstream TTS",
    description=(
        "Upload a WAV file. Returns transcript and timing. "
        "If return_audio=true, also calls upstream /tts and returns base64 audio."
    ),
)
async def whisper(
    audio: UploadFile = File(..., description="WAV file (audio/wav)"),
    return_audio: bool = Form(False, description="Also synthesize transcript via upstream /tts"),
    predefined_voice_id: Optional[str] = Form(None, description="Upstream voice id (filename). If omitted and return_audio=true, the first upstream voice is used."),
    output_format: str = Form("wav", pattern="^(wav|opus|mp3)$"),
):
    if audio.content_type not in ("audio/wav", "audio/x-wav", "application/octet-stream"):
        raise HTTPException(415, "Only WAV is supported for /whisper.")

    assert _asr is not None

    raw = await audio.read()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = Path(tmp.name)
        tmp.write(raw)

    try:
        input_dur = _wav_duration_seconds(tmp_path)
        t0 = time.time()
        text = await run_in_threadpool(_asr.transcribe, str(tmp_path))
        asr_time = time.time() - t0

        resp = WhisperResponse(
            text=text,
            input_duration_s=round(input_dur, 4),
            asr_time_s=round(asr_time, 4),
            rtf_asr=round((input_dur / asr_time), 4) if asr_time > 0 else float("inf"),
        )

        if return_audio:
            if not text:
                raise HTTPException(400, "ASR produced empty transcript; cannot synthesize audio.")
            voice_id = predefined_voice_id
            if voice_id is None:
                # get first available voice (do NOT expose upstream voices as endpoints)
                global _cached_voices, _cached_voices_at
                now = time.time()
                if not _cached_voices or (now - _cached_voices_at) > VOICES_TTL_S:
                    _cached_voices = _tts.get_predefined_voices()
                    _cached_voices_at = now
                _, voice_id = next(iter(_cached_voices.items()))
            try:
                audio_bytes = _tts.tts(text, predefined_voice_id=voice_id, output_format=output_format)
            except requests.RequestException as e:
                raise HTTPException(502, f"Upstream TTS failed: {e}")

            resp.audio_b64 = _b64(audio_bytes)
            resp.output_format = output_format
            resp.selected_voice_id = voice_id

        return resp
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass

# Entrypoint for `python whisper_server.py`
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "whisper_server:app",
        host=WRAPPER_HOST,
        port=WRAPPER_PORT,
        reload=False,
        workers=1,  # keep single process so CUDA context & pipeline remain stable
    )
