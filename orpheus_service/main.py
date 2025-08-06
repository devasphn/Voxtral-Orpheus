from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
# Import the new llama.cpp-based backend
from orpheus_cpp import OrpheusCpp
import numpy as np
import base64
import logging
import io
from scipy.io.wavfile import write as write_wav

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [Orpheus-CPP] - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI()
model = None

@app.on_event("startup")
def load_model():
    """Load the Orpheus model using the OrpheusCpp backend."""
    global model
    logger.info("Loading Orpheus-CPP TTS model...")
    try:
        model = OrpheusCpp(verbose=False, lang="en", options={"n_gpu_layers": -1})
        logger.info("✅ Orpheus-CPP TTS model loaded successfully.")
    except Exception as e:
        logger.error(f"❌ Failed to load Orpheus-CPP model: {e}", exc_info=True)
        model = None

class TTSRequest(BaseModel):
    text: str
    voice: str = "tara"

@app.post("/synthesize")
def synthesize_speech(request: TTSRequest):
    """Takes text and returns base64-encoded WAV audio."""
    if not model:
        raise HTTPException(status_code=503, detail="Model is not loaded or failed to initialize.")
    
    logger.info(f"Synthesizing text: '{request.text[:50]}...' with voice '{request.voice}'")
    try:
        buffer = []
        # The Cpp model streams audio as (sample_rate, numpy_chunk)
        for i, (sr, chunk) in enumerate(model.stream_tts_sync(request.text, options={"voice_id": request.voice})):
            buffer.append(chunk)

        if not buffer:
            raise HTTPException(status_code=500, detail="Audio generation produced no output.")
        
        # The final output is a numpy array that we must convert to a WAV file in memory
        full_audio_np = np.concatenate(buffer, axis=1)
        
        # Use an in-memory bytes buffer to write the WAV file
        bytes_wav = io.BytesIO()
        write_wav(bytes_wav, rate=24000, data=np.concatenate(full_audio_np))
        wav_data = bytes_wav.getvalue()
        
        audio_b64 = base64.b64encode(wav_data).decode('utf-8')
        
        logger.info("Successfully synthesized audio using CPP backend.")
        return {"audio": audio_b64}
    except Exception as e:
        logger.error(f"Error during CPP synthesis: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during audio synthesis: {e}")
