from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from orpheus-speech import OrpheusModel
import torch
import base64
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [Orpheus] - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI()
model = None

@app.on_event("startup")
def load_model():
    """Load the Orpheus model into memory when the service starts."""
    global model
    logger.info("Loading Orpheus TTS model...")
    try:
        model = OrpheusModel(
            model_name="canopylabs/orpheus-tts-0.1-finetune-prod",
            dtype=torch.bfloat16,
            gpu_memory_utilization=0.9
        )
        logger.info("✅ Orpheus TTS model loaded successfully.")
    except Exception as e:
        logger.error(f"❌ Failed to load Orpheus model: {e}", exc_info=True)
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
        audio_chunks = []
        for chunk in model.generate_speech(prompt=request.text, voice=request.voice):
            if chunk is not None:
                audio_chunks.append(chunk)

        if not audio_chunks:
            raise HTTPException(status_code=500, detail="Audio generation produced no output.")
            
        full_audio = b''.join(audio_chunks)
        audio_b64 = base64.b64encode(full_audio).decode('utf-8')
        
        logger.info("Successfully synthesized audio.")
        return {"audio": audio_b64}
    except Exception as e:
        logger.error(f"Error during synthesis: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during audio synthesis: {e}")
