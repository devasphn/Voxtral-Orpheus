from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging
import base64
import numpy as np
import io
import wave
import asyncio
import os
from typing import AsyncGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [Orpheus] - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI()

# Global model variable
orpheus_model = None

@app.on_event("startup")
async def load_model():
    """Load the Orpheus model using the correct orpheus-speech package."""
    global orpheus_model
    logger.info("Loading Orpheus TTS model...")
    
    try:
        # Import here to avoid issues if not installed
        from orpheus_tts import OrpheusModel
        
        # Load the production model
        orpheus_model = OrpheusModel(
            model_name="canopylabs/orpheus-tts-0.1-finetune-prod",
            max_model_len=2048
        )
        logger.info("✅ Orpheus TTS model loaded successfully.")
        
    except ImportError as e:
        logger.error(f"❌ Failed to import orpheus-speech: {e}")
        logger.error("Please install: pip install orpheus-speech")
        orpheus_model = None
    except Exception as e:
        logger.error(f"❌ Failed to load Orpheus model: {e}", exc_info=True)
        orpheus_model = None

class TTSRequest(BaseModel):
    text: str
    voice: str = "tara"

def audio_generator_to_wav(audio_generator, sample_rate: int = 24000) -> bytes:
    """Convert Orpheus audio generator to WAV bytes."""
    # Collect all audio chunks
    audio_chunks = []
    for chunk in audio_generator:
        audio_chunks.append(chunk)
    
    if not audio_chunks:
        raise ValueError("No audio generated")
    
    # Concatenate all chunks
    audio_data = b''.join(audio_chunks)
    
    # Convert to numpy array (assuming 16-bit audio)
    audio_np = np.frombuffer(audio_data, dtype=np.int16)
    
    # Create WAV file in memory
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, 'wb') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)  # 24kHz
        wav_file.writeframes(audio_data)
    
    return wav_buffer.getvalue()

@app.post("/synthesize")
async def synthesize_speech(request: TTSRequest):
    """Takes text and returns base64-encoded WAV audio using Orpheus TTS."""
    if not orpheus_model:
        raise HTTPException(
            status_code=503, 
            detail="Orpheus model is not loaded. Please check server logs."
        )
    
    logger.info(f"Synthesizing text: '{request.text[:50]}...' with voice '{request.voice}'")
    
    try:
        # Generate speech using Orpheus
        audio_generator = orpheus_model.generate_speech(
            prompt=request.text,
            voice=request.voice,
            temperature=0.7,
            repetition_penalty=1.1
        )
        
        # Convert generator to WAV
        wav_data = audio_generator_to_wav(audio_generator, sample_rate=24000)
        
        # Encode as base64
        audio_b64 = base64.b64encode(wav_data).decode('utf-8')
        
        logger.info("Successfully synthesized audio using Orpheus TTS.")
        return {"audio": audio_b64}
        
    except Exception as e:
        logger.error(f"Error during Orpheus synthesis: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail=f"TTS synthesis failed: {str(e)}"
        )

@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": orpheus_model is not None
    }

@app.get("/voices")
def get_available_voices():
    """Get list of available voices."""
    return {
        "voices": ["tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe"],
        "default": "tara"
    }
