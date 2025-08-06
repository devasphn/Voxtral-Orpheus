from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import VoxtralForConditionalGeneration, AutoProcessor
import torch
import base64
import numpy as np
import soundfile as sf
import logging
import tempfile
import subprocess
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [Voxtral] - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI()
processor = None
model = None
device = "cuda" if torch.cuda.is_available() else "cpu"

@app.on_event("startup")
def load_model():
    """Load the Voxtral model into memory when the service starts."""
    global processor, model
    model_id = "mistralai/Voxtral-Mini-3B-2507"
    
    logger.info(f"Loading Voxtral model '{model_id}' to device '{device}'...")
    try:
        processor = AutoProcessor.from_pretrained(model_id)
        model = VoxtralForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map=device,
            attn_implementation="flash_attention_2"  # Enable flash attention
        )
        logger.info("✅ Voxtral model loaded successfully.")
    except Exception as e:
        logger.error(f"❌ Failed to load Voxtral model: {e}", exc_info=True)
        model = None

class ASRRequest(BaseModel):
    audio: str  # Base64 encoded webm audio

@app.post("/process")
def process_audio(request: ASRRequest):
    """Takes base64-encoded audio, returns transcribed text using proper Voxtral workflow."""
    if not model or not processor:
        raise HTTPException(status_code=503, detail="Model is not loaded or failed to initialize.")

    logger.info("Processing incoming audio request...")
    try:
        # Decode base64 audio
        audio_data = base64.b64decode(request.audio)
        
        # Create temporary files
        with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as temp_in:
            temp_in.write(audio_data)
            temp_in_path = temp_in.name
        
        temp_out_path = temp_in_path + '.wav'

        # Convert WebM to WAV at 16kHz (required for Voxtral/Whisper)
        command = [
            "ffmpeg", "-i", temp_in_path, 
            "-ar", "16000",  # 16kHz sampling rate
            "-ac", "1",      # Mono channel
            "-f", "wav", 
            "-y", temp_out_path
        ]
        subprocess.run(command, check=True, capture_output=True, text=True)

        # Load audio for processing
        audio_np, sampling_rate = sf.read(temp_out_path)
        
        # Clean up temp files
        os.remove(temp_in_path)
        os.remove(temp_out_path)
        
        # Use Voxtral for transcription (correct method)
        logger.info("Using Voxtral transcription mode...")
        
        # Apply transcription request properly
        inputs = processor.apply_transcription_request(
            language="en", 
            audio=audio_np,
            model_id="mistralai/Voxtral-Mini-3B-2507",
            sampling_rate=sampling_rate
        )
        
        inputs = inputs.to(device, dtype=torch.bfloat16)
        
        # Generate transcription
        generated_ids = model.generate(**inputs, max_new_tokens=150)
        
        # Decode the transcription
        transcription = processor.batch_decode(
            generated_ids[:, inputs.input_ids.shape[1]:], 
            skip_special_tokens=True
        )[0]
        
        logger.info(f"Transcribed text: '{transcription}'")
        return {"text": transcription.strip()}
        
    except subprocess.CalledProcessError as e:
        logger.error(f"FFMPEG conversion failed: {e.stderr}")
        raise HTTPException(status_code=400, detail="Audio conversion failed.")
    except Exception as e:
        logger.error(f"Error processing audio: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Audio processing error: {str(e)}")

@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy", 
        "model_loaded": model is not None,
        "device": device
    }
