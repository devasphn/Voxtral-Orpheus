from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoProcessor, VoxtralForConditionalGeneration
from mistral_common.protocol.instruct.messages import UserMessage
import torch
import base64
import librosa
import numpy as np
import io
import soundfile as sf
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
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
            attn_implementation="flash_attention_2" # Optimization for speed
        ).to(device)
        logger.info("✅ Voxtral model loaded successfully.")
    except Exception as e:
        logger.error(f"❌ Failed to load Voxtral model: {e}")
        model = None

class ASRRequest(BaseModel):
    audio: str # Base64 encoded webm audio

@app.post("/process")
def process_audio(request: ASRRequest):
    """Takes base64-encoded audio, returns transcribed and generated text."""
    if not model or not processor:
        raise HTTPException(status_code=503, detail="Model is not loaded or failed to initialize.")

    logger.info("Processing incoming audio request...")
    try:
        # Decode the audio from base64
        audio_data = base64.b64decode(request.audio)
        
        # Use soundfile and io.BytesIO to read the in-memory webm file
        audio_np, original_sr = sf.read(io.BytesIO(audio_data))

        # Ensure audio is mono
        if audio_np.ndim > 1:
            audio_np = np.mean(audio_np, axis=1)

        # Resample to the 16kHz required by Voxtral
        if original_sr != 16000:
            logger.info(f"Resampling audio from {original_sr}Hz to 16000Hz.")
            audio_np = librosa.resample(y=audio_np, orig_sr=original_sr, target_sr=16000)

        # Prepare the input for the model
        inputs = processor(
            [UserMessage(content=audio_np)],
            sampling_rate=16000,
            return_tensors="pt"
        ).to(device)

        # Generate a response. Voxtral will transcribe and answer in one step.
        generated_ids = model.generate(**inputs, max_new_tokens=150)
        
        # Decode the generated tokens to text
        response_text = processor.decode(generated_ids[0], skip_special_tokens=True)
        
        logger.info(f"Generated response: '{response_text}'")
        return {"text": response_text}
    except Exception as e:
        logger.error(f"Error processing audio: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred while processing audio: {e}")
