from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoProcessor, VoxtralForConditionalGeneration
# The UserMessage object is no longer needed for this simple audio-only task
# from mistral_common.protocol.instruct.messages import UserMessage 
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
            attn_implementation="flash_attention_2"
        ).to(device)
        logger.info("✅ Voxtral model loaded successfully.")
    except Exception as e:
        logger.error(f"❌ Failed to load Voxtral model: {e}", exc_info=True)
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
        audio_data = base64.b64decode(request.audio)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as temp_in:
            temp_in.write(audio_data)
            temp_in_path = temp_in.name
        
        temp_out_path = temp_in_path + '.wav'

        command = [
            "ffmpeg", "-i", temp_in_path,
            "-ar", "16000", "-ac", "1", "-f", "wav", "-y", temp_out_path
        ]
        
        subprocess.run(command, check=True, capture_output=True)

        audio_np, original_sr = sf.read(temp_out_path)
        
        os.remove(temp_in_path)
        os.remove(temp_out_path)
        
        # --- THE FINAL, CRITICAL FIX ---
        # Instead of wrapping the audio in a UserMessage, we pass it directly.
        # The processor is smart enough to handle a raw numpy array correctly
        # when passed this way. This prevents the pydantic/union_tag error.
        inputs = processor(
            audio=audio_np,
            sampling_rate=16000,
            return_tensors="pt"
        ).to(device)
        # --- END OF FIX ---

        generated_ids = model.generate(**inputs, max_new_tokens=150)
        response_text = processor.decode(generated_ids[0], skip_special_tokens=True)
        
        # The model sometimes outputs just the transcription. If so, let's create a simple response.
        # In a real app, you might call another LLM here.
        if request.audio in response_text: # A simple check if it just transcribed
             response_text = "I heard you say: " + response_text

        logger.info(f"Generated response: '{response_text}'")
        return {"text": response_text}
    except subprocess.CalledProcessError as e:
        logger.error(f"FFMPEG conversion failed: {e.stderr.decode()}")
        raise HTTPException(status_code=400, detail=f"Audio conversion failed.")
    except Exception as e:
        logger.error(f"Error processing audio: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred while processing audio.")
