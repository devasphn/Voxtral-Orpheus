from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoProcessor, VoxtralForConditionalGeneration
from mistral_common.protocol.instruct.messages import UserMessage
import torch
import base64
import numpy as np
import soundfile as sf # We can use soundfile again now that we are feeding it a clean WAV
import logging
import tempfile
import subprocess # Import the subprocess module
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
        
        # --- THE DEFINITIVE FFMPEG FIX ---
        # Use temporary files to handle the conversion robustly
        with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as temp_in:
            temp_in.write(audio_data)
            temp_in_path = temp_in.name
        
        # Define the output path for the converted WAV file
        temp_out_path = temp_in_path + '.wav'

        # Build and run the ffmpeg command
        # -i: input file
        # -ar 16000: set audio sample rate to 16kHz
        # -ac 1: set audio channels to 1 (mono)
        # -f wav: output format is WAV
        # -y: overwrite output file if it exists
        command = [
            "ffmpeg",
            "-i", temp_in_path,
            "-ar", "16000",
            "-ac", "1",
            "-f", "wav",
            "-y", temp_out_path
        ]
        
        logger.info(f"Running ffmpeg command: {' '.join(command)}")
        subprocess.run(command, check=True, capture_output=True)

        # Now, load the perfectly formatted WAV file with soundfile
        audio_np, original_sr = sf.read(temp_out_path)
        
        # Clean up the temporary files
        os.remove(temp_in_path)
        os.remove(temp_out_path)
        # --- END OF FIX ---

        inputs = processor(
            [UserMessage(content=audio_np)],
            sampling_rate=16000,
            return_tensors="pt"
        ).to(device)

        generated_ids = model.generate(**inputs, max_new_tokens=150)
        response_text = processor.decode(generated_ids[0], skip_special_tokens=True)
        
        logger.info(f"Generated response: '{response_text}'")
        return {"text": response_text}
    except subprocess.CalledProcessError as e:
        logger.error(f"FFMPEG conversion failed: {e.stderr.decode()}")
        raise HTTPException(status_code=400, detail=f"Audio conversion failed: {e.stderr.decode()}")
    except Exception as e:
        logger.error(f"Error processing audio: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred while processing audio: {e}")
