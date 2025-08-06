from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoProcessor, VoxtralForConditionalGeneration
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
        
        subprocess.run(command, check=True, capture_output=True, text=True)

        audio_np, sampling_rate = sf.read(temp_out_path)
        
        os.remove(temp_in_path)
        os.remove(temp_out_path)
        
        # --- THE FINAL, DEFINITIVE FIX ---
        # The Voxtral model requires the special <|audio|> token in the prompt
        # to know where to insert the audio features. This solves the shape mismatch error.
        prompt = "Please transcribe the following audio. <|audio|>"
        
        # 1. Process the audio using the feature extractor.
        audio_inputs = processor.feature_extractor(
            audio_np, sampling_rate=sampling_rate, return_tensors="pt"
        ).to(device)

        # 2. Process the text prompt using the tokenizer.
        text_inputs = processor.tokenizer(prompt, return_tensors="pt").to(device)
        
        # 3. The model's generate function correctly accepts both sets of inputs.
        generated_ids = model.generate(
            input_features=audio_inputs.input_features,
            input_ids=text_inputs.input_ids,
            attention_mask=text_inputs.attention_mask,
            max_new_tokens=150
        )
        # --- END OF FIX ---
        
        full_response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # The model's output now includes the prompt, so we remove it.
        response_text = full_response.replace("Please transcribe the following audio. ", "").strip()

        logger.info(f"Generated response: '{response_text}'")
        return {"text": response_text}
        
    except subprocess.CalledProcessError as e:
        logger.error(f"FFMPEG conversion failed: {e.stderr}")
        raise HTTPException(status_code=400, detail=f"Audio conversion failed.")
    except Exception as e:
        logger.error(f"Error processing audio: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred while processing audio.")
