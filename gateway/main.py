import asyncio
import json
import os
import httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import logging
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [Gateway] - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI()

# RunPod service URLs - modify these for your RunPod setup
ORPHEUS_URL = os.getenv("ORPHEUS_URL", "http://localhost:8000")
VOXTRAL_URL = os.getenv("VOXTRAL_URL", "http://localhost:8002")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

manager = ConnectionManager()

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await manager.connect(websocket)
    logger.info(f"Client {session_id} connected. Total clients: {len(manager.active_connections)}")
    
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)

            if message.get("type") == "audio":
                logger.info(f"Received audio from {session_id}. Processing...")
                
                start_time = time.time()
                
                # Use timeout for HTTP requests
                timeout = httpx.Timeout(120.0)  # 2 minutes timeout
                
                async with httpx.AsyncClient(timeout=timeout) as client:
                    try:
                        # Step 1: Send audio to Voxtral for transcription
                        logger.info("Sending audio to Voxtral service...")
                        voxtral_payload = {"audio": message["audio"]}
                        
                        voxtral_resp = await client.post(
                            f"{VOXTRAL_URL}/process", 
                            json=voxtral_payload
                        )
                        voxtral_resp.raise_for_status()
                        
                        transcription = voxtral_resp.json()["text"]
                        logger.info(f"Transcription: '{transcription}'")
                        
                        # Step 2: Send transcription to Orpheus for TTS
                        logger.info("Sending text to Orpheus service...")
                        orpheus_payload = {
                            "text": transcription, 
                            "voice": message.get("voice", "tara")
                        }
                        
                        orpheus_resp = await client.post(
                            f"{ORPHEUS_URL}/synthesize", 
                            json=orpheus_payload
                        )
                        orpheus_resp.raise_for_status()
                        
                        audio_b64 = orpheus_resp.json()["audio"]
                        
                        end_time = time.time()
                        processing_time = end_time - start_time
                        
                        logger.info(f"Complete pipeline took {processing_time:.2f}s")
                        
                        # Step 3: Send response back to client
                        await websocket.send_text(json.dumps({
                            "type": "audio_response",
                            "text": transcription,
                            "audio": audio_b64,
                            "processing_time": processing_time
                        }))
                        
                    except httpx.HTTPStatusError as e:
                        error_msg = f"HTTP {e.response.status_code}: {e.response.text}"
                        logger.error(f"HTTP error for {session_id}: {error_msg}")
                        await websocket.send_text(json.dumps({
                            "type": "error", 
                            "message": f"Service error: {error_msg}"
                        }))
                        
                    except httpx.TimeoutException:
                        logger.error(f"Request timeout for {session_id}")
                        await websocket.send_text(json.dumps({
                            "type": "error", 
                            "message": "Request timeout. Please try again."
                        }))
                        
    except WebSocketDisconnect:
        logger.info(f"Client {session_id} disconnected.")
    except Exception as e:
        logger.error(f"Unexpected error for {session_id}: {e}", exc_info=True)
        try:
            await websocket.send_text(json.dumps({
                "type": "error", 
                "message": f"Server error: {str(e)}"
            }))
        except:
            pass  # Connection might be closed
    finally:
        manager.disconnect(websocket)
        logger.info(f"Cleaned up connection for {session_id}")

@app.get("/")
async def get_home():
    """Serve the main HTML interface."""
    try:
        with open("static/client.html", "r") as f:
            return HTMLResponse(f.read())
    except FileNotFoundError:
        return HTMLResponse(
            "<h1>Client not found</h1><p>Ensure client.html exists in static/</p>", 
            status_code=404
        )

@app.get("/health")
async def health_check():
    """Gateway health check."""
    return {
        "status": "healthy",
        "orpheus_url": ORPHEUS_URL,
        "voxtral_url": VOXTRAL_URL,
        "active_connections": len(manager.active_connections)
    }
