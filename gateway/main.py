import asyncio
import json
import os
import httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import logging

# Configure logging to show timestamps and service name
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [Gateway] - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI()

# Get service URLs from environment variables set by docker-compose
ORPHEUS_URL = os.getenv("ORPHEUS_URL", "http://localhost:8001")
VOXTRAL_URL = os.getenv("VOXTRAL_URL", "http://localhost:8002")

# Mount the 'static' directory to serve client.html
app.mount("/static", StaticFiles(directory="static"), name="static")

class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
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

            # Use an async HTTP client to communicate with the other services
            async with httpx.AsyncClient(timeout=90.0) as client:
                if message.get("type") == "audio":
                    logger.info(f"Received audio from {session_id}. Forwarding to Voxtral service.")

                    # Step 1: Send audio to Voxtral Service for ASR
                    voxtral_payload = {"audio": message["audio"]}
                    voxtral_resp = await client.post(f"{VOXTRAL_URL}/process", json=voxtral_payload)
                    voxtral_resp.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
                    response_text = voxtral_resp.json()["text"]

                    logger.info(f"Voxtral returned text: '{response_text}'. Forwarding to Orpheus service.")

                    # Step 2: Send the generated text to Orpheus Service for TTS
                    orpheus_payload = {"text": response_text, "voice": message.get("voice", "tara")}
                    orpheus_resp = await client.post(f"{ORPHEUS_URL}/synthesize", json=orpheus_payload)
                    orpheus_resp.raise_for_status()
                    audio_b64 = orpheus_resp.json()["audio"]

                    logger.info(f"Orpheus returned audio. Sending final response to client {session_id}.")

                    # Step 3: Send the final result back to the client
                    await websocket.send_text(json.dumps({
                        "type": "audio_response",
                        "text": response_text,
                        "audio": audio_b64
                    }))

    except WebSocketDisconnect:
        logger.info(f"Client {session_id} disconnected.")
    except httpx.HTTPStatusError as e:
        error_detail = e.response.json().get("detail", e.response.text)
        logger.error(f"HTTP error communicating with a model service for client {session_id}: {error_detail}")
        await websocket.send_text(json.dumps({"type": "error", "message": f"A model service error occurred: {error_detail}"}))
    except Exception as e:
        logger.error(f"An unexpected error occurred with client {session_id}: {e}")
        await websocket.send_text(json.dumps({"type": "error", "message": f"A server error occurred: {str(e)}"}))
    finally:
        manager.disconnect(websocket)
        logger.info(f"Cleaned up connection for {session_id}. Total clients: {len(manager.active_connections)}")

@app.get("/")
async def get_home():
    """Serves the main HTML user interface."""
    try:
        with open("static/client.html", "r") as f:
            return HTMLResponse(f.read())
    except FileNotFoundError:
        return HTMLResponse("<h1>Client UI not found.</h1><p>Please ensure `gateway/static/client.html` exists.</p>", status_code=404)
