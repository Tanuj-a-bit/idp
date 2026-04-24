import asyncio
import json
import logging
import random
from datetime import datetime
from typing import List

import redis
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

app = FastAPI(title="Maritime Tracker Backend")

# Setup Redis client for listening to tracks from ingestion/tracking modules
r = redis.Redis(host='localhost', port=6379, db=0)

class TrackPoint(BaseModel):
    id: str # MMSI or Track ID
    name: str # Vessel name if known
    lat: float
    lon: float
    timestamp: str
    prediction: List[List[float]] = [] # [[lat, lon], ...] sequences for trajectory

# Active WebSocket connections
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception:
                # Clear stale connections
                pass

manager = ConnectionManager()

@app.get("/")
async def get_status():
    return {"status": "Maritime Tracker API Online", "vessel_count": 0}

@app.websocket("/ws/tracks")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        # We spawn a background listener for Redis messages for each client
        # Usually, one shared listener broadcasting to all clients is more efficient.
        while True:
            # Just keep connection alive or handle client-side commands
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

async def redis_listener():
    """
    Background task to forward Redis 'vessel_tracks' messages to WebSockets.
    """
    pubsub = r.pubsub()
    pubsub.subscribe('vessel_tracks')
    
    print("Redis listener started. Awaiting track data...")
    
    while True:
        message = pubsub.get_message()
        if message and message['type'] == 'message':
            track_data = json.loads(message['data'])
            # Broadcast to all connected clients
            await manager.broadcast(json.dumps(track_data))
        
        await asyncio.sleep(0.01) # Yield

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(redis_listener())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
