import asyncio
import websockets
import json
from datetime import datetime
import redis

# Redis connector for publishing tracks to dashboard
r = redis.Redis(host='localhost', port=6379, db=0)

async def connect_ais_stream():
    """
    Connects to the AISStream.io websocket feed.
    To use this, the user needs an API key from https://aisstream.io.
    """
    API_KEY = "c4f835a669440f486659fb994ef6cbdbe9a7ec38" # Placeholder
    # We should use environment variables for this.
    
    # Subscribe to vessels in the US West Coast (approx)
    # Lat: 30 to 50, Lon: -130 to -115
    subscribe_message = {
        "APIKey": API_KEY,
        "BoundingBoxes": [[[-130, 30], [-115, 50]]],
        "FilterMessageTypes": ["PositionReport"]
    }

    uri = "wss://stream.aisstream.io/v0/stream"
    
    print(f"Connecting to AIS live stream at {uri}...")
    
    try:
        async with websockets.connect(uri) as websocket:
            await websocket.send(json.dumps(subscribe_message))
            
            async for message in websocket:
                data = json.loads(message)
                
                # Extract vessel data
                message_type = data.get("MessageType")
                if message_type == "PositionReport":
                    msg = data["MetaData"]
                    mmsi = msg["MMSI"]
                    ship_name = msg["ShipName"]
                    latitude = msg["Latitude"]
                    longitude = msg["Longitude"]
                    
                    # Store track point
                    track_point = {
                        "mmsi": mmsi,
                        "name": ship_name,
                        "lat": latitude,
                        "lon": longitude,
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    # Publish to Redis channel
                    r.publish('vessel_tracks', json.dumps(track_point))
                    # print(f"Track push: MMSI {mmsi} at ({latitude}, {longitude})")

    except Exception as e:
        print(f"AIS stream error: {e}")

if __name__ == "__main__":
    if REDIS_AVAILABLE := r.ping():
        print("Redis connection OK.")
        asyncio.run(connect_ais_stream())
    else:
        print("Redis not available. Please start redis-server.")
