import asyncio
import logging
import random
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Set up logging so we can see system actions in the console (crucial for production debugging)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Lifespan Context Manager handles startup and shutdown logic cleanly in modern FastAPI
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Start the background telemetry broadcaster
    logger.info("Initializing background services...")
    telemetry_task = asyncio.create_task(telemetry_broadcaster())
    yield
    # Shutdown: Clean up background tasks
    logger.info("Shutting down background services...")
    telemetry_task.cancel()
    try:
        await telemetry_task
    except asyncio.CancelledError:
        pass

# Initialize the FastAPI App with the lifespan manager
app = FastAPI(
    title="AetherInference Gateway",
    description="High-performance, async API gateway for handling and routing AI inference tasks.",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS (Cross-Origin Resource Sharing)
# Why? Web browsers block a website on one domain (like localhost:5500) from talking to an API on another (like localhost:8000).
# Adding CORS middleware permits our frontend dashboard to connect to the backend without safety blocks.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permits all websites (change this to specific domains in strict production environments)
    allow_credentials=True,
    allow_methods=["*"],  # Permits all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Permits all custom headers
)

# Define a Pydantic Model for incoming HTTP request data validation
# Why? Pydantic ensures the user sends exactly the structure we expect. If they send wrong data,
# FastAPI automatically returns a clean 422 validation error before it breaks our code.
class TaskRequest(BaseModel):
    query: str
    task_type: str = "text"  # Can be "text" (for QA RAG) or "image" (for segmentation)

# --- HTTP Endpoints ---

# 1. Health check (Very important for production hosts like Render to know the app is active and hasn't crashed)
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "gateway",
        "uptime_checks": "passing"
    }

# 2. Submit Task (Sends jobs to the background queue)
@app.post("/submit")
async def submit_task(request: TaskRequest):
    logger.info(f"Received task request: query='{request.query}', type='{request.task_type}'")
    
    # In Phase 3, we will push this task into the RabbitMQ queue.
    # For now (Phase 2), we simulate a successful task queueing and return a mock tracking ID.
    mock_task_id = f"task_{random.randint(100000, 999999)}"
    
    return {
        "status": "queued",
        "task_id": mock_task_id,
        "message": "Task received and queued successfully (Mock Mode)."
    }

# --- WebSocket Telemetry ---
# WebSockets keep a permanent open pipe between the client and server.
# This lets the backend stream live CPU, memory, and queue stats to the dashboard 10 times a second without the client asking repeatedly.

class ConnectionManager:
    """Manages active WebSocket connections."""
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"New client connected to telemetry stream. Total clients: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            logger.info(f"Client disconnected. Total clients: {len(self.active_connections)}")

    async def broadcast_json(self, data: dict):
        """Sends data to all connected clients."""
        for connection in self.active_connections:
            try:
                await connection.send_json(data)
            except Exception as e:
                # If a client closed the connection without telling us, catch the error and skip
                logger.error(f"Failed to send telemetry to a client: {e}")

manager = ConnectionManager()

@app.websocket("/ws/telemetry")
async def websocket_telemetry_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        # Keep the connection open. If the client sends messages (which we don't expect for telemetry), we just ignore them.
        while True:
            # Wait for any message from the client to prevent CPU spinning
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# --- Background Telemetry Task ---
# A background task that runs continuously to gather resource stats and send them to the frontend.
async def telemetry_broadcaster():
    logger.info("Starting background telemetry broadcaster...")
    while True:
        # In later phases, we will read active memory, CPU, queue depth, and circuit breaker status.
        # For now, we simulate fluctuations so we can test that the WebSocket connection is alive.
        mock_stats = {
            "cpu_usage": round(random.uniform(10.0, 35.0), 1),
            "memory_usage": round(random.uniform(4.2, 4.9), 2),  # in GB (matching your resume's ~5 GiB stabilization!)
            "active_fds": random.randint(35, 45),                # Simulated active socket file descriptors
            "queue_depth": random.randint(0, 3),                 # Backlog in RabbitMQ
            "average_latency_ms": round(random.uniform(8.0, 22.0), 1),
            "circuit_breaker_state": "CLOSED"                    # Closed = Normal, Open = Error State
        }
        
        await manager.broadcast_json(mock_stats)
        
        # Stream telemetry data every 1 second
        await asyncio.sleep(1.0)

# Note: Background telemetry is now managed via the lifespan handler above
