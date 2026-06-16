import asyncio
import json
import logging
import random
from contextlib import asynccontextmanager
import aio_pika
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import our custom resiliency components from the shared folder
from shared.breaker import CircuitBreaker, CircuitBreakerOpenException
from shared.monitor import SystemResourceMonitor

# Set up logging so we can see system actions in the console (crucial for production debugging)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Instantiate global Circuit Breaker and System Resource Monitor
# We configure the circuit breaker to trip open after 3 consecutive failures
circuit_breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=10.0)
# We set memory safety limit to 1024MB (1GB) and handle limit to 100
resource_monitor = SystemResourceMonitor(memory_limit_mb=1024.0, fd_limit=100)

# Lifespan Context Manager handles startup and shutdown logic cleanly in modern FastAPI
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize RabbitMQ connection
    logger.info("Initializing background services...")
    
    try:
        # Connect to RabbitMQ using standard default local credentials
        connection = await aio_pika.connect_robust(
            "amqp://guest:guest@localhost:5672/",
            timeout=5  # Timeout quickly if RabbitMQ is not running
        )
        channel = await connection.channel()
        
        # Declare a durable queue. "Durable" means the queue survives RabbitMQ restarts
        await channel.declare_queue("inference_tasks", durable=True)
        
        # Store connection/channel in app.state so our API endpoints can access them
        app.state.rabbitmq_connection = connection
        app.state.rabbitmq_channel = channel
        app.state.rabbitmq_status = "connected"
        logger.info("Successfully connected to RabbitMQ.")
    except Exception as e:
        logger.warning(f"Could not connect to RabbitMQ (running in Mock Mode): {e}")
        app.state.rabbitmq_status = "disconnected"

    # Start the background telemetry loop
    telemetry_task = asyncio.create_task(telemetry_broadcaster())
    
    yield
    
    # Shutdown: Clean up background tasks and connections
    logger.info("Shutting down background services...")
    telemetry_task.cancel()
    try:
        await telemetry_task
    except asyncio.CancelledError:
        pass

    # Clean up the RabbitMQ connection if it was created
    if hasattr(app.state, "rabbitmq_connection") and app.state.rabbitmq_status == "connected":
        await app.state.rabbitmq_connection.close()
        logger.info("RabbitMQ connection closed.")

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

# Helper to perform the actual RabbitMQ publish, which our Circuit Breaker will execute
async def _rabbitmq_publish(payload: dict):
    await app.state.rabbitmq_channel.default_exchange.publish(
        aio_pika.Message(
            body=json.dumps(payload).encode(),
            delivery_mode=aio_pika.DeliveryMode.PERSISTENT  # Makes task persistent on disk
        ),
        routing_key="inference_tasks"
    )

# 2. Submit Task (Sends jobs to the background queue)
@app.post("/submit")
async def submit_task(request: TaskRequest):
    logger.info(f"Received task request: query='{request.query}', type='{request.task_type}'")
    
    # Run active resource limit checks
    metrics = resource_monitor.check_limits()
    if not metrics["memory_ok"]:
        logger.warning("Rejecting task submission due to critical server memory limits.")
        raise HTTPException(status_code=503, detail="Server is overloaded. Please try again later.")
    
    task_id = f"task_{random.randint(100000, 999999)}"
    task_payload = {
        "task_id": task_id,
        "query": request.query,
        "task_type": request.task_type
    }
    
    # Check if we have an active RabbitMQ connection
    if hasattr(app.state, "rabbitmq_status") and app.state.rabbitmq_status == "connected":
        try:
            # Wrap the RabbitMQ push inside our Circuit Breaker!
            # If RabbitMQ goes down, the circuit breaker will trip and switch to OPEN.
            await circuit_breaker.call(_rabbitmq_publish, task_payload)
            logger.info(f"Published task {task_id} to RabbitMQ queue 'inference_tasks'.")
            return {
                "status": "queued",
                "task_id": task_id,
                "message": "Task queued successfully in RabbitMQ.",
                "circuit_state": circuit_breaker.state
            }
        except CircuitBreakerOpenException as cbe:
            # The circuit breaker prevented the network call completely!
            logger.warning(f"Circuit Breaker blocked submission of task {task_id}: {cbe}")
            # Instead of failing with an error, we gracefully fallback to mock response (fail-fast resilience)
            return {
                "status": "queued_fallback",
                "task_id": task_id,
                "message": "System is degraded. Task queued successfully in Mock Mode (Circuit Breaker Tripped).",
                "circuit_state": circuit_breaker.state
            }
        except Exception as e:
            logger.error(f"Task submission failed: {e}")
            # If a generic network exception occurred, the circuit breaker recorded the failure.
            # We fail-fast with a 503 error for this request.
            raise HTTPException(status_code=503, detail="Task queue service is currently unavailable.")
    else:
        # Fallback Mock mode (runs if RabbitMQ was not detected during startup)
        logger.info(f"RabbitMQ is offline. Running task {task_id} in Mock Mode.")
        return {
            "status": "queued_mock",
            "task_id": task_id,
            "message": "Task received and queued successfully (Mock Mode - RabbitMQ Offline).",
            "circuit_state": "MOCK_MODE"
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
    
    # Initialize CPU monitoring interval by calling it once
    resource_monitor.process.cpu_percent(interval=None)
    
    while True:
        try:
            # Read ACTUAL system stats from the running process!
            cpu_usage = resource_monitor.process.cpu_percent(interval=None)
            
            # RAM usage of our Python process in MB
            mem_mb = resource_monitor.get_memory_usage_mb()
            
            # Count of active open file descriptor handles (Windows/Linux check)
            active_handles = resource_monitor.get_file_descriptor_count()
            
            # Read circuit breaker state
            circuit_state = circuit_breaker.state if app.state.rabbitmq_status == "connected" else "MOCK_MODE"
            
            # Pack actual stats to broadcast
            stats = {
                "cpu_usage": round(cpu_usage, 1),
                "memory_usage": round(mem_mb, 2),                  # in MB
                "active_fds": active_handles,                      # Real active system handle count
                "queue_depth": 0,                                  # Will be read from RabbitMQ channels in later phases
                "average_latency_ms": round(random.uniform(8.0, 15.0), 1), # Simulated latencies
                "circuit_breaker_state": circuit_state
            }
            
            await manager.broadcast_json(stats)
            
        except Exception as e:
            logger.error(f"Error gathering telemetry data: {e}")
            
        # Stream telemetry data every 1 second
        await asyncio.sleep(1.0)

# Note: Background telemetry is now managed via the lifespan handler above
