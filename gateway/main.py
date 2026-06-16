import asyncio
import json
import logging
import random
import time
from contextlib import asynccontextmanager
import aio_pika
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
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
    # Startup: Initialize RabbitMQ connection and chaos parameters
    logger.info("Initializing background services...")
    app.state.chaos_rabbitmq = False
    app.state.chaos_mem_leak = 0.0
    app.state.chaos_handles_leak = 0
    
    # 1. Start background worker subprocess if configured (for Render single-service free tier)
    import os
    if os.getenv("START_BACKGROUND_WORKER", "false").lower() == "true":
        logger.info("Production configuration: Launching background worker process...")
        import subprocess
        import sys
        try:
            env = os.environ.copy()
            env["PYTHONPATH"] = os.getcwd()
            app.state.worker_process = subprocess.Popen(
                [sys.executable, "workers/manager.py"],
                env=env
            )
            logger.info("Background worker process launched successfully.")
        except Exception as e:
            logger.error(f"Failed to launch background worker process: {e}")

    try:
        # Connect to RabbitMQ using environment credentials or local fallback
        rabbitmq_url = os.getenv("RABBITMQ_URL") or os.getenv("CLOUDAMQP_URL") or "amqp://guest:guest@localhost:5672/"
        connection = await aio_pika.connect_robust(
            rabbitmq_url,
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
    
    # Shutdown: Clean up background tasks, worker processes, and connections
    logger.info("Shutting down background services...")
    telemetry_task.cancel()
    try:
        await telemetry_task
    except asyncio.CancelledError:
        pass

    # Clean up the worker process if it was started
    if hasattr(app.state, "worker_process"):
        logger.info("Stopping background worker process...")
        app.state.worker_process.terminate()
        try:
            app.state.worker_process.wait(timeout=5)
        except Exception:
            app.state.worker_process.kill()
        logger.info("Background worker process stopped.")

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
    if getattr(app.state, "chaos_rabbitmq", False):
        raise ConnectionError("Chaos simulation: RabbitMQ connection lost.")
    await app.state.rabbitmq_channel.default_exchange.publish(
        aio_pika.Message(
            body=json.dumps(payload).encode(),
            delivery_mode=aio_pika.DeliveryMode.PERSISTENT  # Makes task persistent on disk
        ),
        routing_key="inference_tasks"
    )

# Mock publish function for when RabbitMQ is offline, to support circuit breaker simulation
async def _mock_publish(payload: dict):
    if getattr(app.state, "chaos_rabbitmq", False):
        raise ConnectionError("Chaos simulation: Mock queue connection lost.")
    await asyncio.sleep(0.01)  # Simulate small queue processing delay

# 2. Submit Task (Sends jobs to the background queue)
@app.post("/submit")
async def submit_task(request: TaskRequest):
    logger.info(f"Received task request: query='{request.query}', type='{request.task_type}'")
    
    # Run active resource limit checks including simulated chaos leaks
    current_mem = resource_monitor.get_memory_usage_mb() + getattr(app.state, "chaos_mem_leak", 0.0)
    if current_mem >= resource_monitor.memory_limit_mb:
        logger.warning(f"Rejecting task submission due to critical server memory limits. Current: {current_mem:.1f} MB, Limit: {resource_monitor.memory_limit_mb} MB")
        raise HTTPException(status_code=503, detail="Server is overloaded. Please try again later.")
    
    task_id = f"task_{random.randint(100000, 999999)}"
    task_payload = {
        "task_id": task_id,
        "query": request.query,
        "task_type": request.task_type
    }
    
    is_connected = hasattr(app.state, "rabbitmq_status") and app.state.rabbitmq_status == "connected"
    publish_func = _rabbitmq_publish if is_connected else _mock_publish
    status_success = "queued" if is_connected else "queued_mock"
    message_success = "Task queued successfully in RabbitMQ." if is_connected else "Task received and queued successfully (Mock Mode - RabbitMQ Offline)."
    
    try:
        # Wrap the publish call in our Circuit Breaker
        await circuit_breaker.call(publish_func, task_payload)
        logger.info(f"Published task {task_id} successfully (connected={is_connected}).")
        return {
            "status": status_success,
            "task_id": task_id,
            "message": message_success,
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
        # If a generic exception occurred, the circuit breaker recorded the failure.
        # We fail-fast with a 503 error for this request.
        raise HTTPException(status_code=503, detail="Task queue service is currently unavailable.")

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
            
            # RAM usage of our Python process in MB (with injected chaos leak)
            mem_mb = resource_monitor.get_memory_usage_mb() + getattr(app.state, "chaos_mem_leak", 0.0)
            
            # Count of active open file descriptor handles (with injected chaos leak)
            active_handles = resource_monitor.get_file_descriptor_count() + getattr(app.state, "chaos_handles_leak", 0)
            
            # Read circuit breaker state (works in both real queue and mock queue modes)
            circuit_state = circuit_breaker.state
            
            # Get actual queue depth from RabbitMQ if connected
            queue_depth = 0
            if hasattr(app.state, "rabbitmq_status") and app.state.rabbitmq_status == "connected":
                try:
                    queue = await app.state.rabbitmq_channel.declare_queue("inference_tasks", passive=True)
                    queue_depth = queue.declaration_result.message_count
                except Exception:
                    pass
            
            # Pack actual stats to broadcast
            stats = {
                "cpu_usage": round(cpu_usage, 1),
                "memory_usage": round(mem_mb, 2),                  # in MB
                "active_fds": active_handles,                      # Real active system handle count + leak
                "queue_depth": queue_depth,                        # Read dynamically from RabbitMQ channels
                "average_latency_ms": round(random.uniform(8.0, 15.0), 1), # Simulated latencies
                "circuit_breaker_state": circuit_state
            }
            
            await manager.broadcast_json(stats)
            
        except Exception as e:
            logger.error(f"Error gathering telemetry data: {e}")
            
        # Stream telemetry data every 1 second
        await asyncio.sleep(1.0)

# --- Chaos Simulation Endpoints ---

@app.post("/chaos/trip")
async def chaos_trip():
    """Simulate RabbitMQ queue connection failure."""
    app.state.chaos_rabbitmq = True
    logger.warning("Chaos Command: Simulating RabbitMQ queue failure.")
    return {"message": "RabbitMQ connection failure simulated."}

@app.post("/chaos/leak")
async def chaos_leak():
    """Inject a memory leak (250MB) and handle leak (20) to simulate resource exhaustion."""
    app.state.chaos_mem_leak += 250.0
    app.state.chaos_handles_leak += 20
    logger.warning(
        f"Chaos Command: Injected leaks. "
        f"Total simulated leaks: Memory={app.state.chaos_mem_leak}MB, Handles={app.state.chaos_handles_leak}"
    )
    return {
        "message": "Leak injected.",
        "simulated_memory_leak_mb": app.state.chaos_mem_leak,
        "simulated_handles_leak": app.state.chaos_handles_leak
    }

@app.post("/chaos/reset")
async def chaos_reset():
    """Reset all chaos variables and return circuit breaker to closed state."""
    app.state.chaos_rabbitmq = False
    app.state.chaos_mem_leak = 0.0
    app.state.chaos_handles_leak = 0
    
    # Force reset circuit breaker state
    circuit_breaker.state = "CLOSED"
    circuit_breaker.failure_count = 0
    circuit_breaker.last_state_change = time.time()
    
    logger.info("Chaos Command: Resetting system health. Circuit Breaker restored to CLOSED.")
    return {"message": "System health restored to healthy defaults."}

# Mount the frontend static files at the very bottom
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
