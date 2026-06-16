import asyncio
import hashlib
import json
import logging
import signal
import sys
import aio_pika
try:
    from workers.inference_engine import ONNXInferenceEngine
    from workers.semantic_cache import SemanticCache
except ModuleNotFoundError:
    from inference_engine import ONNXInferenceEngine
    from semantic_cache import SemanticCache

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
logger = logging.getLogger("worker_manager")

# Name of the queue we consume from
QUEUE_NAME = "inference_tasks"

def query_to_vector(query_str: str) -> list[float]:
    """Helper to convert a text query string into a 10-float vector.
    
    Why? Our ONNX model expects a numeric vector of shape (1, 10).
    We use MD5 hashing to consistently convert any input text (e.g. 'Is there a lesion?')
    into a repeating sequence of 10 deterministic numbers between -1.0 and 1.0.
    """
    hash_bytes = hashlib.md5(query_str.encode("utf-8")).digest()
    vector = []
    # Take 10 bytes and scale them between -1.0 and 1.0
    for i in range(10):
        val = hash_bytes[i % len(hash_bytes)]
        scaled_val = (val / 127.5) - 1.0
        vector.append(scaled_val)
    return vector

class AsyncWorker:
    def __init__(self):
        self.engine = None
        self.cache = None
        self.connection = None
        self.channel = None
        self.is_running = True

    async def start(self):
        """Starts the worker, loading the model, initializing the cache, and listening to the queue."""
        logger.info("Starting Async Worker...")
        
        # 1. Load the ONNX model once and initialize semantic cache
        try:
            self.engine = ONNXInferenceEngine()
            # Initialize the semantic cache with 95% similarity threshold
            self.cache = SemanticCache(distance_threshold=0.95)
        except Exception as e:
            logger.critical(f"Failed to initialize worker services: {e}")
            sys.exit(1)

        # 2. Connect to RabbitMQ
        import os
        rabbitmq_url = os.getenv("RABBITMQ_URL") or os.getenv("CLOUDAMQP_URL") or "amqp://guest:guest@localhost:5672/"
        logger.info(f"Connecting to RabbitMQ (URL: {rabbitmq_url})...")
        self.connection = await aio_pika.connect_robust(rabbitmq_url)
        self.channel = await self.connection.channel()
        
        # Limit the number of messages pre-fetched to 1 (fair dispatch)
        # This prevents one worker from hogging all the tasks if multiple workers are running.
        await self.channel.set_qos(prefetch_count=1)
        
        # Declare queue (matching the settings of the gateway)
        queue = await self.channel.declare_queue(QUEUE_NAME, durable=True)
        
        logger.info(f"Connected to RabbitMQ. Waiting for tasks on queue '{QUEUE_NAME}'...")
        
        # 3. Start consuming tasks
        # iterator waits for incoming messages asynchronously
        async with queue.iterator() as queue_iter:
            async for message in queue_iter:
                if not self.is_running:
                    break
                    
                # Process the message
                await self.process_message(message)

    async def process_message(self, message: aio_pika.IncomingMessage):
        """Handles an incoming message from the RabbitMQ queue."""
        # Use context manager to handle automatic 'ack' (acknowledgment) or 'nack'
        # If an unhandled exception occurs inside this block, it will 'nack' (requeue) the message automatically.
        async with message.process():
            try:
                # 1. Parse JSON payload
                payload = json.loads(message.body.decode("utf-8"))
                task_id = payload.get("task_id")
                query = payload.get("query", "")
                task_type = payload.get("task_type", "text")
                
                logger.info(f"Processing task {task_id} (Type: {task_type}) - Raw query: '{query}'")
                
                # 2. Check the Semantic Cache first!
                cached_data, similarity = self.cache.query(query)
                if cached_data:
                    predictions = cached_data.get("predictions")
                    latency = cached_data.get("latency_ms")
                    logger.info(
                        f"Task {task_id} resolved via Semantic Cache Hit! "
                        f"Similarity: {similarity:.4f}. Latency: {latency:.2f}ms. "
                        f"Predictions: {predictions}"
                    )
                    return
                
                # 3. Cache Miss: Convert text to numerical input features for our ONNX model
                input_vector = query_to_vector(query)
                
                # 4. Execute ONNX Runtime inference
                # To prevent blocking the async event loop with heavy calculations,
                # we run the CPU-bound inference in a separate thread.
                loop = asyncio.get_running_loop()
                predictions, latency = await loop.run_in_executor(
                    None,  # None uses the default ThreadPoolExecutor
                    self.engine.run_inference,
                    input_vector
                )
                
                # 5. Save the output to the Semantic Cache for future speedups
                cache_payload = {
                    "predictions": predictions,
                    "latency_ms": latency,
                    "model_source": "onnx"
                }
                self.cache.add(query, cache_payload)
                
                logger.info(
                    f"Task {task_id} completed via ONNX Runtime inference. "
                    f"ONNX Latency: {latency:.2f}ms. Output Predictions: {predictions}"
                )
                
            except json.JSONDecodeError:
                logger.error("Failed to parse JSON payload. Discarding invalid task.")
            except Exception as e:
                logger.error(f"Error occurred while running inference on task: {e}")
                # Re-raise to trigger automatic message rejection (requeueing)
                raise e

    async def stop(self):
        """Clean shutdown of the worker."""
        logger.info("Initiating graceful shutdown...")
        self.is_running = False
        if self.connection:
            await self.connection.close()
            logger.info("RabbitMQ connection closed.")

def main():
    # Setup worker
    worker = AsyncWorker()
    
    # Establish async loop
    loop = asyncio.get_event_loop()
    
    # Register shutdown signals (SIGINT from Ctrl+C, SIGTERM from Docker stop)
    def signal_handler():
        logger.info("Shutdown signal received.")
        # Schedule the clean shutdown coroutine on the running loop
        asyncio.create_task(worker.stop())
        
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, signal_handler)
        except NotImplementedError:
            # signal handlers on Windows might throw NotImplementedError in some Python runtimes,
            # so we catch and handle them with standard try/except blocks
            pass

    try:
        loop.run_until_complete(worker.start())
    except KeyboardInterrupt:
        # Fallback for Ctrl+C on Windows
        loop.run_until_complete(worker.stop())
    except Exception as e:
        logger.error(f"Unexpected worker crash: {e}")

if __name__ == "__main__":
    main()
