import time
import logging
import numpy as np
import faiss
import redis
from sentence_transformers import SentenceTransformer

logger = logging.getLogger("semantic_cache")

class SemanticCache:
    """A semantic vector cache utilizing FAISS for indexing and Redis for storage.
    
    Falls back gracefully to a local in-memory dictionary cache if Redis is offline.
    """
    
    def __init__(self, distance_threshold: float = 0.95, redis_url: str = "redis://localhost:6379"):
        self.threshold = distance_threshold
        
        # 1. Load a fast, lightweight embedding model (all-MiniLM-L6-v2 generates 384-dimension vectors)
        logger.info("Loading local embedding model (all-MiniLM-L6-v2)...")
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        
        # 2. Initialize FAISS index for Cosine Similarity (IndexFlatIP matches Inner Product)
        # 384 is the vector size produced by all-MiniLM-L6-v2
        self.index = faiss.IndexFlatIP(384)
        
        # 3. Setup Redis connection with fallback
        try:
            logger.info(f"Connecting to Redis at {redis_url}...")
            self.redis_client = redis.from_url(redis_url, socket_connect_timeout=2)
            # Ping to verify active connection
            self.redis_client.ping()
            self.use_redis = True
            logger.info("Successfully connected to Redis.")
        except Exception as e:
            logger.warning(f"Could not connect to Redis: {e}. Falling back to Local In-Memory Cache.")
            self.use_redis = False
            self.local_cache = {}  # Fallback dictionary mapping vector ID to cached payload

        # Keep a list of actual query texts matching the FAISS vector IDs
        self.id_to_query = []

    def _get_embedding(self, text: str) -> np.ndarray:
        """Converts text into a normalized 1D float32 vector embedding."""
        embedding = self.embedding_model.encode([text])[0]
        # Normalize the vector (necessary for inner-product to represent Cosine Similarity)
        vector = np.array([embedding], dtype=np.float32)
        faiss.normalize_L2(vector)
        return vector[0]

    def query(self, text: str) -> tuple[dict | None, float | None]:
        """Checks if a semantically similar query exists in the cache.
        
        Returns:
            A tuple of (cached_result_dict or None, cosine_similarity or None)
        """
        # If the FAISS index is empty, we immediately miss
        if self.index.ntotal == 0:
            return None, None
            
        start_time = time.perf_counter()
        
        # 1. Generate normalized embedding for the incoming query
        query_vector = self._get_embedding(text)
        query_grid = np.array([query_vector], dtype=np.float32)
        
        # 2. Search FAISS index for the k=1 nearest neighbor
        # distances is a list of similarity scores (1.0 = identical, 0.0 = orthogonal)
        # indices contains the vector IDs in the index
        distances, indices = self.index.search(query_grid, k=1)
        
        closest_distance = float(distances[0][0])
        closest_index = int(indices[0][0])
        
        # FAISS returns -1 if no vector is found
        if closest_index == -1:
            return None, None
            
        # 3. If similarity exceeds our threshold (e.g. 0.95), we have a Semantic Hit!
        if closest_distance >= self.threshold:
            logger.info(f"Semantic Cache HIT! Similarity: {closest_distance:.4f} (Threshold: {self.threshold})")
            
            # Fetch cache payload by ID
            if self.use_redis:
                try:
                    cached_data = self.redis_client.get(f"cache:{closest_index}")
                    if cached_data:
                        return json.loads(cached_data.decode("utf-8")), closest_distance
                except Exception as e:
                    logger.error(f"Failed to read from Redis cache: {e}")
            else:
                # Use local dictionary fallback
                cached_data = self.local_cache.get(closest_index)
                if cached_data:
                    return cached_data, closest_distance
                    
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        logger.info(f"Semantic Cache MISS. Lookup latency: {latency_ms:.2f}ms")
        return None, None

    def add(self, text: str, result: dict):
        """Adds a new query and its output result into the FAISS index and cache store."""
        try:
            # 1. Generate normalized embedding
            vector = self._get_embedding(text)
            vector_grid = np.array([vector], dtype=np.float32)
            
            # 2. Add vector to FAISS index. It automatically returns a new sequential ID.
            # In FAISS, this ID is equal to index.ntotal prior to addition.
            vector_id = self.index.ntotal
            self.index.add(vector_grid)
            
            # Record query mapping
            self.id_to_query.append(text)
            
            # 3. Save result payload under the ID key
            if self.use_redis:
                try:
                    self.redis_client.set(f"cache:{vector_id}", json.dumps(result))
                    logger.info(f"Stored task result in Redis cache at ID {vector_id}.")
                except Exception as e:
                    logger.error(f"Failed to write to Redis: {e}")
            else:
                self.local_cache[vector_id] = result
                logger.info(f"Stored task result in Local Cache at ID {vector_id}.")
                
        except Exception as e:
            logger.error(f"Failed to add item to Semantic Cache: {e}")
            
# Simple manual test if run directly
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    cache = SemanticCache(distance_threshold=0.90)
    
    # Add dummy results
    query_1 = "Check for spleen tissue volume"
    result_1 = {"predictions": [0.95, 0.05], "latency_ms": 12.5, "model": "onnx"}
    cache.add(query_1, result_1)
    
    # Test identical query (should hit)
    print("\n--- Test 1: Identical query ---")
    hit_data, sim = cache.query("Check for spleen tissue volume")
    print(f"Result: {hit_data}, Similarity: {sim}")
    
    # Test semantically similar query (should hit)
    print("\n--- Test 2: Semantically similar query ---")
    hit_data, sim = cache.query("Spleen tissue volume check")
    print(f"Result: {hit_data}, Similarity: {sim}")
    
    # Test completely unrelated query (should miss)
    print("\n--- Test 3: Unrelated query ---")
    hit_data, sim = cache.query("What is the weather today?")
    print(f"Result: {hit_data}, Similarity: {sim}")
