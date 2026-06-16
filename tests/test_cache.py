import pytest
from workers.semantic_cache import SemanticCache

def test_semantic_cache_initialization():
    """Verify that the semantic cache initializes its components correctly."""
    # We disable Redis for unit tests to ensure they are fully self-contained
    cache = SemanticCache(distance_threshold=0.90, redis_url="redis://localhost:9999")
    
    assert cache.index is not None
    assert cache.index.ntotal == 0
    assert cache.embedding_model is not None
    assert cache.use_redis is False  # Should fail and fallback to local in-memory

def test_semantic_cache_hit_and_miss():
    """Test exact hit, semantic hit, and cache miss scenarios."""
    cache = SemanticCache(distance_threshold=0.92, redis_url="redis://localhost:9999")
    
    # Add an entry to the cache
    original_query = "Query about spleen segmentation results"
    payload = {"predictions": [0.1, 0.9], "latency_ms": 15.0}
    cache.add(original_query, payload)
    
    assert cache.index.ntotal == 1
    
    # 1. Test Exact Hit
    hit_payload, similarity = cache.query(original_query)
    assert hit_payload is not None
    assert hit_payload["latency_ms"] == 15.0
    assert similarity >= 0.99  # Should be practically identical
    
    # 2. Test Semantic Hit (Different wording, same meaning)
    similar_query = "Results of the spleen segmentation query"
    hit_payload_2, similarity_2 = cache.query(similar_query)
    assert hit_payload_2 is not None
    assert hit_payload_2["predictions"] == [0.1, 0.9]
    assert similarity_2 >= 0.92  # Cosine similarity should cross the threshold
    
    # 3. Test Cache Miss (Completely different query)
    unrelated_query = "How to write a binary search tree in C++"
    miss_payload, similarity_3 = cache.query(unrelated_query)
    assert miss_payload is None
    assert similarity_3 is None
