import pytest
from workers.inference_engine import ONNXInferenceEngine
from workers.manager import query_to_vector

def test_query_to_vector():
    """Verify that text query hashing generates exactly 10 floats scaled between -1.0 and 1.0."""
    query = "Test query for medical report search"
    vector = query_to_vector(query)
    
    assert isinstance(vector, list)
    assert len(vector) == 10
    
    # Assert all values are floats and scaled between -1.0 and 1.0
    for val in vector:
        assert isinstance(val, float)
        assert -1.0 <= val <= 1.0

def test_onnx_engine_initialization():
    """Test that the ONNX engine loads the exported model successfully."""
    engine = ONNXInferenceEngine()
    assert engine.session is not None
    assert engine.input_name == "input"
    assert engine.output_name == "output"

def test_onnx_inference_success():
    """Test running a valid inference request through the ONNX model."""
    engine = ONNXInferenceEngine()
    test_input = [0.1, 0.2, 0.3, 0.4, 0.5, -0.1, -0.2, -0.3, -0.4, -0.5]
    
    predictions, latency = engine.run_inference(test_input)
    
    # Verify predictions structure
    assert isinstance(predictions, list)
    assert len(predictions) == 2  # The model output dimension (TinyModel output features = 2)
    
    # Verify latency
    assert isinstance(latency, float)
    assert latency > 0.0

def test_onnx_inference_invalid_size():
    """Test that passing a vector of size other than 10 raises ValueError."""
    engine = ONNXInferenceEngine()
    bad_input = [1.0, 2.0, 3.0]  # Only 3 elements instead of 10
    
    # pytest.raises checks that the expected exception is thrown
    with pytest.raises(ValueError, match="Input vector must have exactly 10 features"):
        engine.run_inference(bad_input)
