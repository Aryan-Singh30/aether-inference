import os
import time
import numpy as np
import onnxruntime as ort

class ONNXInferenceEngine:
    """Wrapper class to load and execute an ONNX model using ONNX Runtime."""
    
    def __init__(self, model_filename: str = "model.onnx"):
        # Locate the model file in the same folder as this script
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_path = os.path.join(current_dir, model_filename)
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"ONNX model file not found at: {self.model_path}. Run generate_model.py first.")
            
        print(f"Loading ONNX model from: {self.model_path}...")
        
        # Load the model session. This loads the model graph into memory once.
        # In production, we keep this session open rather than reloading it on every request.
        self.session = ort.InferenceSession(self.model_path)
        
        # Get input and output names from the model graph metadata
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        print(f"Model loaded. Inputs: '{self.input_name}', Outputs: '{self.output_name}'")

    def run_inference(self, input_vector: list[float]) -> tuple[list[float], float]:
        """Runs the model on the input vector.
        
        Args:
            input_vector: A list of 10 float values.
            
        Returns:
            A tuple of (predictions: list of floats, latency_ms: float)
        """
        if len(input_vector) != 10:
            raise ValueError(f"Input vector must have exactly 10 features, got {len(input_vector)}.")
            
        # 1. Convert input to a float32 NumPy array with a batch dimension (shape: 1 x 10)
        # Why? Neural networks process batches of inputs. We wrap our 10 inputs into a 2D grid.
        np_input = np.array([input_vector], dtype=np.float32)
        
        # 2. Measure execution time (latency)
        start_time = time.perf_counter()
        
        # 3. Run model inference through ONNX Runtime
        # The first argument is a list of output names we want to retrieve.
        # The second argument is a dictionary mapping input node names to raw NumPy arrays.
        outputs = self.session.run([self.output_name], {self.input_name: np_input})
        
        end_time = time.perf_counter()
        
        # Calculate latency in milliseconds
        latency_ms = (end_time - start_time) * 1000.0
        
        # Extract the resulting 1D array of output predictions
        predictions = outputs[0][0].tolist()
        
        return predictions, latency_ms

# Simple manual test if run directly
if __name__ == "__main__":
    try:
        engine = ONNXInferenceEngine()
        # Generate 10 random float numbers to simulate an input request
        test_input = [1.0, 2.0, -0.5, 0.2, 1.5, -2.1, 0.0, 0.5, 1.2, -1.0]
        results, latency = engine.run_inference(test_input)
        print(f"Test Input: {test_input}")
        print(f"Predictions: {results}")
        print(f"ONNX Latency: {latency:.4f} ms")
    except Exception as e:
        print(f"Error: {e}")
