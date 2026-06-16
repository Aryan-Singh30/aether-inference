import os
import torch
import torch.nn as nn

class TinyModel(nn.Module):
    """A simple 2-layer Neural Network for demonstration.
    It takes an input of 10 float values (like symptoms or measurements)
    and predicts 2 classification outputs (like health states).
    """
    def __init__(self):
        super(TinyModel, self).__init__()
        self.fc1 = nn.Linear(10, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 2)
        
    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

def export_tiny_model():
    model = TinyModel()
    # Put the model in evaluation mode (turns off dropout, batchnorm, etc.)
    model.eval()
    
    # Create a dummy input (shape: batch_size=1, input_features=10)
    # This acts as a template for ONNX to map the input shapes
    dummy_input = torch.randn(1, 10)
    
    # Get the workers directory path
    model_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(model_dir, "model.onnx")
    
    print(f"Exporting PyTorch model to ONNX format at: {model_path}...")
    
    # Export the model using PyTorch's built-in ONNX exporter
    torch.onnx.export(
        model,
        dummy_input,
        model_path,
        export_params=True,             # Store the trained parameter weights inside the file
        opset_version=15,               # The ONNX version standard to use
        do_constant_folding=True,       # Optimize by pre-calculating constant operations
        input_names=["input"],          # Name of the input node
        output_names=["output"],        # Name of the output node
        # Allow the batch size to be dynamic (so we can process 1 request or 100 requests at once)
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"}
        }
    )
    print("ONNX model export completed successfully!")

if __name__ == "__main__":
    export_tiny_model()
