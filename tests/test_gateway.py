from fastapi.testclient import TestClient
# We import the FastAPI app instance from our gateway package
from gateway.main import app

# Create a TestClient which simulates making requests to our FastAPI server
client = TestClient(app)

def test_health_check():
    """Test that the health endpoint returns 200 OK and correct JSON structure."""
    response = client.get("/health")
    
    # Assert checks if the condition is True. If False, the test fails.
    assert response.status_code == 200
    
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "gateway"

def test_submit_task_success():
    """Test that sending a valid JSON payload returns status 'queued'."""
    payload = {
        "query": "Is there a spleen lesion?",
        "task_type": "image"
    }
    response = client.post("/submit", json=payload)
    
    assert response.status_code == 200
    
    data = response.json()
    assert data["status"].startswith("queued")
    assert "task_id" in data

def test_submit_task_validation_error():
    """Test that sending missing parameters (invalid schema) returns a 422 error."""
    # Sending empty JSON (missing 'query', which is required by our Pydantic model)
    response = client.post("/submit", json={})
    
    # 422 is the HTTP standard code for Unprocessable Entity (Validation Error)
    assert response.status_code == 422
