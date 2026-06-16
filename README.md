# AetherInference 🚀

A distributed, fault-tolerant deep learning inference engine built with Python `asyncio`, RabbitMQ, Redis, FAISS semantic cache, and ONNX Runtime.

This project simulates a production-ready, high-throughput model serving pipeline and includes an interactive real-time telemetry dashboard with a Chaos Engineering sandbox.

---

## Key Features

1. **High-Throughput Task Queueing**: Uses **RabbitMQ** to decouple heavy model inference from user-facing APIs, preventing event-loop freezing.
2. **ONNX Runtime Optimization**: Leverages **ONNX Runtime** for hardware-accelerated deep learning execution (segmentation/classification).
3. **Semantic Cache (FAISS + Redis)**: Embeds incoming text queries using `sentence-transformers` and checks similarity using a local **FAISS** index. If similarity is above 95%, it bypasses inference and reads the result from **Redis** under 10ms.
4. **Resiliency Middleware**: Custom client-side **Circuit Breaker** (Closed ⇄ Open ⇄ Half-Open states) to prevent cascading failures if workers or databases fail.
5. **Real-Time Telemetry & Chaos Panel**: A web-based dashboard showing live metrics (latencies, queue backlogs, memory consumption, active file descriptors) and triggers to inject database failures, worker crashes, or memory leaks to observe the system's auto-recovery.

---

## Technology Stack

* **Language**: Python 3.11+
* **Backend Framework**: FastAPI (Asynchronous)
* **Message Broker**: RabbitMQ (`aio-pika` async driver)
* **Vector Indexing**: FAISS (Facebook AI Similarity Search)
* **Cache Database**: Redis
* **Inference Engine**: ONNX Runtime
* **Containerization**: Docker & Docker Compose
* **Frontend**: HTML5, Vanilla CSS (Glassmorphism), Vanilla JavaScript, WebSockets
