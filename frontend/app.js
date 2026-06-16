// Base URLs
const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
const gatewayHost = window.location.host || 'localhost:8000';
const WS_URL = `${wsProtocol}//${gatewayHost}/ws/telemetry`;
const HTTP_URL = `${window.location.protocol}//${gatewayHost}`;

// Telemetry state
let latencyHistory = [];
const MAX_CHART_POINTS = 20;

// Initialize WebSocket connection
let socket;

function connectWebSocket() {
    logToConsole("ws", "Connecting to telemetry WebSocket...");
    socket = new WebSocket(WS_URL);
    
    const wsIndicator = document.getElementById("ws-indicator");
    const connectionStatus = document.getElementById("connection-status");
    
    socket.onopen = () => {
        wsIndicator.className = "pulse-indicator online";
        connectionStatus.textContent = "Telemetry Online";
        logToConsole("success", "WebSocket connected successfully. Streaming telemetry...");
    };
    
    socket.onclose = (event) => {
        wsIndicator.className = "pulse-indicator offline";
        connectionStatus.textContent = "Telemetry Offline (Reconnecting...)";
        logToConsole("error", `WebSocket disconnected: ${event.reason || 'Server went offline'}. Retrying in 3s...`);
        // Retry connection after 3 seconds
        setTimeout(connectWebSocket, 3000);
    };
    
    socket.onerror = (error) => {
        logToConsole("error", "WebSocket connection error occurred.");
    };
    
    socket.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            updateDashboardMetrics(data);
        } catch (e) {
            console.error("Error parsing telemetry JSON:", e);
        }
    };
}

// Update DOM elements with live telemetry values
function updateDashboardMetrics(data) {
    // 1. CPU
    document.getElementById("metric-cpu").textContent = data.cpu_usage.toFixed(1);
    document.getElementById("bar-cpu").style.width = `${Math.min(data.cpu_usage, 100)}%`;
    
    // 2. Memory (RSS)
    document.getElementById("metric-mem").textContent = data.memory_usage.toFixed(2);
    // Scale bar based on a 1024MB safety limit
    const memPercent = Math.min((data.memory_usage / 1024.0) * 100, 100);
    const barMem = document.getElementById("bar-mem");
    barMem.style.width = `${memPercent}%`;
    if (memPercent > 90) {
        barMem.style.background = "var(--neon-red)";
    } else if (memPercent > 70) {
        barMem.style.background = "var(--neon-orange)";
    } else {
        barMem.style.background = "linear-gradient(90deg, var(--neon-cyan), var(--neon-purple))";
    }
    
    // 3. Sockets & Handles
    document.getElementById("metric-fds").textContent = data.active_fds;
    const fdPercent = Math.min((data.active_fds / 100.0) * 100, 100);
    const barFds = document.getElementById("bar-fds");
    barFds.style.width = `${fdPercent}%`;
    if (fdPercent > 85) {
        barFds.style.background = "var(--neon-red)";
    } else {
        barFds.style.background = "linear-gradient(90deg, var(--neon-cyan), var(--neon-purple))";
    }
    
    // 4. Queue Depth
    document.getElementById("metric-queue").textContent = data.queue_depth;
    document.getElementById("bar-queue").style.width = `${Math.min(data.queue_depth * 10, 100)}%`;
    
    // 5. Circuit Breaker Badge
    const badge = document.getElementById("circuit-badge");
    badge.textContent = data.circuit_breaker_state;
    badge.className = `badge ${data.circuit_breaker_state}`;
    
    // 6. Latency History & SVG Line Chart drawing
    const currentLatency = data.average_latency_ms;
    latencyHistory.push(currentLatency);
    if (latencyHistory.length > MAX_CHART_POINTS) {
        latencyHistory.shift();
    }
    
    drawLatencyChart(latencyHistory);
    
    // Update average latency text
    const sum = latencyHistory.reduce((a, b) => a + b, 0);
    const avg = sum / (latencyHistory.length || 1);
    document.getElementById("avg-latency-text").textContent = `${avg.toFixed(1)}ms`;
}

// Render dynamic SVG Line Chart
function drawLatencyChart(data) {
    const svgWidth = 500;
    const svgHeight = 200;
    const pointsCount = data.length;
    
    if (pointsCount < 2) return;
    
    // We scale the y-axis to represent a max of 40ms latency
    const maxVal = 40.0;
    
    let pathD = "";
    
    for (let i = 0; i < pointsCount; i++) {
        const x = (i / (pointsCount - 1)) * svgWidth;
        const val = Math.min(data[i], maxVal);
        // SVG coordinates start at 0,0 top-left, so we subtract from height
        const y = svgHeight - (val / maxVal) * svgHeight;
        
        if (i === 0) {
            pathD = `M ${x} ${y}`;
        } else {
            pathD += ` L ${x} ${y}`;
        }
    }
    
    // 1. Draw line stroke
    document.getElementById("chart-line").setAttribute("d", pathD);
    
    // 2. Draw filled gradient area under the line
    const areaD = pathD + ` L ${svgWidth} ${svgHeight} L 0 ${svgHeight} Z`;
    document.getElementById("chart-area").setAttribute("d", areaD);
}

// Helper to write event logs into the console box
function logToConsole(type, message) {
    const consoleBox = document.getElementById("console-logs");
    const timeStr = new Date().toLocaleTimeString();
    
    let label = "[SYSTEM]";
    let cssClass = "system";
    
    if (type === "ws") { label = "[WEBSOCKET]"; cssClass = "ws-in"; }
    else if (type === "api") { label = "[API GATEWAY]"; cssClass = "api-in"; }
    else if (type === "success") { label = "[SUCCESS]"; cssClass = "success"; }
    else if (type === "chaos") { label = "[CHAOS INCIDENT]"; cssClass = "chaos"; }
    else if (type === "error") { label = "[FAILURE]"; cssClass = "error"; }
    
    const line = document.createElement("div");
    line.className = `log-line ${cssClass}`;
    line.textContent = `[${timeStr}] ${label} ${message}`;
    
    consoleBox.appendChild(line);
    // Auto scroll to bottom
    consoleBox.scrollTop = consoleBox.scrollHeight;
}

// Setup Event Listeners
document.addEventListener("DOMContentLoaded", () => {
    // 1. Start WebSocket connection
    connectWebSocket();
    
    // 2. Task Submission Form
    const taskForm = document.getElementById("task-form");
    taskForm.addEventListener("submit", async (event) => {
        event.preventDefault();
        
        const queryInput = document.getElementById("task-query");
        const typeSelect = document.getElementById("task-type");
        const btnSubmit = document.getElementById("btn-submit");
        
        const query = queryInput.value.trim();
        const taskType = typeSelect.value;
        
        logToConsole("api", `Submitting task to API: "${query}"`);
        
        // Disable button while processing
        btnSubmit.disabled = true;
        
        try {
            const response = await fetch(`${HTTP_URL}/submit`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ query: query, task_type: taskType })
            });
            
            const result = await response.json();
            
            if (response.ok) {
                if (result.status === "queued_fallback") {
                    logToConsole("chaos", `DEGRADED: ${result.message} (Task ID: ${result.task_id})`);
                } else if (result.status === "queued_mock") {
                    logToConsole("system", `MOCK MODE: ${result.message} (Task ID: ${result.task_id})`);
                } else {
                    logToConsole("success", `QUEUED: ${result.message} (Task ID: ${result.task_id})`);
                }
                queryInput.value = ""; // Clear input on success
            } else {
                logToConsole("error", `Rejected (${response.status}): ${result.detail || 'Server overload / memory limit hit'}`);
            }
        } catch (error) {
            logToConsole("error", `Failed to contact gateway server: ${error.message}`);
        } finally {
            btnSubmit.disabled = false;
        }
    });
    
    // 3. Clear Console Button
    document.getElementById("clear-console").addEventListener("click", () => {
        const consoleBox = document.getElementById("console-logs");
        consoleBox.innerHTML = '<div class="log-line system">[SYSTEM] Console cleared. Waiting for event streams...</div>';
    });
    
    // 4. Chaos Sandbox Buttons
    document.getElementById("chaos-trip").addEventListener("click", async () => {
        logToConsole("chaos", "Simulating Chaos: Tripping Queue Connection...");
        try {
            await fetch(`${HTTP_URL}/chaos/trip`, { method: "POST" });
            logToConsole("success", "Chaos Injection: Queue connection set to offline.");
        } catch (e) {
            logToConsole("error", "Could not send chaos command to server.");
        }
    });

    document.getElementById("chaos-leak").addEventListener("click", async () => {
        logToConsole("chaos", "Simulating Chaos: Injecting RAM & File Handle leak...");
        try {
            await fetch(`${HTTP_URL}/chaos/leak`, { method: "POST" });
            logToConsole("success", "Chaos Injection: Leak started. Watch resource gauges spike!");
        } catch (e) {
            logToConsole("error", "Could not send chaos command to server.");
        }
    });

    document.getElementById("chaos-reset").addEventListener("click", async () => {
        logToConsole("system", "Recovering: Resetting circuit breaker and clearing leaks...");
        try {
            await fetch(`${HTTP_URL}/chaos/reset`, { method: "POST" });
            logToConsole("success", "System Recovery: Services restored to healthy defaults.");
        } catch (e) {
            logToConsole("error", "Could not send recovery command to server.");
        }
    });
});
