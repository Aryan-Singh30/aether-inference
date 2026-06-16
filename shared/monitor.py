import os
import sys
import gc
import logging
import psutil

logger = logging.getLogger("resource_monitor")

class SystemResourceMonitor:
    """Monitors system resource consumption (RAM and Sockets/File Descriptors).
    
    Implements cross-platform support for Windows (which uses Handles)
    and Linux/macOS (which use File Descriptors).
    """
    
    def __init__(self, memory_limit_mb: float = 2048.0, fd_limit: int = 150):
        self.process = psutil.Process(os.getpid())
        self.memory_limit_mb = memory_limit_mb
        self.fd_limit = fd_limit

    def get_memory_usage_mb(self) -> float:
        """Returns the Resident Set Size (RSS) memory of the current process in MB."""
        # rss is the actual physical memory (RAM) being used by this Python process
        bytes_used = self.process.memory_info().rss
        return bytes_used / (1024.0 * 1024.0)

    def get_file_descriptor_count(self) -> int:
        """Returns the number of active open file descriptors or OS handles.
        
        Why? On Linux, psutil supports Process.num_fds().
        On Windows, it throws AttributeError because Windows uses OS Handles.
        We check the operating system platform and call the correct system API.
        """
        if sys.platform == "win32":
            # num_handles() returns the total count of file, folder, and socket handles opened by Windows
            return self.process.num_handles()
        else:
            # num_fds() returns open file descriptors on Unix-like systems
            return self.process.num_fds()

    def check_limits(self) -> dict:
        """Checks if current process memory or file handles exceed configured safety thresholds.
        
        If memory exceeds the limit, it proactively triggers Python's
        garbage collector to clean up dereferenced buffers and stabilize RAM.
        """
        current_mem = self.get_memory_usage_mb()
        current_fds = self.get_file_descriptor_count()
        
        status = {
            "memory_mb": round(current_mem, 2),
            "handles_count": current_fds,
            "memory_ok": current_mem < self.memory_limit_mb,
            "handles_ok": current_fds < self.fd_limit,
            "action_taken": "none"
        }
        
        # 1. Proactive Memory Stabilization (similar to the 28.8 GiB to 5 GiB fix on your resume)
        if not status["memory_ok"]:
            logger.warning(
                f"Memory Alert! RAM usage ({current_mem:.1f} MB) exceeds limit ({self.memory_limit_mb} MB). "
                "Triggering proactive Garbage Collection..."
            )
            # gc.collect() forces Python to free up memory immediately
            collected = gc.collect()
            new_mem = self.get_memory_usage_mb()
            status["action_taken"] = f"gc_collected_{collected}_objects"
            status["memory_mb"] = round(new_mem, 2)
            status["memory_ok"] = new_mem < self.memory_limit_mb
            logger.info(f"RAM stabilized at {new_mem:.1f} MB after GC cleanup.")

        # 2. File Descriptor leak warning
        if not status["handles_ok"]:
            logger.critical(
                f"FD Leak Warning! Process is holding {current_fds} open handles. "
                f"Safety limit is {self.fd_limit}. Potential socket leak detected!"
            )
            
        return status

# Simple manual test if run directly
if __name__ == "__main__":
    monitor = SystemResourceMonitor(memory_limit_mb=100.0, fd_limit=100)
    metrics = monitor.check_limits()
    print("System Resource Metrics:")
    print(f"- Current RAM: {metrics['memory_mb']} MB (Limit: {monitor.memory_limit_mb} MB)")
    print(f"- Current Open Handles: {metrics['handles_count']} (Limit: {monitor.fd_limit})")
    print(f"- Action: {metrics['action_taken']}")
