# Performance Guide

This guide covers performance optimization strategies, benchmarking, and scalability considerations for Agent Diagnostician.

## 🎯 Performance Targets

### Latency Goals

| Operation Type | Target Latency | Typical Range |
|---|---|---|
| Simple detection (rules only) | < 100ms | 50-150ms |
| With embedding analysis | < 500ms | 200-800ms |
| With LLM fallback | < 5s | 2-10s |
| Batch processing | > 10 traces/sec | 5-20 traces/sec |

### Memory Goals

| Component | Target Usage | Notes |
|---|---|---|
| Baseline library | < 100MB | Without large models |
| Per trace processing | < 50MB | Additional memory |
| Embedding cache | Configurable | Default 10k entries |
| LLM context | Variable | Depends on trace size |

### Accuracy vs Speed Tradeoffs

| Detection Mode | Speed | Accuracy | Use Case |
|---|---|---|---|
| Rules only | Fastest | 70-80% | Real-time monitoring |
| Rules + Embeddings | Medium | 85-90% | Most production cases |
| Full LLM pipeline | Slowest | 90-95% | Detailed analysis |

## ⚡ Optimization Strategies

### 1. Selective Detection

Run only the detectors you need:

```python
from agent_diagnostician import Classifier
from agent_diagnostician.models.enums import FailureType

# Fast: Only check tool usage
classifier = Classifier(enabled_detectors=[
    FailureType.TOOL_USE_FAILURE
])

# Comprehensive: All detectors (slower)
classifier = Classifier()  # All detectors enabled

# Custom subset for your use case
classifier = Classifier(enabled_detectors=[
    FailureType.TOOL_USE_FAILURE,
    FailureType.GOAL_SATISFACTION_FAILURE,
    FailureType.CONTEXT_LOSS
])
```

### 2. Embedding Optimization

#### Caching Strategy
```python
from agent_diagnostician.analysis.embeddings import CachedEmbeddingMatcher

# Configure embedding cache
matcher = CachedEmbeddingMatcher(
    model_name="all-MiniLM-L6-v2",  # Faster, smaller model
    cache_size=50000,               # Larger cache for better hit rate
    persistent_cache=True           # Disk-based cache across sessions
)

classifier = Classifier(embedding_matcher=matcher)
```

#### Model Selection
```python
# Performance vs Accuracy tradeoffs for embedding models:

# Fastest: all-MiniLM-L6-v2 (22MB, 384 dimensions)
matcher = EmbeddingMatcher("all-MiniLM-L6-v2")

# Balanced: all-mpnet-base-v2 (420MB, 768 dimensions) 
matcher = EmbeddingMatcher("all-mpnet-base-v2")

# Most accurate: all-roberta-large-v1 (1.34GB, 1024 dimensions)
matcher = EmbeddingMatcher("all-roberta-large-v1")
```

#### Batch Processing
```python
# Process multiple similarities at once
similarities = matcher.batch_similarity(
    query="task description",
    candidates=["tool1", "tool2", "tool3", "tool4"]
)
# More efficient than individual similarity calls
```

### 3. LLM Optimization

#### Model Selection
```python
from agent_diagnostician.analysis.llm_judge import GeminiLLMJudge

# Fastest: Gemini Flash
judge = GeminiLLMJudge(
    model="models/gemini-2.5-flash",
    temperature=0.1,
    max_tokens=1000  # Shorter responses
)

# Most capable: Gemini Pro (slower, more accurate)
judge = GeminiLLMJudge(
    model="models/gemini-pro",
    temperature=0.1
)
```

#### Request Optimization
```python
# Reduce LLM calls with smarter thresholds
detector.MIN_CONFIDENCE_THRESHOLD = 0.4  # Higher threshold = fewer LLM calls
detector.SIMILARITY_THRESHOLD = 0.3      # Lower threshold = catch more with embeddings
```

#### Async Processing
```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

class AsyncClassifier:
    def __init__(self, max_workers=4):
        self.classifier = Classifier()
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
    
    async def diagnose_async(self, trace):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor, 
            self.classifier.diagnose, 
            trace
        )
    
    async def diagnose_batch(self, traces):
        tasks = [self.diagnose_async(trace) for trace in traces]
        return await asyncio.gather(*tasks)

# Usage
async_classifier = AsyncClassifier(max_workers=8)
results = await async_classifier.diagnose_batch(traces)
```

### 4. Memory Optimization

#### Trace Preprocessing
```python
def optimize_trace_for_analysis(trace):
    """Reduce trace size while preserving analysis capability."""
    
    # Truncate very long outputs (keep semantic content)
    for step in trace.steps:
        if step.tool_output and len(str(step.tool_output)) > 10000:
            output_str = str(step.tool_output)
            step.tool_output = output_str[:5000] + "...[truncated]..." + output_str[-1000:]
    
    # Remove duplicate or irrelevant fields
    for step in trace.steps:
        if hasattr(step, 'raw_response'):
            delattr(step, 'raw_response')  # Remove large raw data
    
    return trace

# Usage
optimized_trace = optimize_trace_for_analysis(original_trace)
result = classifier.diagnose(optimized_trace)
```

#### Memory Monitoring
```python
import psutil
import gc

class MemoryAwareClassifier:
    def __init__(self, max_memory_mb=1000):
        self.classifier = Classifier()
        self.max_memory_mb = max_memory_mb
    
    def diagnose(self, trace):
        # Check memory before processing
        memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
        
        if memory_mb > self.max_memory_mb:
            gc.collect()  # Force garbage collection
            
        result = self.classifier.diagnose(trace)
        
        # Clean up after large analyses
        if len(trace.steps) > 50:
            gc.collect()
            
        return result
```

## 📊 Benchmarking

### Built-in Benchmarks

```bash
# Run performance benchmarks
pytest tests/performance/ --benchmark-only

# Detailed benchmark with memory profiling
pytest tests/performance/ --benchmark-only --benchmark-verbose

# Compare different configurations
pytest tests/performance/test_detector_speed.py --benchmark-compare
```

### Custom Benchmarking

```python
import time
from statistics import mean, median
from agent_diagnostician import Classifier

def benchmark_classifier(traces, num_runs=10):
    """Benchmark classifier performance on test traces."""
    
    classifier = Classifier()
    times = []
    
    for run in range(num_runs):
        start_time = time.time()
        
        for trace in traces:
            result = classifier.diagnose(trace)
        
        elapsed = time.time() - start_time
        times.append(elapsed)
    
    return {
        "mean_time": mean(times),
        "median_time": median(times),
        "min_time": min(times),
        "max_time": max(times),
        "traces_per_second": len(traces) / mean(times)
    }

# Usage
benchmark_results = benchmark_classifier(test_traces)
print(f"Processing speed: {benchmark_results['traces_per_second']:.1f} traces/sec")
```

### Memory Profiling

```python
from memory_profiler import profile

@profile
def diagnose_large_batch(traces):
    """Profile memory usage during batch processing."""
    classifier = Classifier()
    results = []
    
    for i, trace in enumerate(traces):
        result = classifier.diagnose(trace) 
        results.append(result)
        
        # Log memory usage every 100 traces
        if i % 100 == 0:
            print(f"Processed {i+1} traces")
    
    return results

# Run with: python -m memory_profiler your_script.py
```

## 🏗️ Scalability Patterns

### 1. Horizontal Scaling

#### Worker Pool Pattern
```python
from multiprocessing import Pool, Queue
from agent_diagnostician import Classifier

def worker_diagnose(trace):
    """Worker function for multiprocessing."""
    classifier = Classifier()
    return classifier.diagnose(trace)

def parallel_diagnose(traces, num_workers=4):
    """Process traces in parallel using worker pool."""
    
    with Pool(num_workers) as pool:
        results = pool.map(worker_diagnose, traces)
    
    return results

# Usage
results = parallel_diagnose(large_trace_batch, num_workers=8)
```

#### Distributed Processing
```python
import redis
import json
from celery import Celery

# Celery task for distributed processing
app = Celery('agent_diagnostician', broker='redis://localhost:6379')

@app.task
def diagnose_trace_task(trace_json):
    """Celery task for distributed trace analysis."""
    from agent_diagnostician import Classifier
    from agent_diagnostician.models.trace import AgentTrace
    
    trace = AgentTrace(**json.loads(trace_json))
    classifier = Classifier()
    result = classifier.diagnose(trace)
    
    return result.dict()

# Submit tasks
results = []
for trace in traces:
    task = diagnose_trace_task.delay(trace.json())
    results.append(task)

# Collect results
diagnoses = [task.get() for task in results]
```

### 2. Streaming Analysis

```python
import asyncio
from asyncio import Queue

class StreamingDiagnosticService:
    """Real-time streaming analysis of agent traces."""
    
    def __init__(self, buffer_size=100):
        self.classifier = Classifier()
        self.trace_queue = Queue(maxsize=buffer_size)
        self.result_callbacks = []
    
    def add_result_callback(self, callback):
        """Add callback for processing results."""
        self.result_callbacks.append(callback)
    
    async def process_stream(self):
        """Continuously process incoming traces."""
        while True:
            try:
                trace = await asyncio.wait_for(
                    self.trace_queue.get(), 
                    timeout=1.0
                )
                
                # Diagnose trace
                result = self.classifier.diagnose(trace)
                
                # Notify callbacks
                for callback in self.result_callbacks:
                    await callback(trace, result)
                
            except asyncio.TimeoutError:
                continue  # No traces to process
    
    async def submit_trace(self, trace):
        """Submit trace for analysis."""
        await self.trace_queue.put(trace)

# Usage
service = StreamingDiagnosticService()

async def handle_result(trace, result):
    if result.failure_type.value != "none":
        print(f"Alert: {trace.run_id} - {result.failure_type.value}")

service.add_result_callback(handle_result)

# Start processing
asyncio.create_task(service.process_stream())
```

### 3. Caching Strategies

#### Redis-based Caching
```python
import redis
import json
import hashlib
from agent_diagnostician import Classifier

class CachedClassifier:
    """Classifier with Redis-based result caching."""
    
    def __init__(self, redis_url="redis://localhost:6379", ttl=3600):
        self.classifier = Classifier()
        self.redis = redis.from_url(redis_url)
        self.ttl = ttl  # Cache TTL in seconds
    
    def _cache_key(self, trace):
        """Generate cache key from trace content."""
        trace_hash = hashlib.sha256(
            trace.json().encode()
        ).hexdigest()
        return f"diagnosis:{trace_hash}"
    
    def diagnose(self, trace):
        """Diagnose with caching."""
        cache_key = self._cache_key(trace)
        
        # Try cache first
        cached = self.redis.get(cache_key)
        if cached:
            return DetectionResult(**json.loads(cached))
        
        # Compute result
        result = self.classifier.diagnose(trace)
        
        # Cache result
        self.redis.setex(
            cache_key, 
            self.ttl, 
            result.json()
        )
        
        return result
```

#### Application-level Caching
```python
from functools import lru_cache
from typing import Tuple

class OptimizedClassifier:
    """Classifier with multi-level caching."""
    
    def __init__(self):
        self.classifier = Classifier()
        self._embedding_cache = {}
        self._llm_cache = {}
    
    @lru_cache(maxsize=10000)
    def _cached_similarity(self, text1: str, text2: str) -> float:
        """Cache embedding similarity calculations."""
        return self.classifier.embeddings.similarity(text1, text2)
    
    def diagnose(self, trace):
        """Diagnose with optimized caching."""
        # Use cached similarity calculations
        original_similarity = self.classifier.embeddings.similarity
        self.classifier.embeddings.similarity = self._cached_similarity
        
        try:
            result = self.classifier.diagnose(trace)
        finally:
            # Restore original method
            self.classifier.embeddings.similarity = original_similarity
        
        return result
```

## 🔧 Configuration Tuning

### Environment-based Configuration

```bash
# Performance tuning environment variables
export AGENT_DIAGNOSTICIAN_EMBEDDING_MODEL="all-MiniLM-L6-v2"  # Faster model
export AGENT_DIAGNOSTICIAN_CACHE_SIZE="50000"                  # Large cache
export AGENT_DIAGNOSTICIAN_LLM_TIMEOUT="10"                    # Shorter timeout
export AGENT_DIAGNOSTICIAN_BATCH_SIZE="10"                     # Batch processing
export AGENT_DIAGNOSTICIAN_MIN_CONFIDENCE="0.4"                # Higher threshold
```

### Runtime Configuration

```python
from agent_diagnostician import Classifier
from agent_diagnostician.config import Config

# Performance-optimized configuration
config = Config(
    # Embedding settings
    embedding_model="all-MiniLM-L6-v2",
    embedding_cache_size=50000,
    
    # LLM settings  
    llm_timeout=10,
    llm_max_retries=2,
    
    # Detection thresholds
    min_confidence_threshold=0.4,
    similarity_threshold=0.3,
    
    # Processing settings
    max_trace_size=100000,  # Truncate large traces
    enable_parallel_detection=True
)

classifier = Classifier(config=config)
```

## 📈 Monitoring & Metrics

### Performance Metrics Collection

```python
import time
from dataclasses import dataclass
from typing import Dict, List

@dataclass
class PerformanceMetrics:
    detector_times: Dict[str, float]
    total_time: float
    cache_hits: int
    cache_misses: int
    memory_peak_mb: float

class InstrumentedClassifier:
    """Classifier with performance monitoring."""
    
    def __init__(self):
        self.classifier = Classifier()
        self.metrics_history: List[PerformanceMetrics] = []
    
    def diagnose(self, trace) -> Tuple[DetectionResult, PerformanceMetrics]:
        """Diagnose with performance tracking."""
        import psutil
        
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Track individual detector times
        detector_times = {}
        
        # Instrument detector calls (simplified)
        result = self.classifier.diagnose(trace)
        
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        metrics = PerformanceMetrics(
            detector_times=detector_times,
            total_time=end_time - start_time,
            cache_hits=getattr(self.classifier, '_cache_hits', 0),
            cache_misses=getattr(self.classifier, '_cache_misses', 0),
            memory_peak_mb=max(start_memory, end_memory)
        )
        
        self.metrics_history.append(metrics)
        return result, metrics
    
    def get_performance_summary(self) -> Dict:
        """Get performance summary statistics."""
        if not self.metrics_history:
            return {}
        
        times = [m.total_time for m in self.metrics_history]
        memories = [m.memory_peak_mb for m in self.metrics_history]
        
        return {
            "avg_time": sum(times) / len(times),
            "max_time": max(times),
            "avg_memory_mb": sum(memories) / len(memories),
            "max_memory_mb": max(memories),
            "total_analyses": len(self.metrics_history)
        }
```

### Real-time Monitoring

```python
import threading
import time
from collections import deque

class PerformanceMonitor:
    """Real-time performance monitoring."""
    
    def __init__(self, window_size=100):
        self.window_size = window_size
        self.recent_times = deque(maxlen=window_size)
        self.recent_memory = deque(maxlen=window_size)
        self.lock = threading.Lock()
    
    def record_analysis(self, duration: float, memory_mb: float):
        """Record performance data point."""
        with self.lock:
            self.recent_times.append(duration)
            self.recent_memory.append(memory_mb)
    
    def get_current_stats(self) -> Dict:
        """Get current performance statistics."""
        with self.lock:
            if not self.recent_times:
                return {}
            
            return {
                "avg_time_recent": sum(self.recent_times) / len(self.recent_times),
                "max_time_recent": max(self.recent_times),
                "avg_memory_recent": sum(self.recent_memory) / len(self.recent_memory),
                "analyses_per_second": len(self.recent_times) / sum(self.recent_times)
            }
    
    def start_monitoring(self, interval=5):
        """Start periodic monitoring output."""
        def monitor_loop():
            while True:
                stats = self.get_current_stats()
                if stats:
                    print(f"Performance: {stats['analyses_per_second']:.1f} traces/sec, "
                          f"{stats['avg_time_recent']:.2f}s avg, "
                          f"{stats['avg_memory_recent']:.1f}MB avg")
                time.sleep(interval)
        
        thread = threading.Thread(target=monitor_loop, daemon=True)
        thread.start()
        return thread

# Usage
monitor = PerformanceMonitor()
monitor.start_monitoring()

# Record performance during analysis
start_time = time.time()
result = classifier.diagnose(trace)
duration = time.time() - start_time
monitor.record_analysis(duration, get_memory_usage())
```

## 🎯 Performance Best Practices

### 1. Choose the Right Detection Mode

```python
# Real-time monitoring: Fast, basic detection
fast_classifier = Classifier(enabled_detectors=[
    FailureType.TOOL_USE_FAILURE  # Rules-based, very fast
])

# Production analysis: Balanced speed/accuracy  
balanced_classifier = Classifier(enabled_detectors=[
    FailureType.TOOL_USE_FAILURE,
    FailureType.GOAL_SATISFACTION_FAILURE,
    FailureType.CONTEXT_LOSS
])

# Deep analysis: Comprehensive but slower
comprehensive_classifier = Classifier()  # All detectors
```

### 2. Optimize Trace Data

```python
# Minimize trace data before analysis
def optimize_trace(trace):
    # Keep only essential fields
    for step in trace.steps:
        # Truncate very long outputs
        if step.tool_output and len(str(step.tool_output)) > 5000:
            step.tool_output = str(step.tool_output)[:5000] + "..."
        
        # Remove debugging fields
        if hasattr(step, 'debug_info'):
            delattr(step, 'debug_info')
    
    return trace
```

### 3. Use Appropriate Batch Sizes

```python
# Optimal batch sizes for different scenarios
BATCH_SIZES = {
    "real_time": 1,        # Process immediately
    "near_real_time": 5,   # Small batches
    "batch_analysis": 50,  # Larger batches for efficiency
    "bulk_processing": 200 # Maximum efficiency
}

def process_traces(traces, mode="batch_analysis"):
    batch_size = BATCH_SIZES[mode]
    classifier = Classifier()
    
    for i in range(0, len(traces), batch_size):
        batch = traces[i:i+batch_size]
        results = [classifier.diagnose(trace) for trace in batch]
        yield results
```

### 4. Monitor Resource Usage

```python
import psutil
import warnings

class ResourceAwareClassifier:
    """Classifier that monitors and manages resource usage."""
    
    def __init__(self, max_memory_mb=2000, max_cpu_percent=80):
        self.classifier = Classifier()
        self.max_memory_mb = max_memory_mb
        self.max_cpu_percent = max_cpu_percent
    
    def diagnose(self, trace):
        # Check system resources
        memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        if memory_mb > self.max_memory_mb:
            warnings.warn(f"High memory usage: {memory_mb:.1f}MB")
        
        if cpu_percent > self.max_cpu_percent:
            warnings.warn(f"High CPU usage: {cpu_percent:.1f}%")
            # Consider reducing detection scope
            return self._fast_diagnose(trace)
        
        return self.classifier.diagnose(trace)
    
    def _fast_diagnose(self, trace):
        # Fallback to faster analysis under resource pressure
        fast_classifier = Classifier(enabled_detectors=[
            FailureType.TOOL_USE_FAILURE
        ])
        return fast_classifier.diagnose(trace)
```

---

## 🚀 Getting Started with Optimization

1. **Baseline Measurement**: Profile your current usage to identify bottlenecks
2. **Selective Detection**: Enable only the detectors you need
3. **Caching**: Implement embedding and result caching for repeated analyses
4. **Batch Processing**: Group traces for more efficient processing
5. **Monitor**: Track performance metrics to identify optimization opportunities

For more specific optimization guidance, see the [Contributing Guide](../CONTRIBUTING.md) or reach out via [GitHub Discussions](https://github.com/your-org/agent-diagnostician/discussions).