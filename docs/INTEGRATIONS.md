# Framework Integrations

This guide shows how to integrate Agent Diagnostician with popular agent frameworks.

## LangChain

### Basic Integration

```python
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from agent_diagnostician import Classifier
from agent_diagnostician.models.trace import AgentTrace

def langchain_to_diagnostician(agent_executor, input_text, result):
    """Convert LangChain execution to Agent Diagnostician trace."""
    
    # Extract steps from intermediate_steps
    steps = []
    for i, (action, observation) in enumerate(result.get("intermediate_steps", [])):
        step = {
            "step_index": i,
            "tool_name": action.tool,
            "tool_input": action.tool_input,
            "tool_output": observation
        }
        
        # Add reasoning if available
        if hasattr(action, 'log') and action.log:
            step["thought"] = action.log
            
        steps.append(step)
    
    # Build trace
    trace_data = {
        "run_id": f"langchain_{hash(input_text)}",
        "task": input_text,
        "status": "success" if result.get("output") else "error",
        "total_steps": len(steps),
        "final_output": result.get("output"),
        "steps": steps
    }
    
    # Add available tools information
    if hasattr(agent_executor, 'tools'):
        available_tools = []
        for tool in agent_executor.tools:
            tool_spec = {
                "name": tool.name,
                "description": tool.description
            }
            # Add schema if available
            if hasattr(tool, 'args_schema') and tool.args_schema:
                tool_spec["schema"] = tool.args_schema.schema()
            available_tools.append(tool_spec)
        trace_data["available_tools"] = available_tools
    
    return AgentTrace(**trace_data)

# Usage example
@tool
def get_weather(city: str) -> str:
    """Get weather for a city."""
    return f"Weather in {city}: Sunny, 75°F"

# Set up agent
agent_executor = create_openai_tools_agent(llm, [get_weather], prompt)

# Run and diagnose
input_text = "What's the weather in Paris?"
result = agent_executor.invoke({"input": input_text})

# Convert to diagnostic format
trace = langchain_to_diagnostician(agent_executor, input_text, result)

# Diagnose
classifier = Classifier()
diagnosis = classifier.diagnose(trace)

print(f"Diagnosis: {diagnosis.failure_type.value}")
print(f"Confidence: {diagnosis.confidence_band.value}")
if diagnosis.fix_direction:
    print(f"Suggestion: {diagnosis.fix_direction}")
```

### Advanced LangChain Integration

```python
from langchain.callbacks.base import BaseCallbackHandler
from typing import Any, Dict, List, Optional
import uuid

class DiagnosticCallbackHandler(BaseCallbackHandler):
    """Callback handler to automatically collect diagnostic data."""
    
    def __init__(self):
        self.runs: Dict[str, Dict] = {}
        
    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], 
                      run_id: uuid.UUID, **kwargs) -> None:
        self.runs[str(run_id)] = {
            "task": inputs.get("input", ""),
            "steps": [],
            "start_time": time.time()
        }
    
    def on_tool_start(self, serialized: Dict[str, Any], input_str: str,
                     run_id: uuid.UUID, parent_run_id: Optional[uuid.UUID] = None, **kwargs) -> None:
        if parent_run_id:
            parent_id = str(parent_run_id)
            if parent_id in self.runs:
                step = {
                    "step_index": len(self.runs[parent_id]["steps"]),
                    "tool_name": serialized.get("name", "unknown"),
                    "tool_input": {"input": input_str},
                    "start_time": time.time()
                }
                self.runs[parent_id]["steps"].append(step)
    
    def on_tool_end(self, output: str, run_id: uuid.UUID, 
                   parent_run_id: Optional[uuid.UUID] = None, **kwargs) -> None:
        if parent_run_id:
            parent_id = str(parent_run_id)
            if parent_id in self.runs and self.runs[parent_id]["steps"]:
                self.runs[parent_id]["steps"][-1]["tool_output"] = output
    
    def on_chain_end(self, outputs: Dict[str, Any], run_id: uuid.UUID, **kwargs) -> None:
        run_data = self.runs.get(str(run_id))
        if run_data:
            run_data["final_output"] = outputs.get("output")
            run_data["status"] = "success"
            run_data["total_steps"] = len(run_data["steps"])
    
    def get_trace(self, run_id: str) -> Optional[AgentTrace]:
        """Convert collected data to AgentTrace."""
        if run_id not in self.runs:
            return None
            
        data = self.runs[run_id]
        return AgentTrace(
            run_id=run_id,
            task=data["task"],
            status=data.get("status", "unknown"),
            total_steps=data["total_steps"],
            final_output=data.get("final_output"),
            steps=data["steps"]
        )

# Usage with callback
callback_handler = DiagnosticCallbackHandler()
result = agent_executor.invoke(
    {"input": "What's the weather in Paris?"},
    config={"callbacks": [callback_handler]}
)

# Automatic diagnosis
for run_id in callback_handler.runs:
    trace = callback_handler.get_trace(run_id)
    if trace:
        diagnosis = classifier.diagnose(trace)
        print(f"Run {run_id}: {diagnosis.failure_type.value}")
```

## OpenAI Assistants API

### Basic Integration

```python
import openai
import json
from agent_diagnostician import Classifier
from agent_diagnostician.models.trace import AgentTrace

def openai_assistant_to_diagnostician(client, thread_id: str, run_id: str):
    """Convert OpenAI Assistant run to Agent Diagnostician trace."""
    
    # Get run details
    run = client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run_id)
    
    # Get run steps
    run_steps = client.beta.threads.runs.steps.list(
        thread_id=thread_id, 
        run_id=run_id,
        order="asc"
    )
    
    # Get thread messages to extract task
    messages = client.beta.threads.messages.list(thread_id=thread_id, order="asc")
    task = ""
    final_output = ""
    
    for message in messages.data:
        if message.role == "user":
            task = message.content[0].text.value
        elif message.role == "assistant" and message.run_id == run_id:
            final_output = message.content[0].text.value
    
    # Convert steps
    steps = []
    step_index = 0
    
    for step in run_steps.data:
        if step.type == "tool_calls":
            for tool_call in step.step_details.tool_calls:
                diagnostic_step = {
                    "step_index": step_index,
                    "tool_name": tool_call.function.name,
                    "tool_input": json.loads(tool_call.function.arguments),
                    "tool_output": tool_call.function.output
                }
                
                # Add error information if available
                if step.last_error:
                    diagnostic_step["error_message"] = step.last_error.message
                    diagnostic_step["step_status"] = "error"
                else:
                    diagnostic_step["step_status"] = "success"
                
                steps.append(diagnostic_step)
                step_index += 1
        
        elif step.type == "message_creation":
            # Handle message creation steps
            message_step = {
                "step_index": step_index,
                "tool_name": "create_message",
                "tool_input": {"message_type": "assistant_response"},
                "tool_output": final_output
            }
            steps.append(message_step)
            step_index += 1
    
    # Build trace
    trace_data = {
        "run_id": run_id,
        "task": task,
        "status": run.status,
        "total_steps": len(steps),
        "final_output": final_output,
        "steps": steps
    }
    
    # Add token usage if available
    if hasattr(run, 'usage') and run.usage:
        trace_data["total_tokens"] = run.usage.total_tokens
    
    return AgentTrace(**trace_data)

# Usage example
client = openai.OpenAI(api_key="your-key")

# Create assistant and thread
assistant = client.beta.assistants.create(
    name="Data Analyst",
    instructions="You help analyze data using available tools.",
    tools=[{"type": "code_interpreter"}],
    model="gpt-4"
)

thread = client.beta.threads.create()

# Add user message
client.beta.threads.messages.create(
    thread_id=thread.id,
    role="user", 
    content="Calculate the average of these numbers: 10, 20, 30, 40, 50"
)

# Run assistant
run = client.beta.threads.runs.create(
    thread_id=thread.id,
    assistant_id=assistant.id
)

# Wait for completion
import time
while run.status in ["queued", "in_progress"]:
    time.sleep(1)
    run = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)

# Convert and diagnose
trace = openai_assistant_to_diagnostician(client, thread.id, run.id)
classifier = Classifier()
diagnosis = classifier.diagnose(trace)

print(f"Assistant run diagnosis: {diagnosis.failure_type.value}")
print(f"Confidence: {diagnosis.confidence_band.value}")
```

### Assistant Tools Integration

```python
def extract_assistant_tools(client, assistant_id: str) -> List[Dict]:
    """Extract tool specifications from OpenAI Assistant."""
    
    assistant = client.beta.assistants.retrieve(assistant_id)
    available_tools = []
    
    for tool in assistant.tools:
        if tool.type == "function":
            tool_spec = {
                "name": tool.function.name,
                "description": tool.function.description,
                "schema": tool.function.parameters
            }
            available_tools.append(tool_spec)
        elif tool.type in ["code_interpreter", "retrieval"]:
            # Built-in tools
            tool_spec = {
                "name": tool.type,
                "description": f"OpenAI built-in {tool.type} tool"
            }
            available_tools.append(tool_spec)
    
    return available_tools

# Enhanced trace with tool information
def enhanced_openai_to_diagnostician(client, thread_id: str, run_id: str):
    trace = openai_assistant_to_diagnostician(client, thread_id, run_id)
    
    # Get assistant ID from run
    run = client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run_id)
    
    # Add available tools
    available_tools = extract_assistant_tools(client, run.assistant_id)
    trace.available_tools = available_tools
    
    return trace
```

## LlamaIndex

### Basic Integration

```python
from llama_index.core import VectorStoreIndex, Document
from llama_index.core.agent import ReActAgent
from llama_index.tools.tool_spec.base import BaseToolSpec
from agent_diagnostician import Classifier
from agent_diagnostician.models.trace import AgentTrace

class DiagnosticReActAgent(ReActAgent):
    """Extended ReActAgent with diagnostic capability."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.execution_trace = {
            "steps": [],
            "task": None,
            "final_output": None
        }
    
    def _run_step(self, step, task):
        """Override to capture step information."""
        
        # Capture step details
        step_data = {
            "step_index": len(self.execution_trace["steps"]),
            "tool_name": getattr(step.action, 'tool_name', 'unknown'),
            "tool_input": getattr(step.action, 'tool_input', {}),
            "thought": getattr(step, 'thought', None)
        }
        
        # Execute original step
        result = super()._run_step(step, task)
        
        # Capture output
        step_data["tool_output"] = getattr(result, 'output', None)
        step_data["step_status"] = "success" if result.is_done else "continuing"
        
        self.execution_trace["steps"].append(step_data)
        return result
    
    def chat(self, message: str) -> str:
        """Override chat to capture full execution."""
        self.execution_trace = {
            "steps": [],
            "task": message,
            "final_output": None
        }
        
        result = super().chat(message)
        self.execution_trace["final_output"] = str(result)
        
        return result
    
    def get_diagnostic_trace(self, run_id: str = None) -> AgentTrace:
        """Convert execution to AgentTrace."""
        import uuid
        
        if run_id is None:
            run_id = str(uuid.uuid4())
        
        # Extract available tools
        available_tools = []
        for tool in self.get_tools():
            tool_spec = {
                "name": tool.metadata.name,
                "description": tool.metadata.description
            }
            available_tools.append(tool_spec)
        
        return AgentTrace(
            run_id=run_id,
            task=self.execution_trace["task"],
            status="success" if self.execution_trace["final_output"] else "error",
            total_steps=len(self.execution_trace["steps"]),
            final_output=self.execution_trace["final_output"],
            steps=self.execution_trace["steps"],
            available_tools=available_tools
        )

# Usage example
from llama_index.tools import FunctionTool

def get_weather(city: str) -> str:
    """Get weather for a city."""
    return f"Weather in {city}: Sunny, 75°F"

def calculate_tip(bill: float, percentage: float = 15.0) -> float:
    """Calculate tip amount."""
    return bill * (percentage / 100)

# Create tools
weather_tool = FunctionTool.from_defaults(fn=get_weather)
tip_tool = FunctionTool.from_defaults(fn=calculate_tip)

# Create diagnostic agent
agent = DiagnosticReActAgent.from_tools([weather_tool, tip_tool])

# Run and diagnose
response = agent.chat("What's the weather in New York and calculate a 20% tip on $50?")

# Get diagnostic trace
trace = agent.get_diagnostic_trace()

# Diagnose
classifier = Classifier()
diagnosis = classifier.diagnose(trace)

print(f"LlamaIndex agent diagnosis: {diagnosis.failure_type.value}")
print(f"Confidence: {diagnosis.confidence_band.value}")
```

## AutoGen

### Basic Integration

```python
import autogen
from agent_diagnostician import Classifier
from agent_diagnostician.models.trace import AgentTrace

class DiagnosticUserProxyAgent(autogen.UserProxyAgent):
    """Extended UserProxyAgent with diagnostic tracking."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.diagnostic_trace = {
            "steps": [],
            "messages": [],
            "task": None
        }
    
    def send(self, message, recipient, request_reply=None):
        """Override send to track interactions."""
        
        # Capture step information
        step_data = {
            "step_index": len(self.diagnostic_trace["steps"]),
            "tool_name": f"send_to_{recipient.name}",
            "tool_input": {"message": message, "recipient": recipient.name},
            "timestamp": time.time()
        }
        
        # Send message
        reply = super().send(message, recipient, request_reply)
        
        # Capture response
        step_data["tool_output"] = reply
        self.diagnostic_trace["steps"].append(step_data)
        self.diagnostic_trace["messages"].append({
            "from": self.name,
            "to": recipient.name,
            "content": message,
            "reply": reply
        })
        
        return reply
    
    def initiate_chat(self, recipient, message=None, **kwargs):
        """Override initiate_chat to capture task."""
        self.diagnostic_trace["task"] = message
        return super().initiate_chat(recipient, message, **kwargs)
    
    def get_diagnostic_trace(self, run_id: str = None) -> AgentTrace:
        """Convert AutoGen execution to AgentTrace."""
        import uuid
        
        if run_id is None:
            run_id = str(uuid.uuid4())
        
        # Determine final output from last message
        final_output = None
        if self.diagnostic_trace["messages"]:
            final_output = self.diagnostic_trace["messages"][-1].get("reply")
        
        return AgentTrace(
            run_id=run_id,
            task=self.diagnostic_trace["task"],
            status="success",  # AutoGen doesn't have explicit status
            total_steps=len(self.diagnostic_trace["steps"]),
            final_output=final_output,
            steps=self.diagnostic_trace["steps"]
        )

# Usage example
config_list = [
    {
        "model": "gpt-4",
        "api_key": "your-openai-key"
    }
]

# Create diagnostic user proxy
user_proxy = DiagnosticUserProxyAgent(
    name="user_proxy",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=10
)

# Create assistant
assistant = autogen.AssistantAgent(
    name="assistant", 
    llm_config={"config_list": config_list}
)

# Run conversation
user_proxy.initiate_chat(
    assistant, 
    message="Calculate the compound interest on $1000 at 5% annual rate for 3 years."
)

# Get diagnostic trace
trace = user_proxy.get_diagnostic_trace()

# Diagnose
classifier = Classifier()
diagnosis = classifier.diagnose(trace)

print(f"AutoGen conversation diagnosis: {diagnosis.failure_type.value}")
```

## Custom Framework Integration

### Generic Integration Template

```python
from agent_diagnostician import Classifier
from agent_diagnostician.models.trace import AgentTrace
from typing import Any, Dict, List, Optional

class GenericAgentTracer:
    """Generic tracer for custom agent frameworks."""
    
    def __init__(self):
        self.current_trace = None
        self.step_counter = 0
    
    def start_trace(self, run_id: str, task: str):
        """Start a new execution trace."""
        self.current_trace = {
            "run_id": run_id,
            "task": task,
            "steps": [],
            "start_time": time.time(),
            "status": "running"
        }
        self.step_counter = 0
    
    def log_step(
        self, 
        tool_name: str,
        tool_input: Dict[str, Any],
        tool_output: Any = None,
        thought: str = None,
        error_message: str = None
    ):
        """Log a single execution step."""
        if not self.current_trace:
            raise ValueError("No active trace. Call start_trace() first.")
        
        step = {
            "step_index": self.step_counter,
            "tool_name": tool_name,
            "tool_input": tool_input,
            "tool_output": tool_output,
            "timestamp": time.time()
        }
        
        if thought:
            step["thought"] = thought
        if error_message:
            step["error_message"] = error_message
            step["step_status"] = "error"
        else:
            step["step_status"] = "success"
        
        self.current_trace["steps"].append(step)
        self.step_counter += 1
    
    def end_trace(
        self, 
        final_output: Any = None,
        status: str = "success",
        available_tools: List[Dict] = None
    ) -> AgentTrace:
        """End trace and return AgentTrace object."""
        if not self.current_trace:
            raise ValueError("No active trace.")
        
        trace_data = {
            "run_id": self.current_trace["run_id"],
            "task": self.current_trace["task"],
            "status": status,
            "total_steps": len(self.current_trace["steps"]),
            "final_output": final_output,
            "steps": self.current_trace["steps"]
        }
        
        if available_tools:
            trace_data["available_tools"] = available_tools
        
        trace = AgentTrace(**trace_data)
        self.current_trace = None  # Reset for next trace
        return trace

# Usage with custom agent
tracer = GenericAgentTracer()
classifier = Classifier()

# Start execution
tracer.start_trace("custom_001", "Book a flight from NYC to LAX")

# Log steps as they happen
tracer.log_step(
    tool_name="search_flights",
    tool_input={"origin": "NYC", "destination": "LAX", "date": "2024-02-15"},
    tool_output=[{"flight": "AA123", "price": "$299", "time": "2:00 PM"}],
    thought="Found several flights, selecting the cheapest option"
)

tracer.log_step(
    tool_name="book_flight", 
    tool_input={"flight_id": "AA123", "passenger": "John Doe"},
    tool_output={"confirmation": "ABC123", "status": "confirmed"}
)

# End trace
trace = tracer.end_trace(
    final_output="Flight AA123 booked successfully. Confirmation: ABC123",
    available_tools=[
        {"name": "search_flights", "description": "Search for available flights"},
        {"name": "book_flight", "description": "Book a specific flight"}
    ]
)

# Diagnose
diagnosis = classifier.diagnose(trace)
print(f"Custom agent diagnosis: {diagnosis.failure_type.value}")
```

## Best Practices for Integration

### 1. Error Handling

```python
def safe_diagnose(trace: AgentTrace) -> Optional[DetectionResult]:
    """Safely diagnose with proper error handling."""
    try:
        classifier = Classifier()
        return classifier.diagnose(trace)
    except Exception as e:
        logger.error(f"Diagnosis failed for {trace.run_id}: {e}")
        return None

def validate_trace_data(trace_data: Dict) -> bool:
    """Validate trace data before creating AgentTrace."""
    required_fields = ["run_id", "task", "status", "total_steps", "steps"]
    
    for field in required_fields:
        if field not in trace_data:
            logger.error(f"Missing required field: {field}")
            return False
    
    if not isinstance(trace_data["steps"], list):
        logger.error("Steps must be a list")
        return False
    
    return True
```

### 2. Performance Optimization

```python
class CachedDiagnosticClassifier:
    """Classifier with caching for repeated diagnoses."""
    
    def __init__(self):
        self.classifier = Classifier()
        self.cache = {}
    
    def diagnose(self, trace: AgentTrace) -> DetectionResult:
        # Create cache key from trace content
        cache_key = self._create_cache_key(trace)
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        result = self.classifier.diagnose(trace)
        self.cache[cache_key] = result
        return result
    
    def _create_cache_key(self, trace: AgentTrace) -> str:
        # Simple hash-based cache key
        import hashlib
        content = f"{trace.task}_{len(trace.steps)}_{trace.status}"
        return hashlib.md5(content.encode()).hexdigest()
```

### 3. Async Processing

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

class AsyncDiagnosticService:
    """Async service for batch diagnosis."""
    
    def __init__(self, max_workers: int = 4):
        self.classifier = Classifier()
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
    
    async def diagnose_batch(self, traces: List[AgentTrace]) -> List[DetectionResult]:
        """Diagnose multiple traces concurrently."""
        loop = asyncio.get_event_loop()
        
        tasks = [
            loop.run_in_executor(
                self.executor, 
                self.classifier.diagnose, 
                trace
            )
            for trace in traces
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Failed to diagnose trace {traces[i].run_id}: {result}")
                processed_results.append(None)
            else:
                processed_results.append(result)
        
        return processed_results

# Usage
async def main():
    service = AsyncDiagnosticService()
    results = await service.diagnose_batch(traces)
    
    for trace, result in zip(traces, results):
        if result:
            print(f"{trace.run_id}: {result.failure_type.value}")
```