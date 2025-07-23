# Multi-Agent Summarization System (MAS)

A collaborative AI system that uses specialized agents to create high-quality document summaries through planning, parallel processing, and critical evaluation.

## Architecture

The system consists of three specialized agents orchestrated by a controller:

1. **Planner Agent** - Analyzes documents and creates optimal summarization plans
2. **Summarizer Agent** - Processes chunks in parallel to create focused summaries  
3. **Critic Agent** - Evaluates and synthesizes summaries into final output
4. **Macro Agent** - Orchestrates the entire pipeline

## Quick Start

### Basic Usage

```python
from macro_agent import MacroAgent

# Initialize the system
agent = MacroAgent()

# Process a document
document = "Your long document text here..."
goal = "Summarize for executives"

results = agent.process_document(document, goal)
print(results['final_summary'])
```

### Command Line Usage

```bash
# Basic summarization
python macro_agent.py document.txt "summarize for executives" -o summary.txt

# With custom settings
python macro_agent.py document.txt "technical overview" --parallel --max-chunk-size 1500 --verbose
```

## Configuration

The system can be configured through the `MASConfig` class:

```python
from utils.config import MASConfig, AgentConfig

config = MASConfig(
    planner_config=AgentConfig(model="gpt-4", temperature=0.1),
    summarizer_config=AgentConfig(model="gpt-4", temperature=0.3),
    critic_config=AgentConfig(model="gpt-4", temperature=0.2),
    parallel_summarization=True,
    max_chunk_size=2000
)

agent = MacroAgent(config)
```

## API Integration

To use with real APIs, update the `_make_api_call` method in `agents/base_agent.py`:

### OpenAI Integration
```python
import openai

def _make_api_call(self, messages: List[Dict[str, str]], **kwargs) -> str:
    response = openai.chat.completions.create(
        model=self.model,
        messages=messages,
        temperature=self.temperature,
        **kwargs
    )
    return response.choices[0].message.content
```

### Anthropic Claude Integration
```python
import anthropic

def _make_api_call(self, messages: List[Dict[str, str]], **kwargs) -> str:
    response = anthropic_client.messages.create(
        model=self.model,
        messages=messages,
        temperature=self.temperature,
        **kwargs
    )
    return response.content[0].text
```

## Features

- **Intelligent Planning**: Documents are analyzed and optimally chunked
- **Parallel Processing**: Multiple summarization tasks run concurrently
- **Quality Assurance**: Critic agent evaluates and improves summaries
- **Comprehensive Logging**: Detailed execution logs and reasoning traces
- **Flexible Configuration**: Customizable for different use cases
- **Error Handling**: Robust fallback mechanisms
- **File Processing**: Direct file input/output support

## Output Structure

```json
{
  "final_summary": "The synthesized final summary",
  "planner_output": {
    "plan": {...},
    "reasoning_log": [...],
    "stats": {...}
  },
  "summarizer_output": {
    "summaries": [...],
    "reasoning_log": [...],
    "stats": {...}
  },
  "critic_output": {
    "evaluation": {...},
    "reasoning_log": [...],
    "stats": {...}
  },
  "metadata": {
    "total_execution_time": 15.2,
    "agent_stats": {...}
  }
}
```

## Examples

See `example_usage.py` for complete examples including:
- Basic document processing
- Custom configuration
- File processing
- Error handling

## Agent Details

### Planner Agent
- Analyzes document structure and content
- Creates chunking strategy based on logical sections
- Generates specific instructions for each chunk
- Provides synthesis guidance for final assembly

### Summarizer Agent
- Processes chunks according to planner instructions
- Supports parallel or sequential processing
- Maintains consistency with overall goal
- Reports confidence and potential issues

### Critic Agent
- Evaluates individual summary quality
- Identifies redundancies, gaps, and inconsistencies
- Synthesizes summaries into coherent final output
- Provides quality metrics and recommendations

## Dependencies

- Python 3.8+
- `openai` or `anthropic` (for API calls)
- `concurrent.futures` (built-in)
- `json`, `re`, `time`, `pathlib` (built-in)

## License

This implementation is provided as an educational example for building multi-agent systems. Modify and extend as needed for your specific use cases.