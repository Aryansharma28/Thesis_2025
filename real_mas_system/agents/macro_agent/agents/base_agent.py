"""
Base Agent class providing common functionality for all agents in the MAS.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import time
import json
import os
from datetime import datetime
from dotenv import load_dotenv
import anthropic

# Load environment variables
load_dotenv()


class BaseAgent(ABC):
    """Base class for all agents."""
    
    def __init__(self, name: str, model: str = "claude-3-5-sonnet-20241022", temperature: float = 0.3):
        self.name = name
        self.model = model
        self.temperature = temperature
        self.reasoning_log = []
        self.execution_stats = {
            "calls_made": 0,
            "total_tokens": 0,
            "execution_time": 0
        }
        
        # Initialize Anthropic client
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment variables")
        self.client = anthropic.Anthropic(api_key=api_key)
        
        print(f"✅ {self.name} initialized with model {self.model}")
    
    @abstractmethod
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
"""Process input data and return results."""
        pass
    
    def log_reasoning(self, step: str, details: str):
        timestamp = datetime.now().isoformat()
        self.reasoning_log.append({
            "timestamp": timestamp,
            "step": step,
            "details": details
        })
    
    def get_reasoning_log(self) -> List[Dict[str, str]]:
        return self.reasoning_log.copy()
    
    def clear_log(self):
        self.reasoning_log.clear()
    
    def _make_api_call(self, messages: List[Dict[str, str]], **kwargs) -> str:
"""Make API call with retry logic."""
        max_retries = 3
        base_delay = 2.0
        
        for attempt in range(max_retries):
            start_time = time.time()
            
            try:
                max_tokens = kwargs.pop('max_tokens', 2000)
                
                system_content = None
                user_messages = []
                
                for msg in messages:
                    if msg.get("role") == "system":
                        system_content = msg["content"]
                    else:
                        user_messages.append(msg)
                
                total_chars = sum(len(str(msg.get("content", ""))) for msg in messages)
                estimated_tokens = int(total_chars / 4)
                
                if estimated_tokens > 150000:
                    max_tokens = min(max_tokens, 2000)
                    print(f"⚠️ Large document detected ({estimated_tokens} est. tokens), reducing max_tokens to {max_tokens}")
                
                print(f"🔄 {self.name} API call attempt {attempt + 1} - {total_chars} chars, ~{estimated_tokens} tokens, max_tokens={max_tokens}")
                
                api_params = {
                    "model": self.model,
                    "messages": user_messages,
                    "temperature": self.temperature,
                    "max_tokens": max_tokens,
                    **kwargs
                }
                
                if system_content:
                    api_params["system"] = system_content
                
                response = self.client.messages.create(**api_params)
                
                time.sleep(1.5)
                
                content = response.content[0].text
                
                execution_time = time.time() - start_time
                self.execution_stats["calls_made"] += 1
                self.execution_stats["execution_time"] += execution_time
                
                tokens_used = 0
                if hasattr(response, 'usage'):
                    input_tokens = getattr(response.usage, 'input_tokens', 0)
                    output_tokens = getattr(response.usage, 'output_tokens', 0)
                    tokens_used = input_tokens + output_tokens
                    self.execution_stats["total_tokens"] += tokens_used
                
                print(f"✅ {self.name} API call successful - {tokens_used} tokens, {execution_time:.2f}s")
                
                return content
                
            except Exception as e:
                execution_time = time.time() - start_time
                self.execution_stats["calls_made"] += 1
                self.execution_stats["execution_time"] += execution_time
                
                error_msg = str(e)
                print(f"⚠️ {self.name} API call attempt {attempt + 1} failed: {error_msg}")
                
                retryable_errors = [
                    "rate_limit", "overloaded", "timeout", "503", "502", "500", "connection"
                ]
                
                is_retryable = any(err in error_msg.lower() for err in retryable_errors)
                
                if attempt < max_retries - 1 and is_retryable:
                    wait_time = base_delay * (2 ** attempt)
                    print(f"⏳ Retryable error, waiting {wait_time:.1f}s before retry...")
                    time.sleep(wait_time)
                    continue
                else:
                    if attempt == max_retries - 1:
                        print(f"❌ All {max_retries} attempts failed for {self.name}")
                    else:
                        print(f"❌ Non-retryable error for {self.name}: {error_msg}")
                    
                    return f"Error in {self.name}: API call failed after {attempt + 1} attempts. {error_msg}"
        
        return f"Error in {self.name}: Unexpected API call failure"
    
    def get_stats(self) -> Dict[str, Any]:
        return self.execution_stats.copy()