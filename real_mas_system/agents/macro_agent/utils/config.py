"""
Configuration management for the Multi-Agent System.
"""
from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class AgentConfig:
    """Individual agent configuration."""
    model: str = "claude-3-5-sonnet-20241022"
    temperature: float = 0.3
    max_tokens: int = 2000
    timeout: int = 60


@dataclass
class MASConfig:
    """Multi-Agent System configuration."""
    
    planner_config: AgentConfig = field(default_factory=lambda: AgentConfig(temperature=0.1, max_tokens=1500))
    summarizer_config: AgentConfig = field(default_factory=lambda: AgentConfig(temperature=0.3, max_tokens=1000))
    critic_config: AgentConfig = field(default_factory=lambda: AgentConfig(temperature=0.2, max_tokens=2000))
    
    parallel_summarization: bool = True
    max_chunk_size: int = 2000
    overlap_size: int = 200
    enable_logging: bool = True
    log_level: str = "INFO"
    
    enable_adaptive_feedback: bool = True
    feedback_max_iterations: int = 2
    feedback_quality_threshold: float = 0.6
    feedback_confidence_threshold: float = 0.6
    feedback_temperature_adjustment: float = 0.1
    feedback_max_tokens_adjustment: int = 200
    
    api_provider: str = "openai"
    rate_limit_delay: float = 1.0
    max_retries: int = 3
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "planner_config": self.planner_config.__dict__,
            "summarizer_config": self.summarizer_config.__dict__,
            "critic_config": self.critic_config.__dict__,
            "parallel_summarization": self.parallel_summarization,
            "max_chunk_size": self.max_chunk_size,
            "overlap_size": self.overlap_size,
            "enable_logging": self.enable_logging,
            "log_level": self.log_level,
            "api_provider": self.api_provider,
            "rate_limit_delay": self.rate_limit_delay,
            "max_retries": self.max_retries
        }


def get_default_config() -> MASConfig:
    return MASConfig()