"""
Logging utilities for the Multi-Agent System.
"""
import logging
import json
from datetime import datetime
from typing import Dict, Any, List
from pathlib import Path


class MASLogger:
    """Multi-Agent System logger with structured output."""
    
    def __init__(self, log_level: str = "INFO", log_file: str = None):
        self.logger = logging.getLogger("MAS")
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(console_formatter)
            self.logger.addHandler(file_handler)
        
        self.execution_log = []
    
    def log_agent_start(self, agent_name: str, input_data: Dict[str, Any]):
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "event": "agent_start",
            "agent": agent_name,
            "input_size": len(str(input_data)) if input_data else 0
        }
        self.execution_log.append(log_entry)
        self.logger.info(f"Starting {agent_name}")
    
    def log_agent_complete(self, agent_name: str, output_data: Dict[str, Any], 
                          reasoning_log: List[Dict[str, str]], stats: Dict[str, Any]):
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "event": "agent_complete",
            "agent": agent_name,
            "output_size": len(str(output_data)) if output_data else 0,
            "reasoning_steps": len(reasoning_log),
            "stats": stats
        }
        self.execution_log.append(log_entry)
        self.logger.info(f"Completed {agent_name} - {stats['calls_made']} API calls, "
                        f"{stats['execution_time']:.2f}s")
    
    def log_mas_start(self, document_length: int, goal: str):
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "event": "mas_start",
            "document_length": document_length,
            "goal": goal
        }
        self.execution_log.append(log_entry)
        self.logger.info(f"Starting MAS - Document: {document_length} chars, Goal: {goal}")
    
    def log_mas_complete(self, final_summary: str, total_time: float):
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "event": "mas_complete",
            "summary_length": len(final_summary),
            "total_time": total_time
        }
        self.execution_log.append(log_entry)
        self.logger.info(f"MAS Complete - Summary: {len(final_summary)} chars, "
                        f"Total time: {total_time:.2f}s")
    
    def log_error(self, agent_name: str, error: Exception):
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "event": "error",
            "agent": agent_name,
            "error_type": type(error).__name__,
            "error_message": str(error)
        }
        self.execution_log.append(log_entry)
        self.logger.error(f"Error in {agent_name}: {error}")
    
    def save_execution_log(self, filepath: str):
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.execution_log, f, indent=2)
    
    def get_execution_summary(self) -> Dict[str, Any]:
        agent_stats = {}
        total_api_calls = 0
        total_time = 0
        
        for entry in self.execution_log:
            if entry["event"] == "agent_complete":
                agent = entry["agent"]
                stats = entry["stats"]
                agent_stats[agent] = stats
                total_api_calls += stats["calls_made"]
                total_time += stats["execution_time"]
        
        return {
            "total_api_calls": total_api_calls,
            "total_execution_time": total_time,
            "agent_stats": agent_stats,
            "total_events": len(self.execution_log)
        }