#!/usr/bin/env python3
"""Macro Agent Controller - Orchestrates the Multi-Agent Summarization System."""

import json
import time
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

# Import agents
from .agents.planner_agent import PlannerAgent
from .agents.summarizer_agent import SummarizerAgent
from .agents.critic_agent import CriticAgent

# Import utilities
from .utils.config import MASConfig, get_default_config
from .utils.logger import MASLogger


class MacroAgent:
    """Main controller for the Multi-Agent Summarization System."""
    
    def __init__(self, config: MASConfig = None, logger: MASLogger = None):
        self.config = config or get_default_config()
        self.logger = logger or MASLogger(
            log_level=self.config.log_level,
            log_file="mas_execution.log" if self.config.enable_logging else None
        )
        
        self.planner = PlannerAgent(
            model=self.config.planner_config.model,
            temperature=self.config.planner_config.temperature,
            max_chunk_size=self.config.max_chunk_size
        )
        
        self.summarizer = SummarizerAgent(
            model=self.config.summarizer_config.model,
            temperature=self.config.summarizer_config.temperature,
            parallel_processing=self.config.parallel_summarization
        )
        
        self.critic = CriticAgent(
            model=self.config.critic_config.model,
            temperature=self.config.critic_config.temperature
        )
    
    def process_document(self, document: str, goal: str, 
                        save_intermediate: bool = False, 
                        output_dir: str = "outputs") -> Dict[str, Any]:
        """Process a document through the complete MAS pipeline."""
        start_time = time.time()
        
        self.logger.log_mas_start(len(document), goal)
        
        try:
            self.logger.log_agent_start("PlannerAgent", {"document_length": len(document), "goal": goal})
            
            planner_input = {
                "document": document,
                "goal": goal,
                "max_chunk_size": self.config.max_chunk_size
            }
            
            planner_result = self.planner.process(planner_input)
            plan = planner_result["plan"]
            
            self.logger.log_agent_complete(
                "PlannerAgent", 
                planner_result, 
                planner_result["reasoning_log"], 
                planner_result["stats"]
            )
            
            self.logger.log_agent_start("SummarizerAgent", {"num_tasks": len(plan["tasks"])})
            
            summarizer_input = {
                "tasks": plan["tasks"],
                "goal": goal,
                "parallel": self.config.parallel_summarization
            }
            
            summarizer_result = self.summarizer.process(summarizer_input)
            summaries = summarizer_result["summaries"]
            
            self.logger.log_agent_complete(
                "SummarizerAgent", 
                summarizer_result, 
                summarizer_result["reasoning_log"], 
                summarizer_result["stats"]
            )
            
            self.logger.log_agent_start("CriticAgent", {"num_summaries": len(summaries)})
            
            critic_input = {
                "summaries": summaries,
                "goal": goal,
                "synthesis_guidance": plan.get("synthesis_guidance", ""),
                "original_document_length": len(document)
            }
            
            critic_result = self.critic.process(critic_input)
            final_summary = critic_result["final_summary"]
            
            self.logger.log_agent_complete(
                "CriticAgent", 
                critic_result, 
                critic_result["reasoning_log"], 
                critic_result["stats"]
            )
            
            total_time = time.time() - start_time
            
            complete_results = {
                "input": {
                    "document": document,
                    "goal": goal,
                    "document_length": len(document)
                },
                "planner_output": planner_result,
                "summarizer_output": summarizer_result,
                "critic_output": critic_result,
                "final_summary": final_summary,
                "metadata": {
                    "total_execution_time": total_time,
                    "config_used": self.config.to_dict(),
                    "agent_stats": {
                        "planner": planner_result["stats"],
                        "summarizer": summarizer_result["stats"],
                        "critic": critic_result["stats"]
                    }
                }
            }
            
            if save_intermediate:
                self._save_intermediate_results(complete_results, output_dir)
            
            self.logger.log_mas_complete(final_summary, total_time)
            
            return complete_results
            
        except Exception as e:
            self.logger.log_error("MacroAgent", e)
            raise
    
    def _save_intermediate_results(self, results: Dict[str, Any], output_dir: str):
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = int(time.time())
        
        with open(output_path / f"mas_results_{timestamp}.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        with open(output_path / f"final_summary_{timestamp}.txt", 'w') as f:
            f.write(results["final_summary"])
        
        self.logger.save_execution_log(str(output_path / f"execution_log_{timestamp}.json"))
    
    def process_file(self, input_file: str, goal: str, output_file: str = None) -> str:
        """Process a document from a file."""
        with open(input_file, 'r', encoding='utf-8') as f:
            document = f.read()
        
        results = self.process_document(document, goal, save_intermediate=True)
        final_summary = results["final_summary"]
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(final_summary)
        
        return final_summary


def main():
    parser = argparse.ArgumentParser(description="Multi-Agent Summarization System")
    parser.add_argument("input_file", help="Path to input document file")
    parser.add_argument("goal", help="Summarization goal (e.g., 'summarize for executives')")
    parser.add_argument("-o", "--output", help="Output file for final summary")
    parser.add_argument("--config", help="Path to config JSON file")
    parser.add_argument("--parallel", action="store_true", help="Enable parallel summarization")
    parser.add_argument("--no-parallel", action="store_true", help="Disable parallel summarization")
    parser.add_argument("--model", help="Model to use (overrides config)")
    parser.add_argument("--max-chunk-size", type=int, help="Maximum chunk size")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    config = get_default_config()
    if args.config:
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
            config = MASConfig(**config_dict)
    
    if args.parallel:
        config.parallel_summarization = True
    elif args.no_parallel:
        config.parallel_summarization = False
    
    if args.model:
        config.planner_config.model = args.model
        config.summarizer_config.model = args.model
        config.critic_config.model = args.model
    
    if args.max_chunk_size:
        config.max_chunk_size = args.max_chunk_size
    
    if args.verbose:
        config.log_level = "DEBUG"
    
    macro_agent = MacroAgent(config)
    
    print(f"Processing document: {args.input_file}")
    print(f"Goal: {args.goal}")
    print(f"Parallel processing: {config.parallel_summarization}")
    print("-" * 50)
    
    try:
        final_summary = macro_agent.process_file(args.input_file, args.goal, args.output)
        
        print("\n" + "="*50)
        print("FINAL SUMMARY")
        print("="*50)
        print(final_summary)
        
        if args.output:
            print(f"\nSummary saved to: {args.output}")
        
        exec_summary = macro_agent.logger.get_execution_summary()
        print(f"\nExecution Summary:")
        print(f"Total API calls: {exec_summary['total_api_calls']}")
        print(f"Total time: {exec_summary['total_execution_time']:.2f}s")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())