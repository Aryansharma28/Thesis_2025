#!/usr/bin/env python3

import json
import time
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path

from .macro_agent import MacroAgent
from .agents.planner_agent import PlannerAgent
from .agents.summarizer_agent import SummarizerAgent
from .agents.critic_agent import CriticAgent

from .utils.feedback_loop import FeedbackLoopSystem, FeedbackConfig
from .utils.config import MASConfig, get_default_config
from .utils.logger import MASLogger


class AdaptiveMacroAgent(MacroAgent):
    """
    Enhanced MacroAgent with adaptive feedback loops for quality improvement.
    
    Key features:
    - Automatic detection of low-quality summaries
    - Intelligent re-summarization with improved instructions
    - Iterative improvement until quality thresholds are met
    - Comprehensive tracking of feedback iterations
    """
    
    def __init__(self, config: MASConfig = None, feedback_config: FeedbackConfig = None, 
                 logger: MASLogger = None):
        # Initialize base components
        self.config = config or get_default_config()
        self.feedback_config = feedback_config or FeedbackConfig()
        self.logger = logger or MASLogger(
            log_level=self.config.log_level,
            log_file="adaptive_mas_execution.log" if self.config.enable_logging else None
        )
        
        # Initialize agents with enhanced critic
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
        
        # Initialize feedback loop system
        self.feedback_system = FeedbackLoopSystem(self.feedback_config)
        
        # Track iterations
        self.current_iteration = 0
        self.max_iterations = self.feedback_config.max_feedback_iterations
    
    def process_document_adaptive(self, document: str, goal: str, 
                                save_intermediate: bool = False, 
                                output_dir: str = "outputs") -> Dict[str, Any]:
        """
        Process document with adaptive feedback loops for quality improvement.
        
        Args:
            document: Input document text
            goal: Summarization goal
            save_intermediate: Save intermediate results
            output_dir: Output directory
            
        Returns:
            Complete results with feedback loop information
        """
        start_time = time.time()
        self.logger.log_mas_start(len(document), goal)
        
        try:
            # Step 1: Initial Planning (same as before)
            planner_result = self._run_planner(document, goal)
            plan = planner_result["plan"]
            original_tasks = plan["tasks"]
            
            # Step 2: Initial Summarization
            summarizer_result = self._run_summarizer(plan["tasks"], goal)
            current_summaries = summarizer_result["summaries"]
            
            # Store original tasks in summaries for feedback loop
            self._attach_original_tasks_to_summaries(current_summaries, original_tasks)
            
            # Step 3: Adaptive Feedback Loop
            feedback_results = self._run_adaptive_feedback_loop(
                current_summaries, original_tasks, goal, plan.get("synthesis_guidance", "")
            )
            
            final_summaries = feedback_results["final_summaries"]
            
            # Step 4: Final Synthesis
            final_critic_result = self._run_final_critic(
                final_summaries, goal, plan.get("synthesis_guidance", ""), len(document)
            )
            
            # Compile comprehensive results
            total_time = time.time() - start_time
            complete_results = self._compile_adaptive_results(
                document, goal, planner_result, summarizer_result, 
                feedback_results, final_critic_result, total_time
            )
            
            # Save results if requested
            if save_intermediate:
                self._save_adaptive_results(complete_results, output_dir)
            
            self.logger.log_mas_complete(final_critic_result["final_summary"], total_time)
            return complete_results
            
        except Exception as e:
            self.logger.log_error("AdaptiveMacroAgent", e)
            raise
    
    def _run_planner(self, document: str, goal: str) -> Dict[str, Any]:
        """Run the planner agent."""
        self.logger.log_agent_start("PlannerAgent", {"document_length": len(document), "goal": goal})
        
        planner_input = {
            "document": document,
            "goal": goal,
            "max_chunk_size": self.config.max_chunk_size
        }
        
        result = self.planner.process(planner_input)
        
        self.logger.log_agent_complete(
            "PlannerAgent", result, result["reasoning_log"], result["stats"]
        )
        
        return result
    
    def _run_summarizer(self, tasks: List[Dict[str, Any]], goal: str, 
                       is_retry: bool = False) -> Dict[str, Any]:
        """Run the summarizer agent."""
        agent_name = "SummarizerAgent_Retry" if is_retry else "SummarizerAgent"
        self.logger.log_agent_start(agent_name, {"num_tasks": len(tasks)})
        
        # Adjust temperature for retries
        if is_retry:
            original_temp = self.summarizer.temperature
            self.summarizer.temperature = min(1.0, original_temp + self.feedback_config.temperature_adjustment)
        
        summarizer_input = {
            "tasks": tasks,
            "goal": goal,
            "parallel": self.config.parallel_summarization
        }
        
        result = self.summarizer.process(summarizer_input)
        
        # Restore original temperature
        if is_retry:
            self.summarizer.temperature = original_temp
        
        self.logger.log_agent_complete(
            agent_name, result, result["reasoning_log"], result["stats"]
        )
        
        return result
    
    def _run_critic(self, summaries: List[Dict[str, Any]], goal: str, 
                   synthesis_guidance: str = "", original_length: int = 0, 
                   evaluation_only: bool = False) -> Dict[str, Any]:
        """Run the critic agent."""
        agent_name = "CriticAgent_Evaluation" if evaluation_only else "CriticAgent"
        self.logger.log_agent_start(agent_name, {"num_summaries": len(summaries)})
        
        if evaluation_only:
            # Run basic evaluation with regular CriticAgent
            critic_input = {
                "summaries": summaries,
                "goal": goal,
                "synthesis_guidance": synthesis_guidance,
                "original_document_length": original_length,
                "evaluation_only": True
            }
            result = self.critic.process(critic_input)
        else:
            # Full critic process
            critic_input = {
                "summaries": summaries,
                "goal": goal,
                "synthesis_guidance": synthesis_guidance,
                "original_document_length": original_length
            }
            result = self.critic.process(critic_input)
        
        self.logger.log_agent_complete(
            agent_name, result, result["reasoning_log"], result["stats"]
        )
        
        return result
    
    def _run_final_critic(self, summaries: List[Dict[str, Any]], goal: str, 
                         synthesis_guidance: str, original_length: int) -> Dict[str, Any]:
        """Run final critic synthesis."""
        return self._run_critic(summaries, goal, synthesis_guidance, original_length, False)
    
    def _attach_original_tasks_to_summaries(self, summaries: List[Dict[str, Any]], 
                                          original_tasks: List[Dict[str, Any]]):
        """Attach original task information to summaries for feedback loop."""
        task_map = {task["task_id"]: task for task in original_tasks}
        
        for summary in summaries:
            task_id = summary.get("task_id")
            if task_id in task_map:
                original_task = task_map[task_id]
                summary["original_content"] = original_task.get("content", "")
                summary["original_instructions"] = original_task.get("instructions", "")
                summary["original_priority"] = original_task.get("priority", "medium")
                summary["original_context"] = original_task.get("context", "")
    
    def _run_adaptive_feedback_loop(self, initial_summaries: List[Dict[str, Any]], 
                                  original_tasks: List[Dict[str, Any]], goal: str, 
                                  synthesis_guidance: str) -> Dict[str, Any]:
        """
        Run the adaptive feedback loop to improve summary quality.
        
        Args:
            initial_summaries: Initial summaries from first pass
            original_tasks: Original tasks from planner
            goal: Overall goal
            synthesis_guidance: Guidance for synthesis
            
        Returns:
            Results including final summaries and feedback history
        """
        current_summaries = initial_summaries.copy()
        feedback_iterations = []
        iteration_count = 0
        
        self.logger.logger.info("Starting adaptive feedback loop")
        
        while iteration_count < self.max_iterations:
            iteration_count += 1
            iteration_start = time.time()
            
            self.logger.logger.info(f"Feedback iteration {iteration_count}/{self.max_iterations}")
            
            # Step 1: Evaluate current summaries
            critic_result = self._run_critic(
                current_summaries, goal, synthesis_guidance, 0, evaluation_only=True
            )
            
            # Step 2: Analyze feedback needs
            feedback_analysis = self.feedback_system.analyze_critic_output(
                critic_result, current_summaries
            )
            
            # Step 3: Check if feedback is needed
            if not feedback_analysis["needs_feedback"]:
                self.logger.logger.info("No feedback needed - quality thresholds met")
                break
            
            problematic_indices = feedback_analysis["problematic_summaries"]
            retry_tasks = feedback_analysis["retry_tasks"]
            
            self.logger.logger.info(
                f"Found {len(problematic_indices)} summaries needing improvement"
            )
            
            # Step 4: Re-summarize problematic chunks
            if retry_tasks:
                retry_result = self._run_summarizer(retry_tasks, goal, is_retry=True)
                retry_summaries = retry_result["summaries"]
                
                # Step 5: Merge retry results
                current_summaries = self.feedback_system.merge_retry_results(
                    current_summaries, retry_summaries, problematic_indices
                )
                
                # Step 6: Compare improvements (optional analysis)
                if hasattr(self.critic, 'compare_summaries'):
                    comparison = self.critic.compare_summaries(
                        initial_summaries, retry_summaries
                    )
                else:
                    comparison = {"improvement_rate": 0.5}  # Default estimate
            else:
                retry_summaries = []
                comparison = {"improvement_rate": 0.0}
            
            # Record iteration
            iteration_data = {
                "iteration": iteration_count,
                "timestamp": datetime.now().isoformat(),
                "summaries_retried": len(retry_tasks),
                "problematic_indices": problematic_indices,
                "feedback_analysis": feedback_analysis["feedback_analysis"],
                "retry_summaries": len(retry_summaries),
                "improvement_comparison": comparison,
                "execution_time": time.time() - iteration_start
            }
            
            feedback_iterations.append(iteration_data)
            self.feedback_system.record_iteration(iteration_data)
            
            # Check if we should continue
            if not self.feedback_system.should_continue_feedback(iteration_count):
                self.logger.logger.info("Maximum feedback iterations reached")
                break
        
        # Final evaluation after all iterations
        final_evaluation = self._run_critic(
            current_summaries, goal, synthesis_guidance, 0, evaluation_only=True
        )
        
        return {
            "final_summaries": current_summaries,
            "feedback_iterations": feedback_iterations,
            "total_iterations": iteration_count,
            "final_evaluation": final_evaluation,
            "feedback_summary": self.feedback_system.get_feedback_summary()
        }
    
    def _compile_adaptive_results(self, document: str, goal: str, 
                                planner_result: Dict[str, Any], 
                                initial_summarizer_result: Dict[str, Any],
                                feedback_results: Dict[str, Any],
                                final_critic_result: Dict[str, Any],
                                total_time: float) -> Dict[str, Any]:
        """Compile comprehensive results including feedback loop information."""
        return {
            "input": {
                "document": document,
                "goal": goal,
                "document_length": len(document)
            },
            "planner_output": planner_result,
            "initial_summarizer_output": initial_summarizer_result,
            "feedback_loop_results": feedback_results,
            "final_critic_output": final_critic_result,
            "final_summary": final_critic_result["final_summary"],
            "adaptive_metadata": {
                "total_execution_time": total_time,
                "feedback_iterations_performed": feedback_results["total_iterations"],
                "max_iterations_allowed": self.max_iterations,
                "final_quality_scores": feedback_results.get("final_evaluation", {}).get("overall_quality", 0.0),
                "improvement_achieved": self._calculate_improvement_metrics(
                    initial_summarizer_result, feedback_results
                ),
                "config_used": {
                    "mas_config": self.config.to_dict(),
                    "feedback_config": {
                        "min_confidence_threshold": self.feedback_config.min_confidence_threshold,
                        "min_individual_score_threshold": self.feedback_config.min_individual_score_threshold,
                        "max_feedback_iterations": self.feedback_config.max_feedback_iterations,
                        "critical_issues": self.feedback_config.critical_issues
                    }
                },
                "agent_stats": {
                    "planner": planner_result["stats"],
                    "initial_summarizer": initial_summarizer_result["stats"],
                    "final_critic": final_critic_result["stats"]
                }
            }
        }
    
    def _calculate_improvement_metrics(self, initial_result: Dict[str, Any], 
                                     feedback_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate metrics showing improvement achieved through feedback."""
        initial_summaries = initial_result["summaries"]
        final_summaries = feedback_results["final_summaries"]
        
        # Calculate average confidence improvement
        initial_confidences = [s.get("confidence", 0.5) for s in initial_summaries]
        final_confidences = [s.get("confidence", 0.5) for s in final_summaries]
        
        avg_initial_confidence = sum(initial_confidences) / len(initial_confidences) if initial_confidences else 0
        avg_final_confidence = sum(final_confidences) / len(final_confidences) if final_confidences else 0
        
        # Calculate issue reduction
        initial_issues = sum(len(s.get("issues", [])) for s in initial_summaries)
        final_issues = sum(len(s.get("issues", [])) for s in final_summaries)
        
        return {
            "confidence_improvement": avg_final_confidence - avg_initial_confidence,
            "initial_avg_confidence": avg_initial_confidence,
            "final_avg_confidence": avg_final_confidence,
            "issues_reduced": initial_issues - final_issues,
            "initial_total_issues": initial_issues,
            "final_total_issues": final_issues,
            "summaries_improved": len([s for s in final_summaries if s.get("is_retry", False)])
        }
    
    def _save_adaptive_results(self, results: Dict[str, Any], output_dir: str):
        """Save comprehensive adaptive results."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = int(time.time())
        
        # Save complete results
        with open(output_path / f"adaptive_mas_results_{timestamp}.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save final summary
        with open(output_path / f"adaptive_final_summary_{timestamp}.txt", 'w') as f:
            f.write(results["final_summary"])
        
        # Save feedback analysis
        feedback_summary = {
            "iterations": results["feedback_loop_results"]["feedback_iterations"],
            "improvement_metrics": results["adaptive_metadata"]["improvement_achieved"],
            "final_quality": results["adaptive_metadata"]["final_quality_scores"]
        }
        
        with open(output_path / f"feedback_analysis_{timestamp}.json", 'w') as f:
            json.dump(feedback_summary, f, indent=2, default=str)
        
        # Save execution log
        self.logger.save_execution_log(str(output_path / f"adaptive_execution_log_{timestamp}.json"))
    
    # Convenience method that wraps the adaptive process
    def process_document(self, document: str, goal: str, 
                        save_intermediate: bool = False, 
                        output_dir: str = "outputs") -> Dict[str, Any]:
        """Main entry point - uses adaptive processing by default."""
        return self.process_document_adaptive(document, goal, save_intermediate, output_dir)