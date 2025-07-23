"""Feedback Loop System for adaptive re-summarization."""

from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
import copy


@dataclass
class FeedbackConfig:
    """Feedback loop configuration."""
    
    min_confidence_threshold: float = 0.6
    min_individual_score_threshold: float = 0.6
    
    critical_issues: List[str] = None
    max_issues_per_summary: int = 3
    
    max_feedback_iterations: int = 2
    enable_final_critic_pass: bool = True
    
    temperature_adjustment: float = 0.1
    max_tokens_adjustment: int = 200
    
    def __post_init__(self):
        if self.critical_issues is None:
            self.critical_issues = [
                "hallucination",
                "factual error",
                "missing key information",
                "incoherent",
                "off-topic"
            ]


class FeedbackLoopSystem:
    """System for adaptive feedback and re-summarization."""
    
    def __init__(self, config: FeedbackConfig = None):
        self.config = config or FeedbackConfig()
        self.iteration_history = []
    
    def analyze_critic_output(self, critic_result: Dict[str, Any], 
                            summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze critic output to identify summaries needing re-processing."""
        evaluation = critic_result.get("evaluation", {})
        individual_scores = evaluation.get("individual_scores", [])
        
        feedback_analysis = {
            "total_summaries": len(summaries),
            "low_confidence_summaries": [],
            "low_score_summaries": [],
            "issue_based_problems": [],
            "overall_quality_issues": []
        }
        
        problematic_indices = set()
        
        for i, summary in enumerate(summaries):
            confidence = summary.get("confidence", 1.0)
            if confidence < self.config.min_confidence_threshold:
                feedback_analysis["low_confidence_summaries"].append({
                    "index": i,
                    "task_id": summary.get("task_id"),
                    "confidence": confidence,
                    "threshold": self.config.min_confidence_threshold
                })
                problematic_indices.add(i)
        
        for score_info in individual_scores:
            task_id = score_info.get("task_id")
            score = score_info.get("score", 1.0)
            issues = score_info.get("issues", [])
            
            summary_idx = None
            for i, summary in enumerate(summaries):
                if summary.get("task_id") == task_id:
                    summary_idx = i
                    break
            
            if summary_idx is not None:
                if score < self.config.min_individual_score_threshold:
                    feedback_analysis["low_score_summaries"].append({
                        "index": summary_idx,
                        "task_id": task_id,
                        "score": score,
                        "threshold": self.config.min_individual_score_threshold,
                        "issues": issues
                    })
                    problematic_indices.add(summary_idx)
                
                critical_issues_found = []
                for issue in issues:
                    for critical_issue in self.config.critical_issues:
                        if critical_issue.lower() in issue.lower():
                            critical_issues_found.append(issue)
                
                if (critical_issues_found or 
                    len(issues) > self.config.max_issues_per_summary):
                    feedback_analysis["issue_based_problems"].append({
                        "index": summary_idx,
                        "task_id": task_id,
                        "critical_issues": critical_issues_found,
                        "total_issues": len(issues),
                        "all_issues": issues
                    })
                    problematic_indices.add(summary_idx)
        
        synthesis_quality = evaluation.get("synthesis_quality", {})
        overall_quality = evaluation.get("overall_quality", 1.0)
        
        if overall_quality < self.config.min_confidence_threshold:
            feedback_analysis["overall_quality_issues"].append({
                "overall_quality": overall_quality,
                "synthesis_scores": synthesis_quality,
                "recommendation": "Consider re-processing multiple summaries"
            })
        
        needs_feedback = len(problematic_indices) > 0
        
        retry_tasks = []
        if needs_feedback:
            retry_tasks = self._create_retry_tasks(
                summaries, list(problematic_indices), feedback_analysis
            )
        
        return {
            "needs_feedback": needs_feedback,
            "problematic_summaries": sorted(list(problematic_indices)),
            "feedback_analysis": feedback_analysis,
            "retry_tasks": retry_tasks
        }
    
    def _create_retry_tasks(self, summaries: List[Dict[str, Any]], 
                           problematic_indices: List[int],
                           feedback_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        retry_tasks = []
        
        for idx in problematic_indices:
            if idx >= len(summaries):
                continue
                
            summary = summaries[idx]
            task_id = summary.get("task_id")
            
            original_content = summary.get("original_content", "")
            original_instructions = summary.get("original_instructions", "")
            
            problems = self._analyze_specific_problems(idx, feedback_analysis)
            
            improved_instructions = self._create_improved_instructions(
                original_instructions, problems
            )
            
            retry_task = {
                "task_id": f"{task_id}_retry",
                "original_task_id": task_id,
                "content": original_content,
                "instructions": improved_instructions,
                "priority": "high",
                "context": f"Retry due to: {', '.join([p['type'] for p in problems])}",
                "retry_metadata": {
                    "original_summary": summary.get("summary", ""),
                    "original_confidence": summary.get("confidence", 0),
                    "problems_identified": problems,
                    "retry_iteration": len(self.iteration_history) + 1
                }
            }
            
            retry_tasks.append(retry_task)
        
        return retry_tasks
    
    def _analyze_specific_problems(self, summary_idx: int, 
                                 feedback_analysis: Dict[str, Any]) -> List[Dict[str, str]]:
        problems = []
        
        for item in feedback_analysis["low_confidence_summaries"]:
            if item["index"] == summary_idx:
                problems.append({
                    "type": "low_confidence",
                    "description": f"Confidence {item['confidence']:.2f} below threshold {item['threshold']:.2f}",
                    "severity": "medium"
                })
        
        for item in feedback_analysis["low_score_summaries"]:
            if item["index"] == summary_idx:
                problems.append({
                    "type": "low_quality_score",
                    "description": f"Quality score {item['score']:.2f} below threshold {item['threshold']:.2f}",
                    "severity": "high",
                    "issues": item.get("issues", [])
                })
        
        for item in feedback_analysis["issue_based_problems"]:
            if item["index"] == summary_idx:
                severity = "critical" if item["critical_issues"] else "medium"
                problems.append({
                    "type": "content_issues",
                    "description": f"Found {len(item['all_issues'])} issues including: {', '.join(item['critical_issues'][:2])}",
                    "severity": severity,
                    "critical_issues": item["critical_issues"],
                    "all_issues": item["all_issues"]
                })
        
        return problems
    
    def _create_improved_instructions(self, original_instructions: str, 
                                    problems: List[Dict[str, str]]) -> str:
        base_instructions = original_instructions
        
        improvements = []
        
        for problem in problems:
            if problem["type"] == "low_confidence":
                improvements.append(
                    "Be more specific and detailed in your summary. "
                    "Include concrete examples and key facts."
                )
            
            elif problem["type"] == "low_quality_score":
                issues = problem.get("issues", [])
                if any("coherence" in issue.lower() for issue in issues):
                    improvements.append(
                        "Focus on logical flow and clear connections between ideas."
                    )
                if any("relevance" in issue.lower() for issue in issues):
                    improvements.append(
                        "Ensure all content directly relates to the specified goal and audience."
                    )
            
            elif problem["type"] == "content_issues":
                critical_issues = problem.get("critical_issues", [])
                if any("hallucination" in issue.lower() for issue in critical_issues):
                    improvements.append(
                        "Stick strictly to the provided content. Do not add information not present in the source."
                    )
                if any("missing" in issue.lower() for issue in critical_issues):
                    improvements.append(
                        "Ensure all key points from the source content are captured."
                    )
                if any("incoherent" in issue.lower() for issue in critical_issues):
                    improvements.append(
                        "Structure the summary with clear topic sentences and logical progression."
                    )
        
        if improvements:
            improved_instructions = f"""{base_instructions}

IMPORTANT: This is a retry with specific improvements needed:
{chr(10).join(f"- {improvement}" for improvement in improvements)}

Please address these specific issues while maintaining the original goal and requirements."""
        else:
            improved_instructions = f"""{base_instructions}

IMPORTANT: This is a retry. Please be more careful and thorough in your analysis and summary creation."""
        
        return improved_instructions
    
    def merge_retry_results(self, original_summaries: List[Dict[str, Any]], 
                          retry_summaries: List[Dict[str, Any]], 
                          problematic_indices: List[int]) -> List[Dict[str, Any]]:
        """Merge retry results back into original summary list."""
        updated_summaries = copy.deepcopy(original_summaries)
        
        retry_mapping = {}
        for retry_summary in retry_summaries:
            original_task_id = retry_summary.get("retry_metadata", {}).get("original_task_id")
            if original_task_id:
                retry_mapping[original_task_id] = retry_summary
        
        for idx in problematic_indices:
            if idx < len(updated_summaries):
                original_task_id = updated_summaries[idx].get("task_id")
                if original_task_id in retry_mapping:
                    retry_summary = retry_mapping[original_task_id]
                    
                    retry_summary["task_id"] = original_task_id
                    retry_summary["is_retry"] = True
                    retry_summary["retry_iteration"] = len(self.iteration_history) + 1
                    
                    updated_summaries[idx] = retry_summary
        
        return updated_summaries
    
    def should_continue_feedback(self, iteration_count: int) -> bool:
        return iteration_count < self.config.max_feedback_iterations
    
    def record_iteration(self, iteration_data: Dict[str, Any]):
        self.iteration_history.append({
            "iteration": len(self.iteration_history) + 1,
            "timestamp": iteration_data.get("timestamp"),
            "summaries_retried": iteration_data.get("summaries_retried", 0),
            "problems_identified": iteration_data.get("problems_identified", []),
            "improvements_achieved": iteration_data.get("improvements_achieved", {})
        })
    
    def get_feedback_summary(self) -> Dict[str, Any]:
        return {
            "total_iterations": len(self.iteration_history),
            "max_iterations_allowed": self.config.max_feedback_iterations,
            "iteration_history": self.iteration_history,
            "config_used": {
                "min_confidence_threshold": self.config.min_confidence_threshold,
                "min_individual_score_threshold": self.config.min_individual_score_threshold,
                "critical_issues": self.config.critical_issues,
                "max_issues_per_summary": self.config.max_issues_per_summary
            }
        }