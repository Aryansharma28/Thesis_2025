"""Enhanced Critic Agent with detailed quality assessment for feedback loops."""
import json
import re
from typing import Dict, Any, List, Tuple
from .critic_agent import CriticAgent


class EnhancedCriticAgent(CriticAgent):
    """Enhanced CriticAgent with feedback loop support."""
    
    def __init__(self, model: str = "gpt-4", temperature: float = 0.2, 
                 detailed_evaluation: bool = True):
        super().__init__(model, temperature)
        self.detailed_evaluation = detailed_evaluation
        self.name = "EnhancedCriticAgent"
    
    def evaluate_individual_summaries(self, summaries: List[Dict[str, Any]], 
                                    goal: str) -> Dict[str, Any]:
        """Perform detailed evaluation of individual summaries."""
        self.log_reasoning("individual_evaluation", f"Evaluating {len(summaries)} summaries individually")
        
        individual_evaluations = []
        
        for i, summary in enumerate(summaries):
            evaluation = self._evaluate_single_summary(summary, goal, i)
            individual_evaluations.append(evaluation)
        
        # Compute overall statistics
        scores = [eval_data["overall_score"] for eval_data in individual_evaluations]
        avg_score = sum(scores) / len(scores) if scores else 0
        
        quality_distribution = {
            "excellent": len([s for s in scores if s >= 0.8]),
            "good": len([s for s in scores if 0.6 <= s < 0.8]),
            "fair": len([s for s in scores if 0.4 <= s < 0.6]),
            "poor": len([s for s in scores if s < 0.4])
        }
        
        return {
            "individual_scores": individual_evaluations,
            "overall_quality": avg_score,
            "quality_distribution": quality_distribution,
            "total_summaries": len(summaries),
            "summaries_needing_attention": len([s for s in scores if s < 0.6])
        }
    
    def _evaluate_single_summary(self, summary: Dict[str, Any], goal: str, 
                                index: int) -> Dict[str, Any]:
        task_id = summary.get("task_id", f"summary_{index}")
        summary_text = summary.get("summary", "")
        confidence = summary.get("confidence", 0.5)
        existing_issues = summary.get("issues", [])
        
        self.log_reasoning("single_evaluation", f"Evaluating {task_id}")
        
        quality_scores = self._assess_summary_quality(summary_text, goal)
        
        detected_issues = self._detect_summary_issues(summary, goal)
        
        all_issues = list(set(existing_issues + detected_issues))
        
        overall_score = self._calculate_overall_score(quality_scores, confidence, all_issues)
        
        severity = self._determine_severity(overall_score, all_issues)
        
        return {
            "task_id": task_id,
            "index": index,
            "score": overall_score,
            "confidence": confidence,
            "quality_breakdown": quality_scores,
            "issues": all_issues,
            "severity": severity,
            "needs_retry": overall_score < 0.6 or severity in ["high", "critical"],
            "improvement_suggestions": self._generate_improvement_suggestions(
                quality_scores, all_issues
            )
        }
    
    def _assess_summary_quality(self, summary_text: str, goal: str) -> Dict[str, float]:
        if not summary_text:
            return {
                "coherence": 0.0,
                "completeness": 0.0,
                "relevance": 0.5,
                "conciseness": 0.0,
                "accuracy": 0.5
            }
        
        word_count = len(summary_text.split())
        sentence_count = len(re.split(r'[.!?]+', summary_text))
        
        coherence_score = min(1.0, max(0.1, sentence_count / max(1, word_count / 12)))
        
        completeness_score = min(1.0, max(0.2, word_count / 150)) if word_count < 300 else 0.8
        
        relevance_score = 0.7  # Would need semantic analysis for real assessment
        if goal and any(word.lower() in summary_text.lower() for word in goal.split()[:3]):
            relevance_score = min(1.0, relevance_score + 0.2)
        
        conciseness_score = 1.0 if word_count <= 200 else max(0.3, 1.0 - (word_count - 200) / 500)
        
        accuracy_score = 0.8
        
        return {
            "coherence": coherence_score,
            "completeness": completeness_score,
            "relevance": relevance_score,
            "conciseness": conciseness_score,
            "accuracy": accuracy_score
        }
    
    def _detect_summary_issues(self, summary: Dict[str, Any], goal: str) -> List[str]:
        issues = []
        summary_text = summary.get("summary", "")
        confidence = summary.get("confidence", 1.0)
        
        if confidence < 0.5:
            issues.append("very low confidence")
        elif confidence < 0.7:
            issues.append("low confidence")
        
        word_count = len(summary_text.split())
        if word_count < 20:
            issues.append("too short")
        elif word_count > 500:
            issues.append("too verbose")
        
        if not summary_text.strip():
            issues.append("empty summary")
        
        if re.search(r'\b(I think|maybe|possibly|unclear)\b', summary_text, re.IGNORECASE):
            issues.append("contains uncertainty markers")
        
        if summary_text.count('.') < 2 and word_count > 50:
            issues.append("lack of sentence structure")
        
        words = summary_text.lower().split()
        if len(words) != len(set(words)) and len(words) > 10:
            word_freq = {}
            for word in words:
                word_freq[word] = word_freq.get(word, 0) + 1
            max_freq = max(word_freq.values())
            if max_freq > len(words) * 0.2:
                issues.append("excessive repetition")
        
        if goal and len(goal.split()) > 1:
            goal_words = set(goal.lower().split())
            summary_words = set(summary_text.lower().split())
            overlap = len(goal_words.intersection(summary_words))
            if overlap == 0:
                issues.append("may not address goal")
        
        return issues
    
    def _calculate_overall_score(self, quality_scores: Dict[str, float], 
                               confidence: float, issues: List[str]) -> float:
        quality_avg = sum(quality_scores.values()) / len(quality_scores)
        
        base_score = (quality_avg * 0.7) + (confidence * 0.3)
        
        issue_penalty = 0
        critical_issues = ["empty summary", "very low confidence", "off-topic"]
        major_issues = ["too short", "excessive repetition", "incoherent"]
        
        for issue in issues:
            if any(critical in issue.lower() for critical in critical_issues):
                issue_penalty += 0.3
            elif any(major in issue.lower() for major in major_issues):
                issue_penalty += 0.15
            else:
                issue_penalty += 0.05
        
        final_score = max(0.0, base_score - issue_penalty)
        return min(1.0, final_score)
    
    def _determine_severity(self, score: float, issues: List[str]) -> str:
        critical_issues = ["empty summary", "factual error", "hallucination"]
        high_issues = ["very low confidence", "incoherent", "off-topic"]
        
        if any(critical in str(issues).lower() for critical in critical_issues):
            return "critical"
        elif score < 0.3 or any(high in str(issues).lower() for high in high_issues):
            return "high"
        elif score < 0.6 or len(issues) > 3:
            return "medium"
        elif score < 0.8 or len(issues) > 1:
            return "low"
        else:
            return "none"
    
    def _generate_improvement_suggestions(self, quality_scores: Dict[str, float], 
                                        issues: List[str]) -> List[str]:
        suggestions = []
        
        if quality_scores.get("coherence", 1.0) < 0.6:
            suggestions.append("Improve logical flow and sentence structure")
        
        if quality_scores.get("completeness", 1.0) < 0.6:
            suggestions.append("Include more key details and main points")
        
        if quality_scores.get("relevance", 1.0) < 0.6:
            suggestions.append("Focus more closely on the specified goal")
        
        if quality_scores.get("conciseness", 1.0) < 0.6:
            suggestions.append("Be more concise and eliminate unnecessary details")
        
        for issue in issues:
            if "too short" in issue:
                suggestions.append("Expand with more detail and examples")
            elif "too verbose" in issue:
                suggestions.append("Condense to focus on essential points")
            elif "repetition" in issue:
                suggestions.append("Remove repetitive content")
            elif "confidence" in issue:
                suggestions.append("Be more specific and factual")
            elif "uncertainty" in issue:
                suggestions.append("Use more definitive language")
        
        return list(set(suggestions))
    
    def compare_summaries(self, original_summaries: List[Dict[str, Any]], 
                         retry_summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compare original and retry summaries to assess improvement."""
        self.log_reasoning("summary_comparison", 
                          f"Comparing {len(original_summaries)} original vs {len(retry_summaries)} retry summaries")
        
        improvements = []
        regressions = []
        
        retry_map = {}
        for retry in retry_summaries:
            original_task_id = retry.get("retry_metadata", {}).get("original_task_id")
            if original_task_id:
                retry_map[original_task_id] = retry
        
        for original in original_summaries:
            task_id = original.get("task_id")
            if task_id in retry_map:
                retry = retry_map[task_id]
                comparison = self._compare_single_summary(original, retry)
                
                if comparison["improved"]:
                    improvements.append(comparison)
                elif comparison["regressed"]:
                    regressions.append(comparison)
        
        return {
            "total_comparisons": len(retry_map),
            "improvements": improvements,
            "regressions": regressions,
            "improvement_rate": len(improvements) / len(retry_map) if retry_map else 0,
            "regression_rate": len(regressions) / len(retry_map) if retry_map else 0
        }
    
    def _compare_single_summary(self, original: Dict[str, Any], 
                               retry: Dict[str, Any]) -> Dict[str, Any]:
        original_conf = original.get("confidence", 0.5)
        retry_conf = retry.get("confidence", 0.5)
        
        original_issues = len(original.get("issues", []))
        retry_issues = len(retry.get("issues", []))
        
        confidence_improved = retry_conf > original_conf + 0.1
        issues_reduced = retry_issues < original_issues
        
        improved = confidence_improved or issues_reduced
        regressed = retry_conf < original_conf - 0.1 or retry_issues > original_issues
        
        return {
            "task_id": original.get("task_id"),
            "improved": improved,
            "regressed": regressed,
            "confidence_change": retry_conf - original_conf,
            "issues_change": retry_issues - original_issues,
            "original_confidence": original_conf,
            "retry_confidence": retry_conf,
            "improvements_made": [] if not improved else [
                "confidence increased" if confidence_improved else "",
                "issues reduced" if issues_reduced else ""
            ]
        }