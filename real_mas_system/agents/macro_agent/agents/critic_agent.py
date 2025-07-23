"""
Critic Agent - Evaluates and synthesizes individual summaries into final output.
"""
import json
import re
from typing import Dict, Any, List
from .base_agent import BaseAgent
from ..prompts.critic_prompts import CRITIC_SYSTEM_PROMPT, get_critic_prompt


class CriticAgent(BaseAgent):
    """Evaluates summaries and creates unified final output."""
    
    def __init__(self, model: str = "gpt-4", temperature: float = 0.2):
        super().__init__("CriticAgent", model, temperature)
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate summaries and create final unified summary."""
        summaries = input_data["summaries"]
        goal = input_data["goal"]
        synthesis_guidance = input_data.get("synthesis_guidance", "")
        original_length = input_data.get("original_document_length", 0)
        
        self.log_reasoning("input_analysis", 
                          f"Evaluating {len(summaries)} summaries for goal: {goal}")
        
        initial_analysis = self._analyze_summaries(summaries)
        self.log_reasoning("initial_analysis", 
                          f"Average confidence: {initial_analysis['avg_confidence']:.2f}, "
                          f"Total issues: {initial_analysis['total_issues']}")
        
        messages = [
            {"role": "system", "content": CRITIC_SYSTEM_PROMPT},
            {"role": "user", "content": get_critic_prompt(summaries, goal, synthesis_guidance)}
        ]
        
        self.log_reasoning("llm_evaluation", "Generating evaluation and synthesis with LLM")
        response = self._make_api_call(messages, max_tokens=3000)
        
        try:
            result = self._parse_critic_response(response)
            self.log_reasoning("synthesis_complete", 
                             f"Final summary: {len(result['final_summary'])} chars, "
                             f"Confidence: {result.get('confidence', 'N/A')}")
            
            result["metadata"] = {
                "original_document_length": original_length,
                "num_summaries_processed": len(summaries),
                "compression_ratio": len(result["final_summary"]) / original_length if original_length > 0 else 0,
                "initial_analysis": initial_analysis
            }
            
        except Exception as e:
            self.log_reasoning("fallback_synthesis", f"LLM parsing failed: {e}, using fallback")
            result = self._create_fallback_synthesis(summaries, goal, str(e))
        
        return {
            "evaluation": result.get("evaluation", {}),
            "final_summary": result["final_summary"],
            "metadata": result.get("metadata", {}),
            "reasoning_log": self.get_reasoning_log(),
            "stats": self.get_stats()
        }
    
    def _analyze_summaries(self, summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not summaries:
            return {"avg_confidence": 0, "total_issues": 0, "total_length": 0}
        
        confidences = [s.get("confidence", 0.5) for s in summaries]
        issues = [issue for s in summaries for issue in s.get("issues", [])]
        total_length = sum(len(s.get("summary", "")) for s in summaries)
        
        return {
            "avg_confidence": sum(confidences) / len(confidences),
            "total_issues": len(issues),
            "total_length": total_length,
            "num_summaries": len(summaries),
            "issue_types": list(set(issues))
        }
    
    def _parse_critic_response(self, response: str) -> Dict[str, Any]:
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            result = json.loads(json_str)
            
            if "final_summary" not in result:
                raise ValueError("Missing final_summary field")
            
            result.setdefault("evaluation", {})
            result.setdefault("issues_found", {})
            result.setdefault("confidence", 0.7)
            result.setdefault("recommendations", [])
            
            return result
        else:
            raise ValueError("No JSON found in response")
    
    def _create_fallback_synthesis(self, summaries: List[Dict[str, Any]], 
                                  goal: str, error_msg: str) -> Dict[str, Any]:
        self.log_reasoning("fallback_creation", "Creating simple concatenation fallback")
        
        summary_texts = []
        seen_content = set()
        
        for summary in summaries:
            summary_text = summary.get("summary", "")
            
            if summary_text and not self._is_duplicate_content(summary_text, seen_content):
                summary_texts.append(summary_text)
                seen_content.add(summary_text.lower()[:100])
        
        final_summary = " ".join(summary_texts)
        
        avg_confidence = sum(s.get("confidence", 0.5) for s in summaries) / len(summaries) if summaries else 0
        total_issues = sum(len(s.get("issues", [])) for s in summaries)
        
        return {
            "evaluation": {
                "overall_quality": max(0.3, avg_confidence - 0.2),
                "synthesis_quality": {
                    "coherence": 0.4,
                    "completeness": 0.6,
                    "consistency": 0.5
                }
            },
            "issues_found": {
                "processing_errors": [f"Critic agent failed: {error_msg}"],
                "synthesis_method": ["Used fallback concatenation method"]
            },
            "final_summary": final_summary,
            "confidence": max(0.3, avg_confidence - 0.3),
            "recommendations": [
                "Review original summaries for quality",
                "Consider manual review due to processing failure"
            ]
        }
    
    def _is_duplicate_content(self, text: str, seen_content: set) -> bool:
        text_signature = text.lower()[:100]
        
        for seen in seen_content:
            text_words = set(text_signature.split())
            seen_words = set(seen.split())
            
            if len(text_words) > 0:
                overlap = len(text_words.intersection(seen_words)) / len(text_words)
                if overlap > 0.7:
                    return True
        
        return False
    
    def evaluate_summary_quality(self, summary: str, goal: str) -> Dict[str, float]:
        word_count = len(summary.split())
        sentence_count = len(re.split(r'[.!?]+', summary))
        
        scores = {
            "length_appropriateness": min(1.0, max(0.1, word_count / 200)),
            "sentence_structure": min(1.0, sentence_count / max(1, word_count / 15)),
            "goal_relevance": 0.8,
            "coherence": 0.8,
            "completeness": 0.8
        }
        
        scores["overall"] = sum(scores.values()) / len(scores)
        return scores