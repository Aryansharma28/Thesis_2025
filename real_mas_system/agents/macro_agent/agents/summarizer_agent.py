"""
Summarizer Agent - Creates focused summaries of document chunks.
"""
import json
import re
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from .base_agent import BaseAgent
from ..prompts.summarizer_prompts import SUMMARIZER_SYSTEM_PROMPT, get_summarizer_prompt


class SummarizerAgent(BaseAgent):
    """Creates summaries of individual document chunks."""
    
    def __init__(self, model: str = "gpt-4", temperature: float = 0.3, 
                 parallel_processing: bool = True, max_workers: int = 3):
        super().__init__("SummarizerAgent", model, temperature)
        self.parallel_processing = parallel_processing
        self.max_workers = max_workers
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize all tasks from the planner."""
        tasks = input_data["tasks"]
        goal = input_data.get("goal", "")
        parallel = input_data.get("parallel", self.parallel_processing)
        
        self.log_reasoning("input_analysis", f"Processing {len(tasks)} summarization tasks")
        
        if parallel and len(tasks) > 1:
            summaries = self._process_tasks_parallel(tasks, goal)
        else:
            summaries = self._process_tasks_sequential(tasks, goal)
        
        self.log_reasoning("completion", f"Generated {len(summaries)} summaries")
        
        return {
            "summaries": summaries,
            "reasoning_log": self.get_reasoning_log(),
            "stats": self.get_stats()
        }
    
    def _process_tasks_sequential(self, tasks: List[Dict[str, Any]], goal: str) -> List[Dict[str, Any]]:
        self.log_reasoning("processing_mode", "Sequential processing")
        summaries = []
        
        for i, task in enumerate(tasks):
            self.log_reasoning("task_start", f"Processing task {i+1}/{len(tasks)}: {task['task_id']}")
            summary = self._summarize_single_task(task, goal)
            summaries.append(summary)
        
        return summaries
    
    def _process_tasks_parallel(self, tasks: List[Dict[str, Any]], goal: str) -> List[Dict[str, Any]]:
        self.log_reasoning("processing_mode", f"Parallel processing with {self.max_workers} workers")
        summaries = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_task = {
                executor.submit(self._summarize_single_task, task, goal): task 
                for task in tasks
            }
            
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    summary = future.result()
                    summaries.append(summary)
                    self.log_reasoning("task_complete", f"Completed task: {task['task_id']}")
                except Exception as e:
                    self.log_reasoning("task_error", f"Error in task {task['task_id']}: {e}")
                    fallback_summary = self._create_fallback_summary(task, str(e))
                    summaries.append(fallback_summary)
        
        task_order = {task["task_id"]: i for i, task in enumerate(tasks)}
        summaries.sort(key=lambda s: task_order.get(s["task_id"], float('inf')))
        
        return summaries
    
    def _summarize_single_task(self, task: Dict[str, Any], goal: str) -> Dict[str, Any]:
        content = task["content"]
        instructions = task["instructions"]
        context = task.get("context", "")
        priority = task.get("priority", "medium")
        task_id = task["task_id"]
        
        prompt = get_summarizer_prompt(content, instructions, context, goal, priority)
        
        messages = [
            {"role": "system", "content": SUMMARIZER_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
        
        response = self._make_api_call(messages, max_tokens=1000)
        
        try:
            summary_data = self._parse_summary_response(response)
            summary_data["task_id"] = task_id
            summary_data["original_content_length"] = len(content)
            return summary_data
        except Exception as e:
            return self._create_fallback_summary(task, f"Parsing error: {e}")
    
    def _parse_summary_response(self, response: str) -> Dict[str, Any]:
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            summary_data = json.loads(json_str)
            
            required_fields = ["summary", "key_points", "confidence"]
            for field in required_fields:
                if field not in summary_data:
                    raise ValueError(f"Missing required field: {field}")
            
            summary_data.setdefault("issues", [])
            summary_data.setdefault("connections", "")
            
            return summary_data
        else:
            summary_text = response.strip()
            return {
                "summary": summary_text,
                "key_points": self._extract_key_points(summary_text),
                "confidence": 0.7,
                "issues": ["Response not in expected JSON format"],
                "connections": ""
            }
    
    def _extract_key_points(self, summary: str) -> List[str]:
        sentences = re.split(r'[.!?]+', summary)
        key_points = []
        for sentence in sentences[:5]:
            sentence = sentence.strip()
            if len(sentence) > 20:
                key_points.append(sentence)
        return key_points
    
    def _create_fallback_summary(self, task: Dict[str, Any], error_msg: str) -> Dict[str, Any]:
        content = task["content"]
        
        sentences = re.split(r'[.!?]+', content)
        summary_sentences = []
        char_count = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if char_count + len(sentence) < 200 and len(sentence) > 10:
                summary_sentences.append(sentence)
                char_count += len(sentence)
            if len(summary_sentences) >= 3:
                break
        
        fallback_summary = ". ".join(summary_sentences) + "."
        
        return {
            "task_id": task["task_id"],
            "summary": fallback_summary,
            "key_points": summary_sentences,
            "confidence": 0.3,
            "issues": [f"Processing failed: {error_msg}", "Using fallback extraction"],
            "connections": "",
            "original_content_length": len(content)
        }