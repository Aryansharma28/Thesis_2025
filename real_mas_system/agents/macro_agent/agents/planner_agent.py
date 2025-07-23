"""
Planner Agent - Decomposes documents into optimal summarization tasks.
"""
import json
import re
from typing import Dict, Any, List
from .base_agent import BaseAgent
from ..prompts.planner_prompts import PLANNER_SYSTEM_PROMPT, get_planner_prompt


class PlannerAgent(BaseAgent):
    """Analyzes documents and creates summarization plans."""
    
    def __init__(self, model: str = "gpt-4", temperature: float = 0.1, max_chunk_size: int = 2000):
        super().__init__("PlannerAgent", model, temperature)
        self.max_chunk_size = max_chunk_size
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a summarization plan for the given document."""
        document = input_data["document"]
        goal = input_data["goal"]
        max_chunk_size = input_data.get("max_chunk_size", self.max_chunk_size)
        
        self.log_reasoning("input_analysis", f"Document length: {len(document)} chars, Goal: {goal}")
        
        structure_analysis = self._analyze_document_structure(document)
        self.log_reasoning("structure_analysis", f"Found {structure_analysis['sections']} sections, "
                          f"{structure_analysis['paragraphs']} paragraphs")
        
        messages = [
            {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
            {"role": "user", "content": get_planner_prompt(document, goal, max_chunk_size)}
        ]
        
        self.log_reasoning("llm_planning", "Generating plan with LLM")
        response = self._make_api_call(messages, max_tokens=2000)
        
        try:
            plan = self._parse_plan_response(response)
            self.log_reasoning("plan_validation", f"Generated {len(plan['tasks'])} tasks")
            
            plan = self._validate_and_adjust_plan(plan, document, max_chunk_size)
            self.log_reasoning("plan_adjustment", f"Final plan has {len(plan['tasks'])} tasks")
            
        except Exception as e:
            self.log_reasoning("fallback_planning", f"LLM parsing failed: {e}, using fallback")
            plan = self._create_fallback_plan(document, goal, max_chunk_size)
        
        return {
            "plan": plan,
            "reasoning_log": self.get_reasoning_log(),
            "stats": self.get_stats()
        }
    
    def _analyze_document_structure(self, document: str) -> Dict[str, int]:
        lines = document.split('\n')
        paragraphs = [p for p in document.split('\n\n') if p.strip()]
        
        potential_headers = []
        for line in lines:
            line = line.strip()
            if line and len(line) < 100 and (line.isupper() or line.istitle()):
                potential_headers.append(line)
        
        return {
            "lines": len(lines),
            "paragraphs": len(paragraphs),
            "sections": len(potential_headers),
            "chars": len(document)
        }
    
    def _parse_plan_response(self, response: str) -> Dict[str, Any]:
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            plan = json.loads(json_str)
            
            required_fields = ["analysis", "strategy", "tasks", "synthesis_guidance"]
            for field in required_fields:
                if field not in plan:
                    raise ValueError(f"Missing required field: {field}")
            
            for i, task in enumerate(plan["tasks"]):
                required_task_fields = ["task_id", "content", "instructions", "priority"]
                for field in required_task_fields:
                    if field not in task:
                        raise ValueError(f"Task {i} missing field: {field}")
            
            return plan
        else:
            raise ValueError("No JSON found in response")
    
    def _validate_and_adjust_plan(self, plan: Dict[str, Any], document: str, 
                                 max_chunk_size: int) -> Dict[str, Any]:
        adjusted_tasks = []
        
        for task in plan["tasks"]:
            content = task["content"]
            
            if len(content) > max_chunk_size:
                self.log_reasoning("chunk_splitting", f"Splitting large chunk of {len(content)} chars")
                sub_chunks = self._split_content(content, max_chunk_size)
                
                for i, sub_chunk in enumerate(sub_chunks):
                    adjusted_task = task.copy()
                    adjusted_task["task_id"] = f"{task['task_id']}_part_{i+1}"
                    adjusted_task["content"] = sub_chunk
                    adjusted_task["instructions"] = f"{task['instructions']} (Part {i+1} of {len(sub_chunks)})"
                    adjusted_tasks.append(adjusted_task)
            else:
                adjusted_tasks.append(task)
        
        plan["tasks"] = adjusted_tasks
        return plan
    
    def _split_content(self, content: str, max_size: int) -> List[str]:
        sentences = re.split(r'(?<=[.!?])\s+', content)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= max_size:
                current_chunk += sentence + " "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + " "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _create_fallback_plan(self, document: str, goal: str, max_chunk_size: int) -> Dict[str, Any]:
        self.log_reasoning("fallback_creation", "Creating simple chunk-based fallback plan")
        
        paragraphs = [p.strip() for p in document.split('\n\n') if p.strip()]
        
        tasks = []
        current_chunk = ""
        chunk_id = 1
        
        for paragraph in paragraphs:
            if len(current_chunk) + len(paragraph) <= max_chunk_size:
                current_chunk += paragraph + "\n\n"
            else:
                if current_chunk:
                    tasks.append({
                        "task_id": f"chunk_{chunk_id}",
                        "content": current_chunk.strip(),
                        "instructions": f"Summarize this section focusing on key points relevant to: {goal}",
                        "priority": "medium",
                        "context": f"This is part {chunk_id} of the document"
                    })
                    chunk_id += 1
                current_chunk = paragraph + "\n\n"
        
        if current_chunk:
            tasks.append({
                "task_id": f"chunk_{chunk_id}",
                "content": current_chunk.strip(),
                "instructions": f"Summarize this section focusing on key points relevant to: {goal}",
                "priority": "medium",
                "context": f"This is part {chunk_id} of the document"
            })
        
        return {
            "analysis": f"Document contains {len(paragraphs)} paragraphs, {len(document)} characters",
            "strategy": "Simple paragraph-based chunking due to LLM parsing failure",
            "tasks": tasks,
            "synthesis_guidance": f"Combine all summaries into a coherent summary for: {goal}"
        }