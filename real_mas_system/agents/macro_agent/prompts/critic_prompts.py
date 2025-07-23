"""
Prompts for the Critic Agent.
"""

CRITIC_SYSTEM_PROMPT = """You are a Critic Agent in a multi-agent summarization system. Your role is to evaluate, synthesize, and refine individual summaries into a cohesive final summary.

Your responsibilities:
1. Evaluate individual summaries for quality, accuracy, and relevance
2. Identify redundancies, gaps, and inconsistencies across summaries
3. Synthesize summaries into a coherent, unified final summary
4. Ensure the final summary meets the user's goal and requirements
5. Flag any issues or potential hallucinations

Evaluation criteria:
- Coherence: Does the summary flow logically?
- Completeness: Are key points covered adequately?
- Accuracy: Are facts and claims properly represented?
- Relevance: Does it address the user's specific goal?
- Conciseness: Is it appropriately detailed without redundancy?
- Consistency: Are there contradictions between sections?

Output your evaluation and final summary as JSON:
{
    "evaluation": {
        "overall_quality": 0.85,
        "individual_scores": [
            {"task_id": "chunk_1", "score": 0.9, "issues": ["minor inconsistency"]},
            {"task_id": "chunk_2", "score": 0.8, "issues": []}
        ],
        "synthesis_quality": {
            "coherence": 0.9,
            "completeness": 0.8,
            "consistency": 0.85
        }
    },
    "issues_found": {
        "redundancies": ["Repeated information about X"],
        "gaps": ["Missing conclusion from section Y"],
        "inconsistencies": ["Conflicting dates in sections 1 and 3"],
        "potential_hallucinations": []
    },
    "final_summary": "Your synthesized, polished final summary",
    "confidence": 0.85,
    "recommendations": ["Suggestions for improvement if needed"]
}"""

def get_critic_prompt(summaries: list, goal: str, synthesis_guidance: str = "") -> str:
    
    summaries_text = ""
    for i, summary in enumerate(summaries, 1):
        summaries_text += f"\n--- Summary {i} (Task: {summary.get('task_id', 'unknown')}) ---\n"
        summaries_text += f"Content: {summary.get('summary', '')}\n"
        summaries_text += f"Key Points: {', '.join(summary.get('key_points', []))}\n"
        summaries_text += f"Confidence: {summary.get('confidence', 'N/A')}\n"
        if summary.get('issues'):
            summaries_text += f"Issues: {', '.join(summary.get('issues', []))}\n"
        if summary.get('connections'):
            summaries_text += f"Connections: {summary.get('connections', '')}\n"
        summaries_text += "\n"
    
    prompt = f"""Individual summaries to evaluate and synthesize:
{summaries_text}

User Goal: {goal}"""
    
    if synthesis_guidance:
        prompt += f"\n\nSynthesis Guidance: {synthesis_guidance}"
    
    prompt += f"""\n\nPlease evaluate these {len(summaries)} individual summaries and create a unified final summary. Consider:

1. Quality Assessment:
   - Evaluate each summary for accuracy, relevance, and coherence
   - Identify any potential issues or inconsistencies
   - Check for redundancies across summaries

2. Synthesis Strategy:
   - How to best combine the information
   - What order and structure to use
   - How to resolve any conflicts or gaps

3. Final Summary Creation:
   - Create a cohesive summary that meets the user's goal
   - Ensure appropriate length and detail level
   - Maintain accuracy while improving flow and readability

Provide your complete evaluation and final summary in the specified JSON format."""
    
    return prompt