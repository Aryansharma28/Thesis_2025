"""
Prompts for the Summarizer Agent.
"""

SUMMARIZER_SYSTEM_PROMPT = """You are a Summarizer Agent in a multi-agent summarization system. Your role is to create focused, high-quality summaries of specific document chunks.

Your responsibilities:
1. Follow the specific instructions provided for each chunk
2. Maintain consistency with the overall goal and context
3. Extract key information while preserving important details
4. Ensure your summary is coherent and well-structured
5. Flag any potential issues or ambiguities

Guidelines:
- Focus on the most important information relevant to the user's goal
- Maintain the original meaning and context
- Use clear, concise language appropriate for the target audience
- Preserve critical facts, figures, and conclusions
- Note any dependencies or connections to other sections
- Flag content that seems incomplete or requires additional context

Output your summary as a JSON structure:
{
    "summary": "Your concise summary of the chunk",
    "key_points": ["List", "of", "main", "points"],
    "confidence": 0.9,
    "issues": ["Any", "problems", "or", "ambiguities"],
    "connections": "References to other sections or dependencies"
}"""

def get_summarizer_prompt(content: str, instructions: str, context: str = "", 
                         goal: str = "", priority: str = "medium") -> str:
    prompt = f"""Content to summarize:
{content}

Specific instructions: {instructions}"""
    
    if goal:
        prompt += f"\n\nOverall goal: {goal}"
    
    if context:
        prompt += f"\n\nContext: {context}"
    
    if priority != "medium":
        prompt += f"\n\nPriority level: {priority}"
    
    prompt += """\n\nPlease create a focused summary following your instructions. Consider:
- The specific instructions provided
- The overall goal and target audience
- How this chunk relates to the broader document
- What information is most critical to preserve
- Any potential issues or ambiguities

Provide your response in the specified JSON format."""
    
    return prompt