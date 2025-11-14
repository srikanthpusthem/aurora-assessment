import httpx
import logging
from typing import List
from app.core.config import settings
from app.models.message import Message

logger = logging.getLogger(__name__)

# Groq API endpoint
GROQ_API_BASE = "https://api.groq.com/openai/v1"

SYSTEM_PROMPT = """You are an information extraction system. 
You must answer strictly based on the provided messages. 
If the information is not present, reply: "No information found."
"""


def build_user_prompt(question: str, messages: List[Message]) -> str:
    """
    Build user prompt template with question and retrieved messages.
    
    Args:
        question: The user's question
        messages: List of retrieved Message objects
        
    Returns:
        Formatted prompt string
    """
    messages_text = "\n".join([
        f"Message {i+1} (User: {msg.user_id}): {msg.text}"
        for i, msg in enumerate(messages)
    ])
    
    prompt = f"""Question: {question}

Relevant messages:

{messages_text}

Answer in one short factually correct sentence."""
    
    return prompt


async def extract_answer(question: str, messages: List[Message]) -> str:
    """
    Extract factual answer from question and retrieved messages using Groq API.
    
    Args:
        question: The user's question
        messages: List of retrieved Message objects
        
    Returns:
        Single-sentence factual answer or "No information found."
    """
    if not messages:
        return "No information found."
    
    url = f"{GROQ_API_BASE}/chat/completions"
    headers = {
        "Authorization": f"Bearer {settings.groq_api_key}",
        "Content-Type": "application/json"
    }
    
    user_prompt = build_user_prompt(question, messages)
    
    payload = {
        "model": settings.groq_model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.0,  # Deterministic responses
        "max_tokens": 150   # Force short answers
    }
    
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            
            data = response.json()
            
            # Groq uses OpenAI-compatible response format
            answer = data["choices"][0]["message"]["content"].strip()
            
            # Ensure it's a single sentence
            if answer.endswith('.'):
                pass
            elif not answer.endswith(('.', '!', '?')):
                answer += '.'
            
            logger.info(f"Extracted answer: {answer[:100]}...")
            return answer
            
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error extracting answer: {e}")
        # Fallback to "No information found" on error
        return "No information found."
    except Exception as e:
        logger.error(f"Error extracting answer: {e}")
        # Fallback to "No information found" on error
        return "No information found."
