import httpx
import logging
import time
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
    
    logger.debug(f"[LLM] Built prompt: length={len(prompt)} chars, "
               f"question_length={len(question)}, messages_count={len(messages)}")
    logger.debug(f"[LLM] Prompt preview (first 200 chars): {prompt[:200]}...")
    
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
    extract_start = time.time()
    
    if not messages:
        logger.warning("[LLM] No messages provided, returning 'No information found.'")
        return "No information found."
    
    logger.info(f"[LLM] Building prompt with {len(messages)} messages...")
    logger.debug(f"[LLM] Question: \"{question}\"")
    logger.debug(f"[LLM] System prompt length: {len(SYSTEM_PROMPT)} chars")
    
    user_prompt = build_user_prompt(question, messages)
    
    url = f"{GROQ_API_BASE}/chat/completions"
    headers = {
        "Authorization": f"Bearer {settings.groq_api_key[:10]}...",  # Partial key for logging
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": settings.groq_model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.0,  # Deterministic responses
        "max_tokens": 150   # Force short answers
    }
    
    logger.info(f"[LLM] Calling Groq API (model={settings.groq_model}, "
               f"temperature={payload['temperature']}, max_tokens={payload['max_tokens']})...")
    logger.debug(f"[LLM] API endpoint: {url}")
    logger.debug(f"[LLM] Request payload size: {len(str(payload))} chars")
    
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            api_start = time.time()
            response = await client.post(url, json=payload, headers=headers)
            api_time = time.time() - api_start
            response.raise_for_status()
            
            logger.debug(f"[LLM] API call completed in {api_time:.2f}s (status: {response.status_code})")
            
            data = response.json()
            
            # Groq uses OpenAI-compatible response format
            answer = data["choices"][0]["message"]["content"].strip()
            
            logger.debug(f"[LLM] Raw answer from API: \"{answer}\"")
            logger.debug(f"[LLM] Response structure: {list(data.keys())}")
            if "usage" in data:
                logger.debug(f"[LLM] Token usage: {data['usage']}")
            
            # Ensure it's a single sentence
            original_answer = answer
            if answer.endswith('.'):
                pass
            elif not answer.endswith(('.', '!', '?')):
                answer += '.'
            
            if answer != original_answer:
                logger.debug(f"[LLM] Added punctuation to answer")
            
            extract_time = time.time() - extract_start
            answer_preview = answer[:100] + "..." if len(answer) > 100 else answer
            logger.info(f"[LLM] Extracted answer: \"{answer_preview}\" in {extract_time:.2f}s")
            logger.debug(f"[LLM] Full answer: \"{answer}\"")
            
            return answer
            
    except httpx.HTTPStatusError as e:
        extract_time = time.time() - extract_start
        logger.error(f"[LLM] HTTP error extracting answer after {extract_time:.2f}s: {e}", exc_info=True)
        # Fallback to "No information found" on error
        return "No information found."
    except Exception as e:
        extract_time = time.time() - extract_start
        logger.error(f"[LLM] Error extracting answer after {extract_time:.2f}s: {e}", exc_info=True)
        # Fallback to "No information found" on error
        return "No information found."
