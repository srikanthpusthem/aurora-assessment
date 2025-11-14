import logging
from typing import List
from app.models.message import Message
from app.core.embeddings import embed_text
from app.services.indexing import get_index
from app.core.config import settings

logger = logging.getLogger(__name__)


async def retrieve_relevant_messages(question: str, k: int = None) -> List[Message]:
    """
    Retrieve top-K relevant messages for a given question.
    
    Steps:
    1. Embed the incoming question
    2. Query FAISS index
    3. Return top-K relevant messages
    
    Args:
        question: The user's question
        k: Number of messages to retrieve (defaults to settings.top_k)
        
    Returns:
        List of Message objects sorted by relevance
    """
    if k is None:
        k = settings.top_k
    
    try:
        # Embed the question
        logger.info(f"Embedding question: {question[:50]}...")
        question_embedding = await embed_text(question)
        
        # Get the index and search
        index = get_index()
        if not index.is_ready():
            raise RuntimeError("Index not ready. Ensure index is built on startup.")
        
        # Search for top-K messages
        results = index.search(question_embedding, k=k)
        
        # Extract messages (ignore distance scores for now)
        messages = [msg for msg, _ in results]
        
        logger.info(f"Retrieved {len(messages)} relevant messages")
        return messages
        
    except Exception as e:
        logger.error(f"Error retrieving messages: {e}")
        raise


def format_messages_for_prompt(messages: List[Message]) -> str:
    """
    Format messages for LLM prompt.
    
    Args:
        messages: List of Message objects
        
    Returns:
        Formatted string for prompt
    """
    formatted = []
    for i, msg in enumerate(messages, 1):
        formatted.append(f"Message {i} (User: {msg.user_id}): {msg.text}")
    
    return "\n".join(formatted)

