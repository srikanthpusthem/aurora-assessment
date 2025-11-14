import logging
import time
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
    retrieval_start = time.time()
    
    if k is None:
        k = settings.top_k
    
    logger.info(f"[RETRIEVAL] Starting retrieval for question: \"{question}\"")
    logger.debug(f"[RETRIEVAL] Retrieval parameters: k={k}")
    
    try:
        # Embed the question
        embed_start = time.time()
        logger.info(f"[RETRIEVAL] Embedding question...")
        question_embedding = await embed_text(question)
        embed_time = time.time() - embed_start
        
        logger.info(f"[RETRIEVAL] Question embedding: shape={question_embedding.shape}, "
                   f"dtype={question_embedding.dtype} in {embed_time:.2f}s")
        logger.debug(f"[RETRIEVAL] Sample embedding (first 5 values): {question_embedding[:5].tolist()}")
        
        # Get the index and search
        index = get_index()
        if not index.is_ready():
            raise RuntimeError("Index not ready. Ensure index is built on startup.")
        
        logger.debug(f"[RETRIEVAL] Index is ready, searching with k={k}...")
        
        # Search for top-K messages
        search_start = time.time()
        results = index.search(question_embedding, k=k)
        search_time = time.time() - search_start
        
        # Log search results
        if results:
            distances = [dist for _, dist in results]
            logger.info(f"[RETRIEVAL] Found {len(results)} results: distances={[f'{d:.4f}' for d in distances]} in {search_time:.4f}s")
            
            logger.debug(f"[RETRIEVAL] Retrieved messages:")
            for i, (msg, distance) in enumerate(results, 1):
                msg_preview = msg.text[:50] + "..." if len(msg.text) > 50 else msg.text
                logger.debug(f"[RETRIEVAL]   {i}. Message(id='{msg.id}', user_id='{msg.user_id}', "
                           f"distance={distance:.4f}, text='{msg_preview}')")
        else:
            logger.warning(f"[RETRIEVAL] No results found in {search_time:.4f}s")
        
        # Extract messages (ignore distance scores for now)
        messages = [msg for msg, _ in results]
        
        retrieval_time = time.time() - retrieval_start
        logger.info(f"[RETRIEVAL] Retrieved {len(messages)} relevant messages in {retrieval_time:.2f}s total")
        
        return messages
        
    except Exception as e:
        retrieval_time = time.time() - retrieval_start
        logger.error(f"[RETRIEVAL] Error retrieving messages after {retrieval_time:.2f}s: {e}", exc_info=True)
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

