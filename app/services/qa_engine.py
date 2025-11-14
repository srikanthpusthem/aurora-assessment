import logging
from app.services.retrieval import retrieve_relevant_messages
from app.core.llm import extract_answer

logger = logging.getLogger(__name__)


async def answer_question(question: str) -> str:
    """
    High-level QA engine orchestrator.
    
    Steps:
    1. Retrieve top-K messages using vector search
    2. Pass question + messages to LLM extractor
    3. Return final answer
    
    Args:
        question: The user's question
        
    Returns:
        Final answer string
    """
    try:
        logger.info(f"Processing question: {question[:100]}...")
        
        # Step 1: Retrieve top-K relevant messages
        messages = await retrieve_relevant_messages(question)
        
        if not messages:
            logger.warning("No messages retrieved for question")
            return "No information found."
        
        # Step 2: Extract answer using LLM
        answer = await extract_answer(question, messages)
        
        logger.info(f"Generated answer: {answer[:100]}...")
        return answer
        
    except Exception as e:
        logger.error(f"Error in QA engine: {e}")
        # Return safe fallback on error
        return "No information found."

