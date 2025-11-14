import logging
import time
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
    qa_start = time.time()
    
    try:
        logger.info(f"[QA] Processing question: \"{question}\"")
        logger.debug(f"[QA] Question length: {len(question)} chars")
        
        # Step 1: Retrieve top-K relevant messages
        retrieval_start = time.time()
        logger.info(f"[QA] Step 1/2: Retrieving relevant messages...")
        messages = await retrieve_relevant_messages(question)
        retrieval_time = time.time() - retrieval_start
        
        if not messages:
            qa_time = time.time() - qa_start
            logger.warning(f"[QA] No messages retrieved for question after {retrieval_time:.2f}s")
            logger.info(f"[QA] Returning 'No information found.' (total: {qa_time:.2f}s)")
            return "No information found."
        
        logger.info(f"[QA] Step 1/2: Retrieved {len(messages)} messages in {retrieval_time:.2f}s")
        logger.debug(f"[QA] Retrieved message IDs: {[msg.id for msg in messages]}")
        
        # Step 2: Extract answer using LLM
        extraction_start = time.time()
        logger.info(f"[QA] Step 2/2: Extracting answer using LLM...")
        answer = await extract_answer(question, messages)
        extraction_time = time.time() - extraction_start
        
        qa_time = time.time() - qa_start
        
        answer_preview = answer[:100] + "..." if len(answer) > 100 else answer
        logger.info(f"[QA] Step 2/2: Extracted answer in {extraction_time:.2f}s")
        logger.info(f"[QA] Generated answer: \"{answer_preview}\" in {qa_time:.2f}s total")
        logger.debug(f"[QA] Full answer: \"{answer}\"")
        logger.debug(f"[QA] Timing breakdown: retrieval={retrieval_time:.2f}s, "
                    f"extraction={extraction_time:.2f}s, total={qa_time:.2f}s")
        
        return answer
        
    except Exception as e:
        qa_time = time.time() - qa_start
        logger.error(f"[QA] Error in QA engine after {qa_time:.2f}s: {e}", exc_info=True)
        # Return safe fallback on error
        return "No information found."

