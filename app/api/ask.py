from fastapi import APIRouter, HTTPException
import logging
import time
from app.models.request import AskRequest
from app.models.response import AskResponse
from app.services.qa_engine import answer_question

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest) -> AskResponse:
    """
    POST /api/ask endpoint to answer questions about users.
    
    Args:
        request: AskRequest containing the question
        
    Returns:
        AskResponse with the answer
        
    Raises:
        HTTPException: If processing fails
    """
    request_start = time.time()
    try:
        # Validate request (Pydantic handles this automatically)
        question = request.question.strip()
        
        logger.info(f"[REQUEST] POST /api/ask - Question: \"{question}\"")
        logger.debug(f"[REQUEST] Full request: {request.dict()}")
        
        if not question:
            logger.warning("[REQUEST] Empty question received")
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        # Call QA engine
        answer = await answer_question(question)
        
        # Calculate processing time
        processing_time = time.time() - request_start
        
        # Log response
        answer_preview = answer[:100] + "..." if len(answer) > 100 else answer
        logger.info(f"[RESPONSE] 200 OK - Answer: \"{answer_preview}\" ({processing_time:.2f}s total)")
        logger.debug(f"[RESPONSE] Full answer: \"{answer}\"")
        
        # Return response
        return AskResponse(answer=answer)
        
    except HTTPException as e:
        processing_time = time.time() - request_start
        logger.warning(f"[RESPONSE] {e.status_code} {e.detail} ({processing_time:.2f}s)")
        raise
    except Exception as e:
        processing_time = time.time() - request_start
        logger.error(f"[RESPONSE] 500 Internal Server Error after {processing_time:.2f}s: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Internal server error while processing question"
        )

