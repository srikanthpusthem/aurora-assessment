import httpx
import logging
import time
from typing import List
from app.models.message import Message
from app.core.config import settings

logger = logging.getLogger(__name__)


async def fetch_messages() -> List[Message]:
    """
    Fetch messages from Aurora's public API and normalize into Message objects.
    
    Returns:
        List of Message objects
        
    Raises:
        httpx.HTTPError: If the API request fails
        ValueError: If the response cannot be parsed
    """
    fetch_start = time.time()
    try:
        logger.info(f"[STEP 1/4] Fetching messages from Aurora API: {settings.messages_api_url}")
        
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            response = await client.get(settings.messages_api_url)
            response.raise_for_status()
            
            fetch_time = time.time() - fetch_start
            logger.info(f"[STEP 1/4] HTTP GET completed in {fetch_time:.2f}s (status: {response.status_code})")
            
            data = response.json()
            
            # API returns {total: int, items: [...]}
            if isinstance(data, dict) and "items" in data:
                items = data["items"]
                total = data.get('total', 'unknown')
                logger.info(f"[STEP 1/4] Received {len(items)} messages from API (total: {total})")
                logger.debug(f"[STEP 1/4] Response structure: dict with 'items' and 'total' keys")
            else:
                # Fallback: assume it's a list
                items = data if isinstance(data, list) else []
                logger.info(f"[STEP 1/4] Received {len(items)} messages from API")
                logger.debug(f"[STEP 1/4] Response structure: list (fallback)")
            
            # Parse JSON response into Message objects
            parse_start = time.time()
            messages = [Message(**msg) for msg in items]
            parse_time = time.time() - parse_start
            
            logger.info(f"[STEP 1/4] Successfully parsed {len(messages)} messages in {parse_time:.3f}s")
            
            # Log sample messages for debugging
            if messages:
                logger.debug(f"[STEP 1/4] Sample messages (first 3):")
                for i, msg in enumerate(messages[:3], 1):
                    msg_preview = msg.text[:50] + "..." if len(msg.text) > 50 else msg.text
                    logger.debug(f"[STEP 1/4]   {i}. Message(id='{msg.id}', user_id='{msg.user_id}', "
                               f"text='{msg_preview}')")
            
            total_time = time.time() - fetch_start
            logger.info(f"[STEP 1/4] Message ingestion complete in {total_time:.2f}s")
            
            return messages
            
    except httpx.HTTPError as e:
        fetch_time = time.time() - fetch_start
        logger.error(f"[STEP 1/4] HTTP error while fetching messages after {fetch_time:.2f}s: {e}", exc_info=True)
        raise
    except ValueError as e:
        fetch_time = time.time() - fetch_start
        logger.error(f"[STEP 1/4] Error parsing messages after {fetch_time:.2f}s: {e}", exc_info=True)
        raise
    except Exception as e:
        fetch_time = time.time() - fetch_start
        logger.error(f"[STEP 1/4] Unexpected error while fetching messages after {fetch_time:.2f}s: {e}", exc_info=True)
        raise

