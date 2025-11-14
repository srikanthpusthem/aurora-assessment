import httpx
import logging
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
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            logger.info(f"Fetching messages from {settings.messages_api_url}")
            response = await client.get(settings.messages_api_url)
            response.raise_for_status()
            
            data = response.json()
            logger.info(f"Received {len(data)} messages from API")
            
            # Parse JSON response into Message objects
            messages = [Message(**msg) for msg in data]
            
            logger.info(f"Successfully parsed {len(messages)} messages")
            return messages
            
    except httpx.HTTPError as e:
        logger.error(f"HTTP error while fetching messages: {e}")
        raise
    except ValueError as e:
        logger.error(f"Error parsing messages: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error while fetching messages: {e}")
        raise

