import numpy as np
import httpx
import logging
from typing import List
from app.core.config import settings
from app.models.message import Message

logger = logging.getLogger(__name__)

# HuggingFace Inference API endpoint
HF_API_BASE = "https://api-inference.huggingface.co"


async def embed_text(text: str) -> np.ndarray:
    """
    Embed a single string into vector space using HuggingFace Inference API.
    
    Args:
        text: The text to embed
        
    Returns:
        numpy array of embeddings
    """
    url = f"{HF_API_BASE}/pipeline/feature-extraction/{settings.hf_embedding_model}"
    headers = {
        "Authorization": f"Bearer {settings.hf_api_token}",
        "Content-Type": "application/json"
    }
    payload = {"inputs": text}
    
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            
            # HuggingFace returns embeddings as a list
            embedding = response.json()
            
            # Handle both single text (returns list) and batch (returns list of lists)
            if isinstance(embedding, list) and len(embedding) > 0:
                if isinstance(embedding[0], list):
                    # Batch response, take first item
                    embedding = embedding[0]
            
            return np.array(embedding, dtype=np.float32)
            
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 503:
            # Model is loading, wait and retry
            logger.warning("Model is loading, waiting 10 seconds before retry...")
            import asyncio
            await asyncio.sleep(10)
            # Retry once
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                embedding = response.json()
                if isinstance(embedding, list) and len(embedding) > 0:
                    if isinstance(embedding[0], list):
                        embedding = embedding[0]
                return np.array(embedding, dtype=np.float32)
        else:
            logger.error(f"HTTP error embedding text: {e}")
            raise
    except Exception as e:
        logger.error(f"Error embedding text: {e}")
        raise


async def embed_messages(messages: List[Message]) -> np.ndarray:
    """
    Embed a list of messages into vector space using HuggingFace Inference API.
    
    Args:
        messages: List of Message objects
        
    Returns:
        numpy array of shape (len(messages), embedding_dim) compatible with FAISS
    """
    if not messages:
        return np.array([], dtype=np.float32)
    
    # Extract text content from messages
    texts = [msg.text for msg in messages]
    
    url = f"{HF_API_BASE}/pipeline/feature-extraction/{settings.hf_embedding_model}"
    headers = {
        "Authorization": f"Bearer {settings.hf_api_token}",
        "Content-Type": "application/json"
    }
    payload = {"inputs": texts}
    
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            
            # HuggingFace returns embeddings as a list of lists for batch
            embeddings_data = response.json()
            
            # Convert to numpy array
            embeddings = np.array(embeddings_data, dtype=np.float32)
            
            logger.info(f"Embedded {len(messages)} messages into {embeddings.shape} array")
            return embeddings
            
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 503:
            # Model is loading, wait and retry
            logger.warning("Model is loading, waiting 10 seconds before retry...")
            import asyncio
            await asyncio.sleep(10)
            # Retry once
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                embeddings_data = response.json()
                embeddings = np.array(embeddings_data, dtype=np.float32)
                logger.info(f"Embedded {len(messages)} messages into {embeddings.shape} array")
                return embeddings
        else:
            logger.error(f"HTTP error embedding messages: {e}")
            raise
    except Exception as e:
        logger.error(f"Error embedding messages: {e}")
        raise
