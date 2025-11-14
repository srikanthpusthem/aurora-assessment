import numpy as np
import httpx
import logging
import time
from typing import List
from app.core.config import settings
from app.models.message import Message

logger = logging.getLogger(__name__)

# HuggingFace Inference API endpoint (updated 2024)
HF_API_BASE = "https://router.huggingface.co/hf-inference"


async def embed_text(text: str) -> np.ndarray:
    """
    Embed a single string into vector space using HuggingFace Inference API.
    
    Args:
        text: The text to embed
        
    Returns:
        numpy array of embeddings
    """
    embed_start = time.time()
    url = f"{HF_API_BASE}/models/{settings.hf_embedding_model}"
    
    text_preview = text[:50] + "..." if len(text) > 50 else text
    logger.info(f"[EMBEDDING] Embedding text: \"{text_preview}\"")
    logger.debug(f"[EMBEDDING] Full text: \"{text}\"")
    logger.debug(f"[EMBEDDING] API endpoint: {url}")
    logger.debug(f"[EMBEDDING] Model: {settings.hf_embedding_model}")
    
    headers = {
        "Authorization": f"Bearer {settings.hf_api_token[:10]}...",  # Partial token for logging
        "Content-Type": "application/json"
    }
    payload = {"inputs": text}
    
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            api_start = time.time()
            response = await client.post(url, json=payload, headers=headers)
            api_time = time.time() - api_start
            response.raise_for_status()
            
            logger.debug(f"[EMBEDDING] API call completed in {api_time:.2f}s (status: {response.status_code})")
            
            # HuggingFace returns embeddings as a list
            embedding = response.json()
            
            # Handle both single text (returns list) and batch (returns list of lists)
            if isinstance(embedding, list) and len(embedding) > 0:
                if isinstance(embedding[0], list):
                    # Batch response, take first item
                    embedding = embedding[0]
            
            embedding_array = np.array(embedding, dtype=np.float32)
            embed_time = time.time() - embed_start
            
            logger.info(f"[EMBEDDING] Generated embedding: shape={embedding_array.shape}, "
                       f"dtype={embedding_array.dtype} in {embed_time:.2f}s")
            logger.debug(f"[EMBEDDING] Sample embedding (first 5 values): {embedding_array[:5].tolist()}")
            logger.debug(f"[EMBEDDING] Embedding stats: min={embedding_array.min():.4f}, "
                        f"max={embedding_array.max():.4f}, mean={embedding_array.mean():.4f}")
            
            return embedding_array
            
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 503:
            # Model is loading, wait and retry
            logger.warning(f"[EMBEDDING] Model is loading (503), waiting 10 seconds before retry...")
            import asyncio
            await asyncio.sleep(10)
            # Retry once
            logger.info(f"[EMBEDDING] Retrying API call...")
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                embedding = response.json()
                if isinstance(embedding, list) and len(embedding) > 0:
                    if isinstance(embedding[0], list):
                        embedding = embedding[0]
                embedding_array = np.array(embedding, dtype=np.float32)
                embed_time = time.time() - embed_start
                logger.info(f"[EMBEDDING] Generated embedding (retry): shape={embedding_array.shape} in {embed_time:.2f}s")
                return embedding_array
        else:
            embed_time = time.time() - embed_start
            logger.error(f"[EMBEDDING] HTTP error embedding text after {embed_time:.2f}s: {e}", exc_info=True)
            raise
    except Exception as e:
        embed_time = time.time() - embed_start
        logger.error(f"[EMBEDDING] Error embedding text after {embed_time:.2f}s: {e}", exc_info=True)
        raise


async def embed_messages(messages: List[Message]) -> np.ndarray:
    """
    Embed a list of messages into vector space using HuggingFace Inference API.
    
    Args:
        messages: List of Message objects
        
    Returns:
        numpy array of shape (len(messages), embedding_dim) compatible with FAISS
    """
    embed_start = time.time()
    
    if not messages:
        logger.warning("[EMBEDDING] No messages to embed, returning empty array")
        return np.array([], dtype=np.float32)
    
    # Extract text content from messages
    texts = [msg.text for msg in messages]
    
    logger.info(f"[STEP 2/4] Embedding {len(messages)} messages using HuggingFace API...")
    logger.debug(f"[STEP 2/4] Model: {settings.hf_embedding_model}")
    logger.debug(f"[STEP 2/4] Sample texts (first 3):")
    for i, text in enumerate(texts[:3], 1):
        text_preview = text[:50] + "..." if len(text) > 50 else text
        logger.debug(f"[STEP 2/4]   {i}. \"{text_preview}\"")
    
    url = f"{HF_API_BASE}/models/{settings.hf_embedding_model}"
    headers = {
        "Authorization": f"Bearer {settings.hf_api_token[:10]}...",  # Partial token for logging
        "Content-Type": "application/json"
    }
    payload = {"inputs": texts}
    
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            api_start = time.time()
            response = await client.post(url, json=payload, headers=headers)
            api_time = time.time() - api_start
            response.raise_for_status()
            
            logger.debug(f"[STEP 2/4] API call completed in {api_time:.2f}s (status: {response.status_code})")
            
            # HuggingFace returns embeddings as a list of lists for batch
            embeddings_data = response.json()
            
            # Convert to numpy array
            embeddings = np.array(embeddings_data, dtype=np.float32)
            embed_time = time.time() - embed_start
            
            logger.info(f"[STEP 2/4] Generated embeddings: shape={embeddings.shape}, "
                       f"dtype={embeddings.dtype} in {embed_time:.2f}s")
            logger.debug(f"[STEP 2/4] Embedding dimension: {embeddings.shape[1]}")
            logger.debug(f"[STEP 2/4] Sample embedding (first message, first 5 values): "
                        f"{embeddings[0][:5].tolist()}")
            logger.debug(f"[STEP 2/4] Embedding stats: min={embeddings.min():.4f}, "
                        f"max={embeddings.max():.4f}, mean={embeddings.mean():.4f}")
            
            return embeddings
            
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 503:
            # Model is loading, wait and retry
            logger.warning(f"[STEP 2/4] Model is loading (503), waiting 10 seconds before retry...")
            import asyncio
            await asyncio.sleep(10)
            # Retry once
            logger.info(f"[STEP 2/4] Retrying API call...")
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                embeddings_data = response.json()
                embeddings = np.array(embeddings_data, dtype=np.float32)
                embed_time = time.time() - embed_start
                logger.info(f"[STEP 2/4] Generated embeddings (retry): shape={embeddings.shape} in {embed_time:.2f}s")
                return embeddings
        else:
            embed_time = time.time() - embed_start
            logger.error(f"[STEP 2/4] HTTP error embedding messages after {embed_time:.2f}s: {e}", exc_info=True)
            raise
    except Exception as e:
        embed_time = time.time() - embed_start
        logger.error(f"[STEP 2/4] Error embedding messages after {embed_time:.2f}s: {e}", exc_info=True)
        raise
