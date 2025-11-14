import faiss
import numpy as np
import logging
import time
from typing import List, Tuple, Optional
from app.models.message import Message
from app.services.ingestion import fetch_messages
from app.core.embeddings import embed_messages

logger = logging.getLogger(__name__)


class VectorIndex:
    """FAISS index builder and manager for message embeddings."""
    
    def __init__(self):
        self.messages: List[Message] = []
        self.embeddings: Optional[np.ndarray] = None
        self.index: Optional[faiss.Index] = None
        self._is_built = False
    
    async def build_index(self) -> None:
        """
        Build FAISS index from all message embeddings on startup.
        
        Steps:
        1. Fetch messages via ingestion service
        2. Embed all messages
        3. Create FAISS index (L2 distance)
        4. Store index in memory
        """
        build_start = time.time()
        try:
            logger.info("[STEP 3/4] Starting to build FAISS index...")
            
            # Fetch messages (already logged in ingestion service)
            messages = await fetch_messages()
            if not messages:
                logger.warning("[STEP 3/4] No messages fetched from API, skipping index build")
                return
            
            self.messages = messages
            logger.debug(f"[STEP 3/4] Stored {len(messages)} messages in index")
            
            # Embed all messages (already logged in embeddings service)
            embeddings = await embed_messages(messages)
            self.embeddings = embeddings
            
            # Get embedding dimension
            dimension = embeddings.shape[1]
            logger.info(f"[STEP 3/4] Embedding dimension: {dimension}")
            logger.debug(f"[STEP 3/4] Embeddings shape: {embeddings.shape}, dtype: {embeddings.dtype}")
            
            # Create FAISS index (L2 distance)
            # Using IndexFlatL2 for exact search
            index_start = time.time()
            logger.info(f"[STEP 3/4] Creating FAISS IndexFlatL2 (dimension={dimension})...")
            self.index = faiss.IndexFlatL2(dimension)
            index_create_time = time.time() - index_start
            logger.debug(f"[STEP 3/4] Index created in {index_create_time:.3f}s")
            
            # Add embeddings to index
            add_start = time.time()
            logger.info(f"[STEP 3/4] Adding {len(embeddings)} embeddings to index...")
            self.index.add(embeddings)
            add_time = time.time() - add_start
            logger.debug(f"[STEP 3/4] Embeddings added in {add_time:.3f}s")
            
            self._is_built = True
            build_time = time.time() - build_start
            
            logger.info(f"[STEP 3/4] Successfully built FAISS index: {self.index.ntotal} vectors, "
                       f"dimension={dimension}, type=IndexFlatL2 in {build_time:.2f}s")
            logger.debug(f"[STEP 3/4] Index statistics: ntotal={self.index.ntotal}, "
                        f"is_trained={self.index.is_trained}")
            
        except Exception as e:
            build_time = time.time() - build_start
            logger.error(f"[STEP 3/4] Error building FAISS index after {build_time:.2f}s: {e}", exc_info=True)
            raise
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Tuple[Message, float]]:
        """
        Search the FAISS index for top-K most similar messages.
        
        Args:
            query_embedding: Query vector of shape (embedding_dim,)
            k: Number of results to return
            
        Returns:
            List of tuples (Message, distance_score) sorted by relevance
        """
        search_start = time.time()
        
        if not self._is_built or self.index is None:
            raise RuntimeError("Index not built. Call build_index() first.")
        
        if len(self.messages) == 0:
            logger.warning("[SEARCH] No messages in index, returning empty results")
            return []
        
        logger.debug(f"[SEARCH] Query embedding shape: {query_embedding.shape}, dtype: {query_embedding.dtype}")
        logger.debug(f"[SEARCH] Requested k: {k}, available messages: {len(self.messages)}")
        
        # Reshape query embedding to (1, embedding_dim) for FAISS
        query_embedding = query_embedding.reshape(1, -1)
        
        # Search
        k = min(k, len(self.messages))
        logger.debug(f"[SEARCH] Searching index with k={k}...")
        
        distances, indices = self.index.search(query_embedding, k)
        search_time = time.time() - search_start
        
        logger.debug(f"[SEARCH] Search completed in {search_time:.4f}s")
        logger.debug(f"[SEARCH] Found {len(distances[0])} results")
        logger.debug(f"[SEARCH] Distances: {distances[0].tolist()}")
        logger.debug(f"[SEARCH] Indices: {indices[0].tolist()}")
        
        # Return messages with their distances
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.messages):
                results.append((self.messages[idx], float(distance)))
        
        if results:
            logger.debug(f"[SEARCH] Top result: Message(id='{results[0][0].id}', "
                       f"distance={results[0][1]:.4f}, preview='{results[0][0].text[:50]}...')")
        
        return results
    
    def is_ready(self) -> bool:
        """Check if the index is built and ready for search."""
        return self._is_built and self.index is not None


# Global index instance
_index: Optional[VectorIndex] = None


def get_index() -> VectorIndex:
    """Get or create the global vector index instance."""
    global _index
    if _index is None:
        _index = VectorIndex()
    return _index

