import faiss
import numpy as np
import logging
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
        try:
            logger.info("Starting to build FAISS index...")
            
            # Fetch messages
            messages = await fetch_messages()
            if not messages:
                logger.warning("No messages fetched from API")
                return
            
            self.messages = messages
            logger.info(f"Fetched {len(messages)} messages")
            
            # Embed all messages
            embeddings = await embed_messages(messages)
            self.embeddings = embeddings
            
            # Get embedding dimension
            dimension = embeddings.shape[1]
            logger.info(f"Embedding dimension: {dimension}")
            
            # Create FAISS index (L2 distance)
            # Using IndexFlatL2 for exact search
            self.index = faiss.IndexFlatL2(dimension)
            
            # Add embeddings to index
            self.index.add(embeddings)
            
            self._is_built = True
            logger.info(f"Successfully built FAISS index with {self.index.ntotal} vectors")
            
        except Exception as e:
            logger.error(f"Error building FAISS index: {e}")
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
        if not self._is_built or self.index is None:
            raise RuntimeError("Index not built. Call build_index() first.")
        
        if len(self.messages) == 0:
            return []
        
        # Reshape query embedding to (1, embedding_dim) for FAISS
        query_embedding = query_embedding.reshape(1, -1)
        
        # Search
        k = min(k, len(self.messages))
        distances, indices = self.index.search(query_embedding, k)
        
        # Return messages with their distances
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.messages):
                results.append((self.messages[idx], float(distance)))
        
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

