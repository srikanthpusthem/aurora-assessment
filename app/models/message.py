from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class Message(BaseModel):
    """Message model matching Aurora API response structure."""
    
    id: str
    user_id: str
    message: str  # API uses "message" not "text"
    user_name: Optional[str] = None
    timestamp: Optional[datetime] = None
    
    @property
    def text(self) -> str:
        """Alias for message field for backward compatibility."""
        return self.message

