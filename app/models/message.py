from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class Message(BaseModel):
    """Message model matching Aurora API response structure."""
    
    id: str
    user_id: str
    text: str
    timestamp: Optional[datetime] = None

