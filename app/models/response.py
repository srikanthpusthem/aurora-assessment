from pydantic import BaseModel


class AskResponse(BaseModel):
    """Response model for the /api/ask endpoint."""
    
    answer: str

