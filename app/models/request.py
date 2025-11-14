from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    """Request model for the /api/ask endpoint."""
    
    question: str = Field(..., description="The question to answer about users", min_length=1)

