from pydantic import BaseModel, Field


class RetrievalCandidate(BaseModel):
    """
    Common representation for a candidate returned by
    any retrieval strategy.
    """

    index_id: int = Field(..., ge=0)

    score: float

    retriever: str = Field(..., min_length=1)