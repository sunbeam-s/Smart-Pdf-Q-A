import pydantic
from typing import Optional

class RAGChunkandSrc(pydantic.BaseModel):
    chunks: list[str]
    source_id: Optional[str] = None

class RAGUpsertResult(pydantic.BaseModel):
    ingested: int

class RAGSearchResult(pydantic.BaseModel):
    context: list[str]
    sources: list[str]

class RAGQueryResult(pydantic.BaseModel):
    answer: str
    sources: list[str]
    num_contexts: int