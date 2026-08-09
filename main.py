import logging
import os
import uuid
import base64
import tempfile
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI
import inngest
import inngest.fast_api
from inngest.experimental import ai
from data_loader import load_and_chunk_pdf, embed_texts
from vector_db import Qdrant_storage
from custom_types import RAGChunkandSrc, RAGSearchResult, RAGUpsertResult, RAGQueryResult

load_dotenv(dotenv_path=Path(__file__).parent / ".env")

_signing_key = os.getenv("INNGEST_SIGNING_KEY")
_event_key = os.getenv("INNGEST_EVENT_KEY")

inngest_client = inngest.Inngest(
    app_id="rag_app",
    logger=logging.getLogger("uvicorn"),
    is_production=bool(_signing_key),
    signing_key=_signing_key,
    event_key=_event_key,
    serializer=inngest.PydanticSerializer(),
)

@inngest_client.create_function(
    fn_id="RAG: Ingest PDF",
    trigger=inngest.TriggerEvent(event="rag/ingest_pdf"),
)
async def rag_ingest_pdf(ctx: inngest.Context):

    def _load(ctx: inngest.Context) -> RAGChunkandSrc:
        pdf_b64 = ctx.event.data["pdf_base64"]
        source_id = ctx.event.data.get("source_id", "upload.pdf")
        pdf_bytes = base64.b64decode(pdf_b64)
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(pdf_bytes)
            tmp_path = tmp.name
        chunks = load_and_chunk_pdf(tmp_path)
        return RAGChunkandSrc(chunks=chunks, source_id=source_id)

    def _upsert(chunks_and_src: RAGChunkandSrc) -> RAGUpsertResult:
        chunks = chunks_and_src.chunks
        source_id = chunks_and_src.source_id
        vecs = embed_texts(chunks)
        ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}:{i}")) for i in range(len(chunks))]
        payloads = [{"source": source_id, "text": chunks[i]} for i in range(len(chunks))]
        Qdrant_storage().upsert(ids, vecs, payloads)
        return RAGUpsertResult(ingested=len(chunks))

    chunks_and_src = await ctx.step.run("load_and_chunk", lambda: _load(ctx), output_type=RAGChunkandSrc)
    ingested = await ctx.step.run("embed_and_upsert", lambda: _upsert(chunks_and_src), output_type=RAGUpsertResult)
    return ingested.model_dump()

@inngest_client.create_function(
    fn_id="RAG: Query PDF",
    trigger=inngest.TriggerEvent(event="rag/query_pdf"),
)
async def rag_query_pdf_ai(ctx: inngest.Context):

    def _search(question: str, top_k: int = 5) -> RAGSearchResult:
        query_vec = embed_texts([question])[0]
        store = Qdrant_storage()
        found = store.search(query_vec, top_k=top_k)
        return RAGSearchResult(context=found["context"], sources=found["sources"])

    question = ctx.event.data["question"]
    top_k = ctx.event.data.get("top_k", 5)

    found = await ctx.step.run("search", lambda: _search(question, top_k), output_type=RAGSearchResult)

    context_block = "\n\n".join(f" -{c}" for c in found.context)
    user_content = (
        "Use the following context to answer the question.\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question:\n{question}\n\n"
        "Answer concisely using the context above."
    )

    adapter = ai.openai.Adapter(
        auth_key=os.getenv("GENAI_API_KEY"),
        model="gemini-3.5-flash",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )

    res = await ctx.step.ai.infer(
        "llm-answer",
        adapter=adapter,
        body={
            "model": "gemini-3.5-flash",
            "max_tokens": 512,
            "temperature": 0.3,
            "messages": [
                {"role": "system", "content": "You answer questions using only the provided context."},
                {"role": "user", "content": user_content}
            ]
        }
    )

    answer = res["choices"][0]["message"]["content"].strip()
    return {"answer": answer, "sources": found.sources, "num_contexts": len(found.context)}

app = FastAPI()

inngest.fast_api.serve(app, inngest_client, functions=[rag_ingest_pdf, rag_query_pdf_ai])