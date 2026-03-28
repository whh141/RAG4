"""FastAPI application entrypoint."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.routes_chat import router as chat_router
from .api.routes_kb import router as kb_router
from .db.sqlite_mgr import SQLiteManager

app = FastAPI(title="Agentic RAG System", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(kb_router)
app.include_router(chat_router)


@app.on_event("startup")
def on_startup() -> None:
    SQLiteManager().init_db()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
