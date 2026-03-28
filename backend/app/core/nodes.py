"""LangGraph node implementations for query rewriting, routing, retrieval, and answer generation."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from tavily import TavilyClient

from app.core.state import AgentState, IntentType, RetrievedDocument
from app.db.chroma_mgr import ChromaManager


@dataclass(frozen=True)
class NodeRuntime:
    """Runtime dependencies for graph nodes."""

    llm: ChatOpenAI
    chroma_mgr: ChromaManager
    tavily_client: TavilyClient


def _required_env(name: str) -> str:
    value = os.getenv(name)
    if value is None or not value.strip():
        raise ValueError(f"Required environment variable is missing: {name}")
    return value


def build_runtime() -> NodeRuntime:
    """Construct strict runtime dependencies from environment variables."""
    model_name = _required_env("LLM_MODEL")
    api_key = _required_env("LLM_API_KEY")
    base_url = _required_env("LLM_BASE_URL")
    tavily_api_key = _required_env("TAVILY_API_KEY")

    llm = ChatOpenAI(model=model_name, api_key=api_key, base_url=base_url, temperature=0)
    chroma_mgr = ChromaManager()
    tavily_client = TavilyClient(api_key=tavily_api_key)

    return NodeRuntime(llm=llm, chroma_mgr=chroma_mgr, tavily_client=tavily_client)


@lru_cache(maxsize=1)
def get_runtime() -> NodeRuntime:
    """Return singleton runtime dependencies for node execution."""
    return build_runtime()


def _format_chat_history(chat_history: list[dict[str, str]]) -> str:
    if not chat_history:
        return ""
    return "\n".join(f"{msg['role']}: {msg['content']}" for msg in chat_history)


def _normalize_local_results(raw: dict[str, Any]) -> list[RetrievedDocument]:
    docs = raw.get("documents", [[]])
    metadatas = raw.get("metadatas", [[]])
    distances = raw.get("distances", [[]])

    first_docs = docs[0] if docs else []
    first_metadatas = metadatas[0] if metadatas else []
    first_distances = distances[0] if distances else []

    normalized: list[RetrievedDocument] = []
    for idx, content in enumerate(first_docs):
        metadata = first_metadatas[idx] if idx < len(first_metadatas) and first_metadatas[idx] else {}
        score = 1 - first_distances[idx] if idx < len(first_distances) else 0.0
        normalized.append(
            RetrievedDocument(
                source_type="local",
                source=str(metadata.get("document_id", "unknown_document")),
                title=str(metadata.get("title", metadata.get("file_name", "Local Document"))),
                content=str(content),
                score=float(score),
            )
        )
    return normalized


def _normalize_web_results(raw: list[dict[str, Any]]) -> list[RetrievedDocument]:
    normalized: list[RetrievedDocument] = []
    for item in raw:
        normalized.append(
            RetrievedDocument(
                source_type="web",
                source=str(item["url"]),
                title=str(item.get("title", "Web Result")),
                content=str(item["content"]),
                score=float(item.get("score", 0.0)),
                url=str(item["url"]),
            )
        )
    return normalized


def query_rewrite_node(state: AgentState) -> AgentState:
    """Rewrite contextual user query into a standalone query."""
    system_prompt = (
        "You are a query rewriting engine. Rewrite the latest user query into a standalone query "
        "using chat history context. Return only the rewritten query text."
    )
    history_text = _format_chat_history(state["chat_history"])

    runtime = get_runtime()
    response = runtime.llm.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(
                content=(
                    f"Session ID: {state['session_id']}\n"
                    f"Chat History:\n{history_text or '(empty)'}\n\n"
                    f"User Query:\n{state['user_query']}"
                )
            ),
        ]
    )
    if not isinstance(response, AIMessage) or not response.content:
        raise ValueError("query_rewrite_node produced an empty LLM response")

    state["standalone_query"] = str(response.content).strip()
    if not state["standalone_query"]:
        raise ValueError("query_rewrite_node produced blank standalone_query")
    return state


def intent_router_node(state: AgentState) -> AgentState:
    """Classify query intent into LOCAL, WEB, or COMPLEX."""
    system_prompt = (
        "Classify the query intent into exactly one label: LOCAL, WEB, or COMPLEX. "
        "Return strict JSON: {\"intent\":\"LOCAL|WEB|COMPLEX\"}.\n"
        "LOCAL: answerable mainly from private/local KB.\n"
        "WEB: needs fresh public web information.\n"
        "COMPLEX: requires decomposition, both local and web evidence, or multi-step reasoning."
    )

    runtime = get_runtime()
    response = runtime.llm.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=state["standalone_query"]),
        ]
    )
    if not isinstance(response, AIMessage) or not response.content:
        raise ValueError("intent_router_node produced an empty LLM response")

    parsed = json.loads(str(response.content))
    intent_raw = str(parsed["intent"]).upper().strip()
    try:
        state["intent_type"] = IntentType(intent_raw)
    except ValueError as exc:
        raise ValueError(f"intent_router_node returned unsupported intent: {intent_raw}") from exc

    return state


def local_retrieve_node(state: AgentState) -> AgentState:
    """Retrieve semantically similar local chunks from ChromaDB."""
    runtime = get_runtime()
    raw = runtime.chroma_mgr.query(query_text=state["standalone_query"], n_results=6)
    state["retrieved_docs"] = _normalize_local_results(raw)
    if not state["retrieved_docs"]:
        raise ValueError("local_retrieve_node returned no documents")
    return state


def web_search_node(state: AgentState) -> AgentState:
    """Retrieve web evidence via Tavily search."""
    runtime = get_runtime()
    raw = runtime.tavily_client.search(
        query=state["standalone_query"],
        topic="general",
        search_depth="advanced",
        max_results=6,
        include_answer=False,
        include_images=False,
    )
    results = raw.get("results")
    if not isinstance(results, list) or len(results) == 0:
        raise ValueError("web_search_node returned no results")

    state["retrieved_docs"] = _normalize_web_results(results)
    return state


def complex_reason_node(state: AgentState) -> AgentState:
    """Decompose query and perform hybrid retrieval (local + web)."""
    decompose_prompt = (
        "Decompose the query into 2-4 focused sub-questions for hybrid retrieval. "
        "Return strict JSON: {\"sub_queries\":[\"...\"]}."
    )

    runtime = get_runtime()
    response = runtime.llm.invoke(
        [
            SystemMessage(content=decompose_prompt),
            HumanMessage(content=state["standalone_query"]),
        ]
    )
    if not isinstance(response, AIMessage) or not response.content:
        raise ValueError("complex_reason_node decomposition returned empty response")

    parsed = json.loads(str(response.content))
    sub_queries = parsed["sub_queries"]
    if not isinstance(sub_queries, list) or not sub_queries:
        raise ValueError("complex_reason_node produced invalid sub_queries")

    aggregated_docs: list[RetrievedDocument] = []

    for query in sub_queries:
        sub_query = str(query).strip()
        if not sub_query:
            raise ValueError("complex_reason_node encountered blank sub_query")

        local_raw = runtime.chroma_mgr.query(query_text=sub_query, n_results=3)
        aggregated_docs.extend(_normalize_local_results(local_raw))

        web_raw = runtime.tavily_client.search(
            query=sub_query,
            topic="general",
            search_depth="advanced",
            max_results=3,
            include_answer=False,
            include_images=False,
        )
        web_results = web_raw.get("results")
        if not isinstance(web_results, list):
            raise ValueError("complex_reason_node web results are malformed")
        aggregated_docs.extend(_normalize_web_results(web_results))

    if not aggregated_docs:
        raise ValueError("complex_reason_node returned no retrieved documents")

    aggregated_docs.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    state["retrieved_docs"] = aggregated_docs[:12]
    return state


def rerank_filter_node(state: AgentState) -> AgentState:
    """Extract and compress core facts from retrieved documents."""
    if not state["retrieved_docs"]:
        raise ValueError("rerank_filter_node requires non-empty retrieved_docs")

    evidence_blocks: list[str] = []
    for idx, doc in enumerate(state["retrieved_docs"], start=1):
        evidence_blocks.append(
            f"[DOC-{idx}] source={doc.get('source')} title={doc.get('title')}\n{doc.get('content')}"
        )

    system_prompt = (
        "Extract only factual statements relevant to answering the query. "
        "Do not invent facts. Keep explicit citations by referencing DOC ids in brackets, e.g., [DOC-2]."
    )

    runtime = get_runtime()
    response = runtime.llm.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(
                content=(
                    f"Query:\n{state['standalone_query']}\n\n"
                    f"Evidence:\n{"\n\n".join(evidence_blocks)}"
                )
            ),
        ]
    )

    if not isinstance(response, AIMessage) or not response.content:
        raise ValueError("rerank_filter_node produced empty filtered facts")

    state["filtered_facts"] = str(response.content).strip()
    if not state["filtered_facts"]:
        raise ValueError("rerank_filter_node produced blank filtered_facts")

    return state


def cot_generate_node(state: AgentState) -> AgentState:
    """Generate final answer with strict <think> and <answer> tags including citations."""
    if not state["filtered_facts"].strip():
        raise ValueError("cot_generate_node requires non-empty filtered_facts")

    system_prompt = (
        "You are a rigorous QA engine. Output MUST follow this exact XML-like schema:\n"
        "<think>...</think>\n<answer>...</answer>\n"
        "Rules:\n"
        "1) Use only provided filtered facts.\n"
        "2) Include source citations in answer body (e.g., [DOC-1], [DOC-3]).\n"
        "3) Do not output text outside tags."
    )

    runtime = get_runtime()
    response = runtime.llm.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(
                content=(
                    f"Question:\n{state['standalone_query']}\n\n"
                    f"Filtered Facts:\n{state['filtered_facts']}"
                )
            ),
        ]
    )

    if not isinstance(response, AIMessage) or not response.content:
        raise ValueError("cot_generate_node produced empty final answer")

    final_text = str(response.content).strip()
    if "<think>" not in final_text or "</think>" not in final_text:
        raise ValueError("cot_generate_node response missing <think> tag block")
    if "<answer>" not in final_text or "</answer>" not in final_text:
        raise ValueError("cot_generate_node response missing <answer> tag block")

    state["final_answer"] = final_text
    return state


__all__ = [
    "query_rewrite_node",
    "intent_router_node",
    "local_retrieve_node",
    "web_search_node",
    "complex_reason_node",
    "rerank_filter_node",
    "cot_generate_node",
]
