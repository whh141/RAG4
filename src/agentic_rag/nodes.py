"""Node skeletons for the Agentic RAG LangGraph workflow."""

from typing import Any

from .state import GraphState


def intent_router_node(state: GraphState) -> GraphState:
    """Classify user intent into LOCAL_FACT, WEB_SEARCH, or COMPLEX_REASONING."""

    raise NotImplementedError("Step 1 skeleton: implementation is added in step 3.")


def local_retriever_node(state: GraphState) -> GraphState:
    """Retrieve Top-K local knowledge documents from Chroma."""

    raise NotImplementedError("Step 1 skeleton: implementation is added in step 3.")


def web_searcher_node(state: GraphState) -> GraphState:
    """Search Tavily for time-sensitive web evidence and normalize as documents."""

    raise NotImplementedError("Step 1 skeleton: implementation is added in step 3.")


def complex_reasoner_node(state: GraphState) -> GraphState:
    """Decompose complex queries and combine local+web retrieval outputs."""

    raise NotImplementedError("Step 1 skeleton: implementation is added in step 3.")


def reranker_compressor_node(state: GraphState) -> GraphState:
    """Rerank retrieved documents and compress them into filtered factual context."""

    raise NotImplementedError("Step 1 skeleton: implementation is added in step 3.")


def cot_generator_node(state: GraphState) -> GraphState:
    """Generate final response using <think> and <answer> blocks with citations."""

    raise NotImplementedError("Step 1 skeleton: implementation is added in step 3.")


def route_by_intent(state: GraphState) -> str:
    """Map `intent_type` to next node key in the graph."""

    intent = state["intent_type"]
    if intent == "LOCAL_FACT":
        return "local_retriever"
    if intent == "WEB_SEARCH":
        return "web_searcher"
    if intent == "COMPLEX_REASONING":
        return "complex_reasoner"
    raise ValueError(f"Unsupported intent_type: {intent}")


def build_graph() -> Any:
    """Build and return the LangGraph workflow object.

    Step 1 only defines the graph topology and conditional routing skeleton.
    """

    raise NotImplementedError("Step 1 skeleton: implementation is added in step 3.")
