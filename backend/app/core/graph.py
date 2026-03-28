"""Graph assembly for Agentic RAG workflow."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from app.core.nodes import (
    complex_reason_node,
    cot_generate_node,
    intent_router_node,
    local_retrieve_node,
    query_rewrite_node,
    rerank_filter_node,
    web_search_node,
)
from app.core.state import AgentState, IntentType


def route_by_intent(state: AgentState) -> str:
    """Return downstream node key based on classified intent."""
    intent = state["intent_type"]
    if intent == IntentType.LOCAL:
        return "local_retrieve_node"
    if intent == IntentType.WEB:
        return "web_search_node"
    if intent == IntentType.COMPLEX:
        return "complex_reason_node"
    raise ValueError(f"Unsupported intent_type for routing: {intent}")


def build_graph():
    """Compile and return the Agentic RAG StateGraph."""
    graph_builder = StateGraph(AgentState)

    graph_builder.add_node("query_rewrite_node", query_rewrite_node)
    graph_builder.add_node("intent_router_node", intent_router_node)
    graph_builder.add_node("local_retrieve_node", local_retrieve_node)
    graph_builder.add_node("web_search_node", web_search_node)
    graph_builder.add_node("complex_reason_node", complex_reason_node)
    graph_builder.add_node("rerank_filter_node", rerank_filter_node)
    graph_builder.add_node("cot_generate_node", cot_generate_node)

    graph_builder.add_edge(START, "query_rewrite_node")
    graph_builder.add_edge("query_rewrite_node", "intent_router_node")

    graph_builder.add_conditional_edges(
        "intent_router_node",
        route_by_intent,
        {
            "local_retrieve_node": "local_retrieve_node",
            "web_search_node": "web_search_node",
            "complex_reason_node": "complex_reason_node",
        },
    )

    graph_builder.add_edge("local_retrieve_node", "rerank_filter_node")
    graph_builder.add_edge("web_search_node", "rerank_filter_node")
    graph_builder.add_edge("complex_reason_node", "rerank_filter_node")
    graph_builder.add_edge("rerank_filter_node", "cot_generate_node")
    graph_builder.add_edge("cot_generate_node", END)

    return graph_builder.compile()


GRAPH = build_graph()


__all__ = ["GRAPH", "build_graph", "route_by_intent"]
