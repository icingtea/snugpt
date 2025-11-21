from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from src.models.graph import GraphState
from src.rag.graph_nodes import (
    keyword_router,
    vocab_voter,
    vector_search,
    chat_response,
    error_response,
    error_check,
    needs_vocab_vote,
)


def assemble_graph(memory: MemorySaver):
    """Assemble the LangGraph for SNUGPT."""
    builder = StateGraph(state_schema=GraphState)
    
    # Add nodes
    builder.add_node("keyword_router", keyword_router)
    builder.add_node("vocab_voter", vocab_voter)
    builder.add_node("vector_search", vector_search)
    builder.add_node("chat_response", chat_response)
    builder.add_node("error_response", error_response)
    
    # Start with keyword routing
    builder.add_edge(START, "keyword_router")
    
    # After keyword routing, either go to vocab voter or vector search
    builder.add_conditional_edges(
        "keyword_router",
        needs_vocab_vote,
        {
            True: "vocab_voter",      # No keyword match -> vocab vote
            False: "vector_search",   # Keyword match found -> proceed to search
        },
    )
    
    # After vocab voting, proceed to vector search
    builder.add_edge("vocab_voter", "vector_search")
    
    # After vector search, check for errors
    builder.add_conditional_edges(
        "vector_search",
        error_check,
        {
            True: "error_response",   # Error occurred
            False: "chat_response",   # Success -> generate response
        },
    )
    
    # After chat response, check for errors or end
    builder.add_conditional_edges(
        "chat_response",
        error_check,
        {
            True: "error_response",   # Error occurred
            False: END,               # Success -> end
        },
    )
    
    # Error response always ends
    builder.add_edge("error_response", END)
    
    # Compile the graph with memory
    app = builder.compile(checkpointer=memory)
    return app