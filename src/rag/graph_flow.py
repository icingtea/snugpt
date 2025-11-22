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
    builder = StateGraph(state_schema=GraphState)
    
    builder.add_node("keyword_router", keyword_router)
    builder.add_node("vocab_voter", vocab_voter)
    builder.add_node("vector_search", vector_search)
    builder.add_node("chat_response", chat_response)
    builder.add_node("error_response", error_response)
    
    builder.add_edge(START, "keyword_router")

    builder.add_conditional_edges(
        "keyword_router",
        needs_vocab_vote,
        {
            True: "vocab_voter",
            False: "vector_search",
        },
    )

    builder.add_edge("vocab_voter", "vector_search")

    builder.add_conditional_edges(
        "vector_search",
        error_check,
        {
            True: "error_response",
            False: "chat_response",
        },
    )

    builder.add_conditional_edges(
        "chat_response",
        error_check,
        {
            True: "error_response",
            False: END,            
        },
    )

    builder.add_edge("error_response", END)

    app = builder.compile(checkpointer=memory)
    return app