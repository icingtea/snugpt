import streamlit as st
from rag.graph_flow import assemble_graph
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages.ai import AIMessage
from langchain_core.messages.human import HumanMessage


def run_app():
    st.set_page_config(page_title="SNUGPT", page_icon="🎓", layout="wide")

    memory: MemorySaver = MemorySaver()

    if "graph_state" not in st.session_state:
        from src.models.graph import GraphState
        st.session_state.graph_state = GraphState(
            prompt=None,
            memory=[],
            collections=[],
            filter=None,
            context=[],
            response=None,
            error=None,
        )

    if "graph_config" not in st.session_state:
        st.session_state.graph_config = {"configurable": {"thread_id": "demo"}}

    graph = assemble_graph(memory=memory)

    st.title("🎓 SNUGPT")
    st.caption("Your AI assistant for Shiv Nadar University")

    for message in st.session_state.graph_state["memory"]:
        if isinstance(message, HumanMessage):
            role = "user"
            content = message.content
        elif isinstance(message, AIMessage):
            role = "assistant"
            content = message.content
        else:
            role = "assistant"
            content = getattr(message, "content", str(message))
        
        with st.chat_message(role):
            st.markdown(content)

    if prompt := st.chat_input("Ask about academics, faculty, students, or the menu..."):
        with st.chat_message("user"):
            st.markdown(prompt)

        st.session_state.graph_state["prompt"] = prompt

        new_state = graph.invoke(
            st.session_state.graph_state, st.session_state.graph_config
        )
        st.session_state.graph_state = new_state

        with st.chat_message("assistant"):
            reply = new_state.get("response", "[ERROR] Could not get response.")
            st.markdown(reply)


if __name__ == "__main__":
    run_app()