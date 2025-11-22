import streamlit as st
from rag.graph_flow import assemble_graph
from models.graph import GraphState
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages.ai import AIMessage
from langchain_core.messages.human import HumanMessage


def _set_custom_style():
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Montserrat:ital,wght@0,100..900;1,100..900&family=Ubuntu:ital,wght@0,300;0,400;0,500;0,700;1,300;1,400;1,500;1,700&display=swap');
            
            * {
                font-family: 'Ubuntu', sans-serif !important;
            }

            h1 {
                font-family: 'Montserrat', sans-serif !important;
            }

            .block-container {
                padding-top: 2rem;
                max-width: 900px;
                margin: auto;
            }
            
            .header-container {
                text-align: center;
                padding-bottom: 1.2rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def run_app():
    st.set_page_config(page_title="SNUGPT", page_icon="📑", layout="wide")
    _set_custom_style()

    memory = MemorySaver()

    if "graph_state" not in st.session_state:
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

    with st.container():
        st.markdown(
            """
            <div class="header-container">
                <h1 style="font-size:2.8rem; font-weight:700;">📑 SNUGPT</h1>
                <p style="opacity:0.8; font-size:1.1rem;">Your AI assistant for Shiv Nadar University</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    for message in st.session_state.graph_state.memory:
        if isinstance(message, HumanMessage):
            role = "🐇"
            content = message.content
        elif isinstance(message, AIMessage):
            role = "🤖"
            content = message.content
        else:
            role = "🤖"
            content = getattr(message, "content", str(message))

        with st.chat_message(role):
            st.markdown(content)

    prompt = st.chat_input("Ask about academics, faculty, students, or the menu...")
    if prompt:
        with st.chat_message("🐇"):
            st.markdown(prompt)

        st.session_state.graph_state.prompt = prompt

        new_state_dict = graph.invoke(
            st.session_state.graph_state, st.session_state.graph_config
        )

        st.session_state.graph_state = GraphState(
            prompt=new_state_dict.get("prompt"),
            memory=new_state_dict.get("memory", []),
            collections=new_state_dict.get("collections", []),
            filter=new_state_dict.get("filter"),
            context=new_state_dict.get("context", []),
            response=new_state_dict.get("response"),
            error=new_state_dict.get("error"),
        )

        with st.chat_message("🤖"):
            reply = (
                st.session_state.graph_state.response
                or "[ERROR] Could not get response."
            )
            st.markdown(reply)


if __name__ == "__main__":
    run_app()
