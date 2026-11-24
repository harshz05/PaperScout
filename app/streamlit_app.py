"""Streamlit web interface for PaperScout AI Research Assistant."""

import sys
from pathlib import Path

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# TODO: Swap to retrieval.retriever after integration
from llm.mock_retriever import retrieve_papers
from llm.generator import generate_answer, generate_comparison_table

st.set_page_config(page_title="PaperScout — AI Research Assistant", layout="wide")


def _render_source_papers(papers: list[dict]) -> None:
    """Render retrieved paper metadata and abstracts inside an expander."""
    with st.expander("Retrieved Source Papers"):
        if not papers:
            st.info("No papers matched the current query and category filter.")
            return
        for paper in papers:
            st.markdown(
                f"**{paper['title']}**  \n"
                f"`arXiv:{paper['arxiv_id']}` | Category: `{paper['category']}`"
            )
            st.write(paper["abstract"])
            st.divider()


def main() -> None:
    """Run the PaperScout Streamlit application."""
    st.title("PaperScout — AI Research Assistant")
    st.caption("Explore AI/ML papers with cited Q&A and methodology matrices.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    with st.sidebar:
        st.header("Search & Generation Settings")
        category = st.selectbox(
            "Category",
            options=["All", "cs.CL", "cs.LG", "cs.CV"],
        )
        mode = st.radio(
            "Mode",
            options=["Cited Q&A", "Methodology Matrix"],
        )
        top_k = st.slider(
            "Top-K",
            min_value=1,
            max_value=6,
            value=4,
        )
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and "papers" in message:
                _render_source_papers(message["papers"])

    if prompt := st.chat_input("Ask a research question or request a paper comparison..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        papers = retrieve_papers(prompt, category, top_k)

        with st.chat_message("assistant"):
            try:
                with st.spinner("Generating response..."):
                    if mode == "Cited Q&A":
                        response_text = generate_answer(prompt, papers)
                    else:
                        response_text = generate_comparison_table(papers)
            except Exception as exc:
                st.error(
                    "Unable to reach Ollama (`qwen2.5:3b`). Please verify that the "
                    f"Ollama service is running and the model is available. ({exc})"
                )
                _render_source_papers(papers)
                return

            st.markdown(response_text)
            _render_source_papers(papers)
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": response_text,
                    "papers": papers,
                }
            )


if __name__ == "__main__":
    main()
