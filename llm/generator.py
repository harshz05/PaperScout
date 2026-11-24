"""LLM answer and comparison matrix generation using Ollama."""

from typing import Any

import ollama

MODEL_NAME = "qwen2.5:3b"
INSUFFICIENT_INFO_MSG = (
    "Insufficient information in the retrieved papers to answer this question."
)

QA_SYSTEM_PROMPT = (
    "Answer ONLY using the provided paper abstracts. Cite every claim inline as "
    "[Title (arXiv:ID)]. Never use outside knowledge. If the abstracts do not "
    "contain enough information to answer the question, reply strictly with: "
    "'Insufficient information in the retrieved papers to answer this question.'"
)

MATRIX_SYSTEM_PROMPT = (
    "You are a precise research assistant. Output ONLY a clean Markdown table "
    "comparing the provided papers based strictly on their abstracts. Do not "
    "include any introductory text, concluding remarks, or code fences. "
    "Use the exact columns:\n"
    "| Paper Title | arXiv ID | Core Method | Dataset / Domain | Key Limitation |\n"
    "If an abstract does not specify a field, write 'Not specified in abstract'."
)


def _format_papers_context(papers: list[dict]) -> str:
    """Format retrieved paper dictionaries into a readable prompt context block."""
    sections = []
    for idx, paper in enumerate(papers, start=1):
        sections.append(
            f"Paper {idx}:\n"
            f"Title: {paper.get('title', 'Unknown')}\n"
            f"arXiv ID: {paper.get('arxiv_id', 'Unknown')}\n"
            f"Category: {paper.get('category', 'Unknown')}\n"
            f"Abstract: {paper.get('abstract', '').strip()}"
        )
    return "\n\n".join(sections)


def _extract_content(response: Any) -> str:
    """Extract assistant message text from an Ollama chat response."""
    if isinstance(response, dict):
        return str(response["message"]["content"]).strip()
    return str(response.message.content).strip()


def _call_ollama(system_prompt: str, user_prompt: str) -> str:
    """Send a chat completion request to Ollama using the configured model."""
    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return _extract_content(response)


def generate_answer(question: str, papers: list[dict]) -> str:
    """Generate a cited answer grounded strictly in the retrieved paper abstracts."""
    if not papers:
        return INSUFFICIENT_INFO_MSG

    context = _format_papers_context(papers)
    user_prompt = (
        f"Retrieved Paper Abstracts:\n\n{context}\n\n"
        f"Question: {question}"
    )
    return _call_ollama(QA_SYSTEM_PROMPT, user_prompt)


def generate_comparison_table(papers: list[dict]) -> str:
    """Generate a Markdown comparison matrix across retrieved papers."""
    if not papers:
        return INSUFFICIENT_INFO_MSG

    context = _format_papers_context(papers)
    user_prompt = (
        "Compare the following papers strictly based on their abstracts and "
        "output ONLY a clean Markdown table with columns:\n"
        "| Paper Title | arXiv ID | Core Method | Dataset / Domain | Key Limitation |\n\n"
        f"{context}"
    )
    return _call_ollama(MATRIX_SYSTEM_PROMPT, user_prompt)
